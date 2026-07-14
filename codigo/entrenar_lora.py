"""
entrenar_lora.py
-----------------
Fine-tuning con QLoRA de Qwen2-VL-2B-Instruct sobre el dataset dermatológico
generado por preparar_dataset.py, optimizado para operar dentro de 6GB de VRAM
(RTX 3050) mediante cuantización de 4 bits, batch size mínimo con acumulación
de gradientes, y un Data Collator multimodal con carga perezosa de imágenes.
"""

import os
import torch
from PIL import Image

from datasets import load_dataset

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    TrainerCallback,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


# 1. CONFIGURACIÓN INICIAL DEL PROYECTO

RUTA_MODELO_BASE = "Qwen/Qwen2-VL-2B-Instruct"
RUTA_DATASET_JSONL = os.path.join("datos", "preparados", "dataset_entrenamiento.jsonl")
RUTA_SALIDA_ADAPTADOR = os.path.join("modelos", "checkpoint_lora")

TOTAL_IMAGENES_ESPERADAS = 10015

BATCH_SIZE_POR_DISPOSITIVO = 1
PASOS_ACUMULACION_GRADIENTE = 8
LOGGING_STEPS = 5

LONGITUD_MAXIMA_TOKENS = 4096


# 2. CALLBACK DE PROGRESO DE IMÁGENES

class CallbackProgresoImagenes(TrainerCallback):
    """
    Traduce el 'global_step' interno del Trainer a un conteo real de imágenes
    físicas procesadas por la GPU, considerando que cada paso de optimización
    equivale a (batch_size_por_dispositivo * pasos_acumulacion_gradiente)
    imágenes efectivamente vistas.
    """

    def __init__(self, batch_size_por_dispositivo, pasos_acumulacion_gradiente, total_imagenes_dataset):
        self.imagenes_por_paso_optimizacion = batch_size_por_dispositivo * pasos_acumulacion_gradiente
        self.total_imagenes_dataset = total_imagenes_dataset

    def on_log(self, args, state, control, logs=None, **kwargs):
        imagenes_analizadas = state.global_step * self.imagenes_por_paso_optimizacion
        imagenes_analizadas = min(imagenes_analizadas, self.total_imagenes_dataset)

        print(
            f"[Progreso]: {imagenes_analizadas} / {self.total_imagenes_dataset} imágenes analizadas "
            f"(paso global: {state.global_step})"
        )
        return control
    

# 3. DATA COLLATOR MULTIMODAL AVANZADO (LAZY LOADING REAL)

class ColadorMultimodalQwenVL:
    """
    Recibe un lote de registros LIGEROS (solo rutas de imagen + texto), y
    únicamente en este punto —justo antes de entrar a la GPU— abre las
    imágenes físicas con PIL. Esto garantiza Lazy Loading verdadero: nunca
    se mantienen imágenes decodificadas en memoria fuera de este paso.

    Usa obligatoriamente processor.apply_chat_template(..., tokenize=False)
    para que los tokens espaciales <|image_pad|> se inyecten de forma nativa
    según la plantilla oficial de Qwen2-VL, evitando desalineaciones entre
    los tokens de texto y los tensores de visión.
    """

    def __init__(self, processor, longitud_maxima):
        self.processor = processor
        self.longitud_maxima = longitud_maxima

    def __call__(self, lote_registros):
        textos_formateados = []
        listas_imagenes = []

        for registro in lote_registros:
            mensajes = registro["messages"]

            # Extraer la(s) ruta(s) de imagen definidas en el mensaje del usuario
            imagenes_del_registro = []
            for mensaje in mensajes:
                for bloque_contenido in mensaje["content"]:
                    if bloque_contenido.get("type") == "image":
                        ruta_imagen = bloque_contenido["image"]
                        # --- LAZY LOADING VERDADERO: apertura justo a tiempo ---
                        imagen_pil = Image.open(ruta_imagen).convert("RGB")
                        imagenes_del_registro.append(imagen_pil)

            # Renderiza la plantilla de chat oficial de Qwen2-VL (texto plano,
            # incluyendo los marcadores de imagen), sin tokenizar todavía.
            texto_formateado = self.processor.apply_chat_template(
                mensajes,
                tokenize=False,
                add_generation_prompt=False,
            )

            textos_formateados.append(texto_formateado)
            listas_imagenes.append(imagenes_del_registro)

        # El processor de Qwen2-VL tokeniza el texto Y procesa las imágenes
        # en un solo paso, generando pixel_values / image_grid_thw alineados
        # correctamente con los tokens <|image_pad|> del texto.
        lote_procesado = self.processor(
            text=textos_formateados,
            images=listas_imagenes,
            padding=True,
            truncation=True,
            max_length=self.longitud_maxima,
            return_tensors="pt",
        )

        # Etiquetas para el cálculo de la pérdida: clon de input_ids, con el
        # padding enmascarado (-100) para que no contribuya al loss.
        etiquetas = lote_procesado["input_ids"].clone()
        id_token_relleno = self.processor.tokenizer.pad_token_id
        etiquetas[etiquetas == id_token_relleno] = -100
        lote_procesado["labels"] = etiquetas

        return lote_procesado


# 4. CARGA Y CUANTIZACIÓN EXTREMA (4 BITS)

def cargar_modelo_y_procesador_cuantizados():
    print("Cargando processor y aplicando límite de tokens (model_max_length=4096)...")
    procesador = AutoProcessor.from_pretrained(RUTA_MODELO_BASE, trust_remote_code=True)

    # --- Prevención obligatoria del truncamiento por defecto de SFTTrainer ---
    procesador.tokenizer.model_max_length = LONGITUD_MAXIMA_TOKENS

    configuracion_cuantizacion = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Cargando modelo base Qwen2-VL-2B-Instruct en 4 bits (nf4 + double quant)...")
    modelo = Qwen2VLForConditionalGeneration.from_pretrained(
        RUTA_MODELO_BASE,
        quantization_config=configuracion_cuantizacion,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    modelo.config.use_cache = False  # Requerido junto a gradient checkpointing

    return modelo, procesador


# 5. PARÁMETROS LoRA E INICIALIZACIÓN DEL TRAINER

def preparar_modelo_con_lora(modelo):
    print("Preparando modelo cuantizado para entrenamiento k-bit (QLoRA)...")
    modelo = prepare_model_for_kbit_training(modelo, use_gradient_checkpointing=True)

    configuracion_lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    modelo = get_peft_model(modelo, configuracion_lora)
    modelo.print_trainable_parameters()

    return modelo


def construir_trainer(modelo, procesador, dataset_ligero):
    argumentos_entrenamiento = TrainingArguments(
        output_dir=os.path.join("modelos", "resultados_entrenamiento"),
        per_device_train_batch_size=BATCH_SIZE_POR_DISPOSITIVO,
        gradient_accumulation_steps=PASOS_ACUMULACION_GRADIENTE,
        num_train_epochs=1,
        fp16=False,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=50,          # guarda un checkpoint cada 50 pasos de optimización
        save_total_limit=3,     # conserva solo los 3 checkpoints más recientes (para no llenar el disco)
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,  # Crítico: preserva las columnas 'messages' crudas
    )

    colador_personalizado = ColadorMultimodalQwenVL(
        processor=procesador,
        longitud_maxima=LONGITUD_MAXIMA_TOKENS,
    )

    callback_progreso = CallbackProgresoImagenes(
        batch_size_por_dispositivo=BATCH_SIZE_POR_DISPOSITIVO,
        pasos_acumulacion_gradiente=PASOS_ACUMULACION_GRADIENTE,
        total_imagenes_dataset=TOTAL_IMAGENES_ESPERADAS,
    )

    # Constructor limpio: SIN max_seq_length ni argumentos de longitud que
    # entren en conflicto con la tokenización ya resuelta en el collator.
    entrenador = SFTTrainer(
        model=modelo,
        args=argumentos_entrenamiento,
        train_dataset=dataset_ligero,
        data_collator=colador_personalizado,
        processing_class=procesador.tokenizer,
        callbacks=[callback_progreso],
    )

    return entrenador


# 6. FLUJO PRINCIPAL Y GUARDADO DEL ADAPTADOR

def main():
    print("=" * 70)
    print("ENTRENAMIENTO QLoRA - Qwen2-VL-2B-Instruct (Dermatología HAM10000)")
    print("=" * 70)

    if not os.path.isfile(RUTA_DATASET_JSONL):
        print(f"ERROR: No se encontró el dataset preparado en: {RUTA_DATASET_JSONL}")
        print("Ejecuta primero codigo/preparar_dataset.py")
        return

    print("Cargando dataset LIGERO (solo rutas + texto, sin decodificar imágenes)...")
    dataset_ligero = load_dataset("json", data_files=RUTA_DATASET_JSONL, split="train")
    print(f"Registros disponibles para entrenamiento: {len(dataset_ligero)}")

    modelo, procesador = cargar_modelo_y_procesador_cuantizados()
    modelo = preparar_modelo_con_lora(modelo)

    entrenador = construir_trainer(modelo, procesador, dataset_ligero)

    # Detección automática de checkpoints previos para reanudar sin perder progreso
    ruta_checkpoints = os.path.join("modelos", "resultados_entrenamiento")
    ultimo_checkpoint = None
    if os.path.isdir(ruta_checkpoints):
        checkpoints_existentes = [
            nombre for nombre in os.listdir(ruta_checkpoints)
            if nombre.startswith("checkpoint-")
        ]
        if checkpoints_existentes:
            ultimo_checkpoint = os.path.join(
                ruta_checkpoints,
                sorted(checkpoints_existentes, key=lambda n: int(n.split("-")[-1]))[-1]
            )
            print(f"Se detectó un checkpoint previo. Reanudando desde: {ultimo_checkpoint}")
        else:
            print("No se encontraron checkpoints previos. Iniciando entrenamiento desde cero.")
    else:
        print("No existe carpeta de resultados previa. Iniciando entrenamiento desde cero.")

    print("\nIniciando entrenamiento...\n")
    entrenador.train(resume_from_checkpoint=ultimo_checkpoint)

    print("\nEntrenamiento finalizado. Guardando únicamente el adaptador LoRA...")
    os.makedirs(RUTA_SALIDA_ADAPTADOR, exist_ok=True)
    entrenador.model.save_pretrained(RUTA_SALIDA_ADAPTADOR)
    procesador.save_pretrained(RUTA_SALIDA_ADAPTADOR)

    print(f"Adaptador LoRA guardado exitosamente en: {RUTA_SALIDA_ADAPTADOR}")
    print("=" * 70)


if __name__ == "__main__":
    main()