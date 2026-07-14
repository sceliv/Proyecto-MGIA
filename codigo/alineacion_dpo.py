"""
alineacion_dpo.py
-------------------
Fase de alineación por preferencias (DPO) sobre el adaptador LoRA ya
entrenado en entrenar_lora.py. 

FASE 1: Usa el propio modelo ajustado para generar respuestas alternativas
(muestreo con temperatura) sobre una porción del dataset. Cuando la
categoría de diagnóstico generada no coincide con la real (dx del CSV),
esa generación se etiqueta automáticamente como respuesta RECHAZADA, y el
informe de referencia original (correcto) como respuesta ELEGIDA.

FASE 2: Entrena con DPOTrainer sobre esos pares de preferencia, continuando
el ajuste del mismo adaptador LoRA. Para respetar el límite de 6GB de VRAM,
NO se carga un segundo modelo de referencia: se aprovecha que DPOTrainer,
al recibir un PeftModel, usa automáticamente el modelo base con el
adaptador desactivado como referencia implícita.
"""

import os
import json
import random

import torch
from PIL import Image

from datasets import load_dataset, Dataset

from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from peft import PeftModel, LoraConfig
from trl import DPOTrainer, DPOConfig


# ---------------------------------------------------------------------------
# 1. CONFIGURACIÓN INICIAL / CONSTANTES
# ---------------------------------------------------------------------------

RUTA_MODELO_BASE = "Qwen/Qwen2-VL-2B-Instruct"
RUTA_ADAPTADOR_SFT = os.path.join("modelos", "checkpoint_lora")
RUTA_ADAPTADOR_DPO = os.path.join("modelos", "checkpoint_dpo")
RUTA_DATASET_JSONL = os.path.join("datos", "preparados", "dataset_entrenamiento.jsonl")
RUTA_DATASET_PREFERENCIAS = os.path.join("datos", "preparados", "dataset_preferencias.jsonl")

LONGITUD_MAXIMA_TOKENS = 4096
MAXIMO_TOKENS_NUEVOS_GENERACION = 300

# Tamaño de la muestra usada para CONSTRUIR pares de preferencia. La
# generación con sampling es costosa; no es necesario recorrer los 10,015
# ejemplos para obtener suficientes pares útiles de desacuerdo.
TAMANO_MUESTRA_GENERACION_PREFERENCIAS = 800

SEMILLA_ALEATORIA = 42

# Diagnósticos en español, reutilizados para poder detectar qué categoría
# "cree" haber visto el modelo dentro del texto que generó.
DICCIONARIO_DIAGNOSTICOS = {
    "akiec": "Queratosis Actínica / Carcinoma Intraepitelial",
    "bcc": "Carcinoma Basocelular",
    "bkl": "Lesión Queratósica Benigna",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevus Melanocítico (Lunar Benigno)",
    "vasc": "Lesión Vascular",
}


# ---------------------------------------------------------------------------
# FASE 1 — CONSTRUCCIÓN DEL DATASET DE PREFERENCIAS
# ---------------------------------------------------------------------------

def cargar_modelo_para_generacion_muestreada():
    """Carga el modelo base cuantizado + adaptador LoRA de SFT, listo para
    generar con sampling (temperatura > 0), a diferencia de la generación
    determinista que usamos en evaluar_generacion.py."""

    if not os.path.isdir(RUTA_ADAPTADOR_SFT):
        raise FileNotFoundError(
            f"No se encontró el adaptador SFT en: {RUTA_ADAPTADOR_SFT}\n"
            f"Ejecuta primero codigo/entrenar_lora.py."
        )

    print("Cargando processor desde el adaptador SFT...")
    procesador = AutoProcessor.from_pretrained(RUTA_ADAPTADOR_SFT, trust_remote_code=True)
    procesador.tokenizer.model_max_length = LONGITUD_MAXIMA_TOKENS

    configuracion_cuantizacion = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Cargando modelo base en 4 bits para generación de candidatos...")
    modelo_base = Qwen2VLForConditionalGeneration.from_pretrained(
        RUTA_MODELO_BASE,
        quantization_config=configuracion_cuantizacion,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    modelo_con_adaptador = PeftModel.from_pretrained(modelo_base, RUTA_ADAPTADOR_SFT)
    modelo_con_adaptador.eval()

    return modelo_con_adaptador, procesador


def generar_respuesta_con_muestreo(modelo, procesador, ruta_imagen, texto_prompt):
    """Genera una respuesta ALTERNATIVA usando sampling (do_sample=True),
    en vez de generación determinista, para producir variabilidad real
    entre corridas y así poder detectar desacuerdos con la referencia."""
    imagen_pil = Image.open(ruta_imagen).convert("RGB")

    mensajes_usuario = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ruta_imagen},
                {"type": "text", "text": texto_prompt},
            ],
        }
    ]

    texto_formateado = procesador.apply_chat_template(
        mensajes_usuario, tokenize=False, add_generation_prompt=True
    )

    entradas_modelo = procesador(
        text=[texto_formateado],
        images=[[imagen_pil]],
        padding=True,
        truncation=True,
        max_length=LONGITUD_MAXIMA_TOKENS,
        return_tensors="pt",
    ).to(modelo.device)

    with torch.no_grad():
        ids_generados = modelo.generate(
            **entradas_modelo,
            max_new_tokens=MAXIMO_TOKENS_NUEVOS_GENERACION,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
        )

    longitud_entrada = entradas_modelo["input_ids"].shape[1]
    ids_solo_respuesta = ids_generados[:, longitud_entrada:]

    texto_generado = procesador.batch_decode(
        ids_solo_respuesta, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return texto_generado.strip()


def detectar_diagnostico_en_texto(texto):
    """Busca, dentro de un texto generado, cuál nombre de diagnóstico en
    español (de los 7 posibles) aparece mencionado. Retorna el código dx
    correspondiente, o None si no se detecta ninguno con claridad."""
    texto_normalizado = texto.lower()
    for codigo_dx, nombre_es in DICCIONARIO_DIAGNOSTICOS.items():
        if nombre_es.lower() in texto_normalizado:
            return codigo_dx
    return None


def construir_dataset_preferencias():
    """
    FASE 1 completa: recorre una muestra del dataset, genera una respuesta
    alternativa por muestreo, y compara la categoría detectada contra el dx
    real. Cuando difieren, guarda el par {prompt, elegida=referencia,
    rechazada=generación equivocada}. Cuando coinciden, el ejemplo se omite
    (no aporta señal de preferencia).
    """
    print("=" * 70)
    print("FASE 1 — CONSTRUCCIÓN DEL DATASET DE PREFERENCIAS (DPO)")
    print("=" * 70)

    if not os.path.isfile(RUTA_DATASET_JSONL):
        raise FileNotFoundError(f"No se encontró: {RUTA_DATASET_JSONL}")

    registros = []
    with open(RUTA_DATASET_JSONL, mode="r", encoding="utf-8") as archivo:
        for linea in archivo:
            registros.append(json.loads(linea))

    generador_aleatorio = random.Random(SEMILLA_ALEATORIA)
    muestra = generador_aleatorio.sample(
        registros, min(TAMANO_MUESTRA_GENERACION_PREFERENCIAS, len(registros))
    )
    print(f"Muestra seleccionada para generación: {len(muestra)} ejemplos")

    modelo, procesador = cargar_modelo_para_generacion_muestreada()

    pares_preferencia = []
    total_descartados_por_coincidencia = 0
    total_fallidos = 0

    for indice, registro in enumerate(muestra, start=1):
        mensaje_usuario = registro["messages"][0]
        mensaje_asistente = registro["messages"][1]

        ruta_imagen = None
        texto_prompt = None
        for bloque in mensaje_usuario["content"]:
            if bloque["type"] == "image":
                ruta_imagen = bloque["image"]
            elif bloque["type"] == "text":
                texto_prompt = bloque["text"]

        texto_referencia = mensaje_asistente["content"][0]["text"]
        dx_real = detectar_diagnostico_en_texto(texto_referencia)

        if indice % 25 == 0 or indice == 1:
            print(f"[{indice}/{len(muestra)}] Generando candidato para: {ruta_imagen}")

        try:
            texto_generado = generar_respuesta_con_muestreo(
                modelo, procesador, ruta_imagen, texto_prompt
            )
        except Exception as error:
            total_fallidos += 1
            continue

        dx_generado = detectar_diagnostico_en_texto(texto_generado)

        # Solo construimos el par si hay un desacuerdo claro de categoría.
        # Si el modelo generó texto no reconocible o coincide con el real,
        # no hay señal útil de preferencia para este ejemplo.
        if dx_generado is not None and dx_real is not None and dx_generado != dx_real:
            pares_preferencia.append({
                "images": [ruta_imagen],
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": texto_prompt},
                        ],
                    }
                ],
                "chosen": [
                    {"role": "assistant", "content": [{"type": "text", "text": texto_referencia}]}
                ],
                "rejected": [
                    {"role": "assistant", "content": [{"type": "text", "text": texto_generado}]}
                ],
            })
        else:
            total_descartados_por_coincidencia += 1

    print(f"\nPares de preferencia construidos: {len(pares_preferencia)}")
    print(f"Descartados por coincidencia (sin desacuerdo): {total_descartados_por_coincidencia}")
    print(f"Descartados por error de generación: {total_fallidos}")

    os.makedirs(os.path.dirname(RUTA_DATASET_PREFERENCIAS), exist_ok=True)
    with open(RUTA_DATASET_PREFERENCIAS, mode="w", encoding="utf-8") as archivo_salida:
        for par in pares_preferencia:
            archivo_salida.write(json.dumps(par, ensure_ascii=False) + "\n")

    print(f"Dataset de preferencias guardado en: {RUTA_DATASET_PREFERENCIAS}")

    # --- Liberación explícita de VRAM antes de la Fase 2 ---
    del modelo
    torch.cuda.empty_cache()
    print("Modelo de generación liberado de VRAM.\n")

    return len(pares_preferencia)


# ---------------------------------------------------------------------------
# FASE 2 — ENTRENAMIENTO DPO
# ---------------------------------------------------------------------------

class ColadorDPOMultimodal:
    """
    Análogo al collator de entrenar_lora.py, pero adaptado al formato de
    DPOTrainer: recibe lotes con 'images', 'prompt', 'chosen' y 'rejected',
    y abre las imágenes físicas justo a tiempo (Lazy Loading real), igual
    que hicimos en la Fase SFT.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, lote_registros):
        # DPOTrainer con soporte multimodal espera que el propio dataset
        # ya incluya la referencia a imagen bajo la clave 'images' (rutas),
        # y se encarga de resolverlas usando el processor internamente en
        # versiones recientes de trl. Se deja este collator como capa de
        # control explícito por consistencia con el resto del pipeline.
        return lote_registros


def cargar_modelo_para_entrenamiento_dpo():
    """Recarga el modelo base cuantizado y adjunta el adaptador SFT como
    ENTRENABLE (is_trainable=True), para que DPO continúe ajustando esos
    mismos pesos LoRA en lugar de partir de un adaptador nuevo vacío."""

    print("Cargando processor desde el adaptador SFT...")
    procesador = AutoProcessor.from_pretrained(RUTA_ADAPTADOR_SFT, trust_remote_code=True)
    procesador.tokenizer.model_max_length = LONGITUD_MAXIMA_TOKENS

    configuracion_cuantizacion = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Cargando modelo base en 4 bits para entrenamiento DPO...")
    modelo_base = Qwen2VLForConditionalGeneration.from_pretrained(
        RUTA_MODELO_BASE,
        quantization_config=configuracion_cuantizacion,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    modelo_base.config.use_cache = False

    print(f"Adjuntando adaptador SFT como entrenable desde: {RUTA_ADAPTADOR_SFT}...")
    modelo_con_adaptador = PeftModel.from_pretrained(
        modelo_base, RUTA_ADAPTADOR_SFT, is_trainable=True
    )

    return modelo_con_adaptador, procesador


def ejecutar_entrenamiento_dpo():
    print("=" * 70)
    print("FASE 2 — ENTRENAMIENTO DPO SOBRE EL ADAPTADOR LoRA")
    print("=" * 70)

    if not os.path.isfile(RUTA_DATASET_PREFERENCIAS):
        raise FileNotFoundError(
            f"No se encontró el dataset de preferencias en: {RUTA_DATASET_PREFERENCIAS}\n"
            f"Ejecuta primero la Fase 1 (construir_dataset_preferencias)."
        )

    dataset_preferencias = load_dataset("json", data_files=RUTA_DATASET_PREFERENCIAS, split="train")
    print(f"Pares de preferencia disponibles: {len(dataset_preferencias)}")

    if len(dataset_preferencias) == 0:
        print("ERROR: El dataset de preferencias está vacío. No hay nada que entrenar.")
        return

    modelo, procesador = cargar_modelo_para_entrenamiento_dpo()

    argumentos_dpo = DPOConfig(
        output_dir=os.path.join("modelos", "resultados_dpo"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=3,
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        beta=0.1,  # Controla cuán agresivamente se aleja el modelo de la referencia
        max_length=LONGITUD_MAXIMA_TOKENS,
        max_prompt_length=LONGITUD_MAXIMA_TOKENS // 2,
    )

    # NO se pasa ref_model: al recibir un PeftModel, DPOTrainer usa
    # automáticamente el modelo base (adaptador desactivado) como
    # referencia implícita, evitando cargar una segunda copia en VRAM.
    entrenador_dpo = DPOTrainer(
        model=modelo,
        args=argumentos_dpo,
        train_dataset=dataset_preferencias,
        processing_class=procesador,
    )

    # --- Reanudación automática, igual que en entrenar_lora.py ---
    ruta_checkpoints_dpo = argumentos_dpo.output_dir
    ultimo_checkpoint = None
    if os.path.isdir(ruta_checkpoints_dpo):
        checkpoints_existentes = [
            nombre for nombre in os.listdir(ruta_checkpoints_dpo)
            if nombre.startswith("checkpoint-")
        ]
        if checkpoints_existentes:
            ultimo_checkpoint = os.path.join(
                ruta_checkpoints_dpo,
                sorted(checkpoints_existentes, key=lambda n: int(n.split("-")[-1]))[-1]
            )
            print(f"Reanudando entrenamiento DPO desde: {ultimo_checkpoint}")

    print("\nIniciando entrenamiento DPO...\n")
    entrenador_dpo.train(resume_from_checkpoint=ultimo_checkpoint)

    print("\nEntrenamiento DPO finalizado. Guardando adaptador alineado...")
    os.makedirs(RUTA_ADAPTADOR_DPO, exist_ok=True)
    entrenador_dpo.model.save_pretrained(RUTA_ADAPTADOR_DPO)
    procesador.save_pretrained(RUTA_ADAPTADOR_DPO)

    print(f"Adaptador DPO guardado en: {RUTA_ADAPTADOR_DPO}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# FLUJO PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    total_pares = construir_dataset_preferencias()

    if total_pares == 0:
        print(
            "\nADVERTENCIA: No se generó ningún par de preferencia (el modelo "
            "acertó la categoría en todos los casos muestreados, o no se "
            "detectó texto reconocible). Considera aumentar "
            "TAMANO_MUESTRA_GENERACION_PREFERENCIAS o revisar la calidad del "
            "adaptador SFT antes de continuar."
        )
        return

    ejecutar_entrenamiento_dpo()


if __name__ == "__main__":
    main()