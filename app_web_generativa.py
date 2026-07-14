"""
app_web_generativa.py
------------------------
Interfaz web local (Gradio) para el sistema de análisis dermatológico
asistido. Carga el modelo base Qwen2-VL-2B-Instruct en 4 bits y le adjunta
el adaptador LoRA entrenado con SFT (checkpoint_lora), generando informes
dermatológicos en español a partir de una imagen dermatoscópica.

Incluye un guardrail de verificación de dominio: antes de generar cualquier
informe dermatológico, desactiva temporalmente el adaptador LoRA para usar
el modelo base de propósito general y confirmar que la imagen corresponde
efectivamente a una lesión de piel humana.
"""

import os
import torch
from PIL import Image

import gradio as gr

from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from peft import PeftModel


# ---------------------------------------------------------------------------
# 1. CONFIGURACIÓN INICIAL / CONSTANTES
# ---------------------------------------------------------------------------

RUTA_MODELO_BASE = "Qwen/Qwen2-VL-2B-Instruct"
RUTA_ADAPTADOR_SFT = os.path.join("modelos", "checkpoint_lora")

LONGITUD_MAXIMA_TOKENS = 4096
MAXIMO_TOKENS_NUEVOS_GENERACION = 300

PROMPT_VERIFICACION_DOMINIO = (
    "Observa la imagen adjunta. Responde ÚNICAMENTE con la palabra 'SI' o "
    "'NO', sin ningún texto adicional. ¿Esta imagen corresponde a una "
    "fotografía dermatoscópica o clínica de una lesión de piel humana?"
)

AVISO_LEGAL = (
    "⚠️ Esta herramienta es un proyecto académico de apoyo a la detección y "
    "NO reemplaza el diagnóstico de un profesional de la salud. Cualquier "
    "resultado generado debe ser validado por un dermatólogo certificado."
)


# ---------------------------------------------------------------------------
# 2. CARGA DEL MODELO BASE + ADAPTADOR SFT
# ---------------------------------------------------------------------------

def cargar_modelo_y_adaptador():
    """
    Carga el modelo base cuantizado en 4 bits y le adjunta el adaptador
    LoRA entrenado con SFT. Retorna el modelo y el processor listos para
    inferencia.
    """
    if not os.path.isdir(RUTA_ADAPTADOR_SFT):
        raise FileNotFoundError(
            f"No se encontró el adaptador SFT en: {RUTA_ADAPTADOR_SFT}\n"
            f"Ejecuta primero codigo/entrenar_lora.py antes de lanzar la aplicación."
        )

    print("Cargando processor...")
    procesador = AutoProcessor.from_pretrained(RUTA_ADAPTADOR_SFT, trust_remote_code=True)
    procesador.tokenizer.model_max_length = LONGITUD_MAXIMA_TOKENS

    configuracion_cuantizacion = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Cargando modelo base Qwen2-VL-2B-Instruct en 4 bits...")
    modelo_base = Qwen2VLForConditionalGeneration.from_pretrained(
        RUTA_MODELO_BASE,
        quantization_config=configuracion_cuantizacion,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Adjuntando adaptador SFT desde: {RUTA_ADAPTADOR_SFT}...")
    modelo = PeftModel.from_pretrained(modelo_base, RUTA_ADAPTADOR_SFT)
    modelo.eval()

    print("Modelo y adaptador listos para inferencia.")
    return modelo, procesador


# ---------------------------------------------------------------------------
# 3. GUARDRAIL — VERIFICACIÓN DE DOMINIO CON EL MODELO BASE
# ---------------------------------------------------------------------------

def verificar_imagen_en_dominio(modelo, procesador, imagen_pil):
    """
    Guardrail previo a la generación especializada. Desactiva temporalmente
    el adaptador LoRA (modelo.disable_adapter()) para recuperar el
    comportamiento de propósito general del Qwen2-VL base, y le pregunta
    directamente si la imagen corresponde al dominio dermatológico.
    Retorna True si el modelo base confirma que sí, False en caso contrario.
    """
    mensajes_verificacion = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "imagen_cargada"},
                {"type": "text", "text": PROMPT_VERIFICACION_DOMINIO},
            ],
        }
    ]

    texto_formateado = procesador.apply_chat_template(
        mensajes_verificacion, tokenize=False, add_generation_prompt=True
    )

    entradas_modelo = procesador(
        text=[texto_formateado],
        images=[[imagen_pil]],
        padding=True,
        truncation=True,
        max_length=LONGITUD_MAXIMA_TOKENS,
        return_tensors="pt",
    ).to(modelo.device)

    # --- Desactivación temporal del adaptador: comportamiento de modelo base ---
    with modelo.disable_adapter():
        with torch.no_grad():
            ids_generados = modelo.generate(
                **entradas_modelo,
                max_new_tokens=10,  # Solo necesitamos "SI" o "NO"
                do_sample=False,
            )

    longitud_entrada = entradas_modelo["input_ids"].shape[1]
    ids_solo_respuesta = ids_generados[:, longitud_entrada:]

    texto_respuesta = procesador.batch_decode(
        ids_solo_respuesta, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip().upper()

    print(f"[Guardrail] Respuesta del modelo base sobre el dominio de la imagen: '{texto_respuesta}'")

    return texto_respuesta.startswith("SI") or texto_respuesta.startswith("SÍ")


# ---------------------------------------------------------------------------
# 4. GENERACIÓN DE INFORME
# ---------------------------------------------------------------------------

def generar_informe(modelo, procesador, imagen_pil, texto_prompt):
    """
    Genera un informe determinista (sin sampling, para resultados
    reproducibles en la demo) usando el adaptador SFT activo.
    """
    mensajes_usuario = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "imagen_cargada"},
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
            do_sample=False,
        )

    longitud_entrada = entradas_modelo["input_ids"].shape[1]
    ids_solo_respuesta = ids_generados[:, longitud_entrada:]

    texto_generado = procesador.batch_decode(
        ids_solo_respuesta, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return texto_generado.strip()


# ---------------------------------------------------------------------------
# 5. CONSTRUCCIÓN DE LA INTERFAZ (GRADIO)
# ---------------------------------------------------------------------------

def construir_interfaz(modelo, procesador):

    def manejar_analisis(imagen_subida, edad, sexo, localizacion):
        if imagen_subida is None:
            return "Por favor, sube una imagen dermatoscópica antes de analizar."

        imagen_pil = imagen_subida.convert("RGB")

        # --- GUARDRAIL: verificación de dominio antes de generar el informe ---
        print("Verificando si la imagen corresponde al dominio dermatológico...")
        imagen_es_valida = verificar_imagen_en_dominio(modelo, procesador, imagen_pil)

        if not imagen_es_valida:
            return (
                "⚠️ La imagen proporcionada no parece corresponder a una "
                "fotografía dermatoscópica de una lesión de piel. Este sistema "
                "está diseñado exclusivamente para el análisis de imágenes "
                "clínicas de piel. Por favor, sube una imagen dermatoscópica válida."
            )

        fragmento_edad = f"de {int(edad)} años" if edad else "de edad no especificada"
        sexo_es = sexo if sexo else "no especificado"
        localizacion_es = localizacion if localizacion else "una zona no especificada"

        texto_prompt = (
            f"Analiza la siguiente imagen dermatoscópica. Corresponde a un "
            f"paciente de sexo {sexo_es}, {fragmento_edad}, con la lesión "
            f"localizada en {localizacion_es}. Proporciona un informe "
            f"dermatológico detallado en español, indicando el diagnóstico "
            f"más probable, su naturaleza (benigna o maligna) y una "
            f"recomendación clínica general."
        )

        return generar_informe(modelo, procesador, imagen_pil, texto_prompt)

    with gr.Blocks(title="Análisis Dermatológico Asistido - Proyecto MGIA") as interfaz:
        gr.Markdown("# 🔬 Sistema de Análisis Dermatológico Asistido por IA")
        gr.Markdown(AVISO_LEGAL)

        with gr.Row():
            with gr.Column(scale=1):
                imagen_entrada = gr.Image(type="pil", label="Imagen dermatoscópica")
                edad_entrada = gr.Number(label="Edad del paciente", precision=0)
                sexo_entrada = gr.Dropdown(
                    choices=["masculino", "femenino"], label="Sexo del paciente"
                )
                localizacion_entrada = gr.Dropdown(
                    choices=[
                        "la espalda", "la extremidad inferior", "el tronco",
                        "la extremidad superior", "el abdomen", "el rostro",
                        "el pecho", "el pie", "el cuello", "el cuero cabelludo",
                        "la mano", "la oreja",
                    ],
                    label="Localización de la lesión",
                )
                boton_analizar = gr.Button("Analizar imagen", variant="primary")

            with gr.Column(scale=1):
                salida_informe = gr.Textbox(
                    label="Informe generado por el modelo",
                    lines=14,
                )

        boton_analizar.click(
            fn=manejar_analisis,
            inputs=[imagen_entrada, edad_entrada, sexo_entrada, localizacion_entrada],
            outputs=[salida_informe],
        )

    return interfaz


# ---------------------------------------------------------------------------
# 6. FLUJO PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("INICIANDO APLICACIÓN WEB - Análisis Dermatológico Asistido")
    print("=" * 70)

    modelo, procesador = cargar_modelo_y_adaptador()
    interfaz = construir_interfaz(modelo, procesador)

    interfaz.launch(server_name="127.0.0.1", server_port=7860, share=True)


if __name__ == "__main__":
    main()