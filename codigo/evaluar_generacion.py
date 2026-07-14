"""
evaluar_generacion.py
-----------------------
Carga el modelo base Qwen2-VL-2B-Instruct cuantizado en 4 bits, le adjunta el
adaptador LoRA entrenado en entrenar_lora.py, y ejecuta inferencia real
(generación de texto) sobre una muestra de imágenes dermatoscópicas.
Calcula métricas ROUGE y BLEU comparando las generaciones del modelo contra
los informes de referencia construidos en preparar_dataset.py.
"""

import os
import json
import argparse

import torch
from PIL import Image

from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from peft import PeftModel

from rouge_score import rouge_scorer
import sacrebleu


# 1. CONFIGURACIÓN INICIAL / CONSTANTES

RUTA_MODELO_BASE = "Qwen/Qwen2-VL-2B-Instruct"
RUTA_ADAPTADOR_LORA = os.path.join("modelos", "checkpoint_lora")
RUTA_DATASET_JSONL = os.path.join("datos", "preparados", "dataset_entrenamiento.jsonl")
RUTA_REPORTE_SALIDA = os.path.join("modelos", "reporte_evaluacion.json")

NUMERO_EJEMPLOS_EVALUACION = 20  # Muestra reducida: la generación en 6GB es lenta
MAXIMO_TOKENS_NUEVOS_GENERACION = 300

LONGITUD_MAXIMA_TOKENS = 4096


# 2. CARGA DEL MODELO BASE + ADAPTADOR LoRA

def cargar_modelo_para_inferencia():
    """
    Carga el modelo base cuantizado en 4 bits (idéntica configuración a la
    usada en entrenamiento, para evitar discrepancias numéricas) y le adjunta
    el adaptador LoRA entrenado, dejando el modelo listo en modo evaluación.
    """
    if not os.path.isdir(RUTA_ADAPTADOR_LORA):
        raise FileNotFoundError(
            f"No se encontró el adaptador LoRA en: {RUTA_ADAPTADOR_LORA}\n"
            f"Ejecuta primero codigo/entrenar_lora.py hasta que finalice el entrenamiento."
        )

    print("Cargando processor desde el adaptador LoRA guardado...")
    procesador = AutoProcessor.from_pretrained(RUTA_ADAPTADOR_LORA, trust_remote_code=True)
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

    print(f"Adjuntando adaptador LoRA desde: {RUTA_ADAPTADOR_LORA}...")
    modelo_con_adaptador = PeftModel.from_pretrained(modelo_base, RUTA_ADAPTADOR_LORA)
    modelo_con_adaptador.eval()  # Modo evaluación: desactiva dropout, entre otros

    return modelo_con_adaptador, procesador


# 3. GENERACIÓN REAL DE TEXTO (INFERENCIA)

def generar_informe(modelo, procesador, ruta_imagen, texto_prompt_usuario):
    """
    Ejecuta inferencia real: dada una imagen y un prompt, el modelo GENERA
    (no imita) un informe dermatológico en español, token por token.
    """
    imagen_pil = Image.open(ruta_imagen).convert("RGB")

    mensajes_solo_usuario = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ruta_imagen},
                {"type": "text", "text": texto_prompt_usuario},
            ],
        }
    ]

    # add_generation_prompt=True: le indica al modelo que debe EMPEZAR a
    # responder aquí, en lugar de simplemente formatear una conversación ya
    # completa como hacíamos durante el entrenamiento.
    texto_formateado = procesador.apply_chat_template(
        mensajes_solo_usuario,
        tokenize=False,
        add_generation_prompt=True,
    )

    entradas_modelo = procesador(
        text=[texto_formateado],
        images=[[imagen_pil]],
        padding=True,
        return_tensors="pt",
    ).to(modelo.device)

    with torch.no_grad():
        ids_generados = modelo.generate(
            **entradas_modelo,
            max_new_tokens=MAXIMO_TOKENS_NUEVOS_GENERACION,
            do_sample=False,  # Generación determinista: reproducible para evaluación
        )

    # Recortamos el prompt de entrada, conservando solo los tokens NUEVOS
    # generados por el modelo (la longitud de entrada varía por imagen).
    longitud_entrada = entradas_modelo["input_ids"].shape[1]
    ids_solo_respuesta = ids_generados[:, longitud_entrada:]

    texto_generado = procesador.batch_decode(
        ids_solo_respuesta,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return texto_generado.strip()


# 4. CARGA DE LA MUESTRA DE EVALUACIÓN

def cargar_muestra_evaluacion(numero_ejemplos):
    """
    Toma los últimos N registros del dataset preparado como muestra de
    evaluación. Extrae la ruta de imagen, el prompt original y el informe
    de referencia (respuesta del asistente) de cada uno.

    NOTA IMPORTANTE: dado que entrenar_lora.py entrena sobre el dataset
    completo (sin partición train/test), esta muestra NO es un conjunto de
    prueba verdaderamente independiente. Sirve como verificación cualitativa
    y de consistencia del formato de salida, no como medición rigurosa de
    generalización. Quedará documentado como una mejora pendiente.
    """
    if not os.path.isfile(RUTA_DATASET_JSONL):
        raise FileNotFoundError(f"No se encontró el dataset preparado en: {RUTA_DATASET_JSONL}")

    registros = []
    with open(RUTA_DATASET_JSONL, mode="r", encoding="utf-8") as archivo:
        for linea in archivo:
            registros.append(json.loads(linea))

    muestra = registros[-numero_ejemplos:]

    ejemplos_evaluacion = []
    for registro in muestra:
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

        ejemplos_evaluacion.append({
            "ruta_imagen": ruta_imagen,
            "prompt": texto_prompt,
            "referencia": texto_referencia,
        })

    return ejemplos_evaluacion

# 5. CÁLCULO DE MÉTRICAS (ROUGE + BLEU)

def calcular_metricas(texto_referencia, texto_generado, calculador_rouge):
    """
    Calcula ROUGE-1, ROUGE-2, ROUGE-L y BLEU para un par (referencia, generado).
    Tanto rouge_score como sacrebleu operan a nivel de superposición de
    tokens/n-gramas y son agnósticos al idioma, por lo que funcionan
    correctamente sobre texto en español sin configuración adicional.
    """
    puntuaciones_rouge = calculador_rouge.score(texto_referencia, texto_generado)

    puntuacion_bleu = sacrebleu.sentence_bleu(
        texto_generado,
        [texto_referencia],
    ).score

    return {
        "rouge1": round(puntuaciones_rouge["rouge1"].fmeasure, 4),
        "rouge2": round(puntuaciones_rouge["rouge2"].fmeasure, 4),
        "rougeL": round(puntuaciones_rouge["rougeL"].fmeasure, 4),
        "bleu": round(puntuacion_bleu, 4),
    }


# ---------------------------------------------------------------------------
# 6. FLUJO PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    parser_argumentos = argparse.ArgumentParser(
        description="Evalúa el modelo Qwen2-VL + LoRA generando informes dermatológicos."
    )
    parser_argumentos.add_argument(
        "--imagen", type=str, default=None,
        help="Ruta a una única imagen para generar un informe puntual (modo demo, sin métricas)."
    )
    argumentos = parser_argumentos.parse_args()

    print("=" * 70)
    print("EVALUACIÓN DE GENERACIÓN - Qwen2-VL-2B-Instruct + LoRA (Dermatología)")
    print("=" * 70)

    modelo, procesador = cargar_modelo_para_inferencia()

    # --- Modo demo: una sola imagen indicada por línea de comandos ---
    if argumentos.imagen is not None:
        if not os.path.isfile(argumentos.imagen):
            print(f"ERROR: No se encontró la imagen: {argumentos.imagen}")
            return

        prompt_generico = (
            "Analiza la siguiente imagen dermatoscópica. Proporciona un informe "
            "dermatológico detallado en español, indicando el diagnóstico más "
            "probable, su naturaleza (benigna o maligna) y una recomendación clínica general."
        )

        print(f"\nGenerando informe para: {argumentos.imagen}\n")
        texto_generado = generar_informe(modelo, procesador, argumentos.imagen, prompt_generico)

        print("-" * 70)
        print("INFORME GENERADO POR EL MODELO:")
        print("-" * 70)
        print(texto_generado)
        print("-" * 70)
        return

    # --- Modo evaluación: muestra del dataset + métricas ROUGE/BLEU ---
    print(f"\nCargando muestra de evaluación ({NUMERO_EJEMPLOS_EVALUACION} ejemplos)...")
    muestra_evaluacion = cargar_muestra_evaluacion(NUMERO_EJEMPLOS_EVALUACION)
    print(f"Ejemplos cargados: {len(muestra_evaluacion)}")

    calculador_rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,  # El stemmer por defecto de rouge_score está pensado para inglés
    )

    resultados_por_ejemplo = []
    acumulador_metricas = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0}

    for indice, ejemplo in enumerate(muestra_evaluacion, start=1):
        print(f"\n[{indice}/{len(muestra_evaluacion)}] Generando informe para: {ejemplo['ruta_imagen']}")

        try:
            texto_generado = generar_informe(
                modelo, procesador, ejemplo["ruta_imagen"], ejemplo["prompt"]
            )
        except Exception as error:
            print(f"  ADVERTENCIA: Falló la generación para este ejemplo. Se omite. Detalle: {error}")
            continue

        metricas = calcular_metricas(ejemplo["referencia"], texto_generado, calculador_rouge)

        print(f"  ROUGE-1: {metricas['rouge1']} | ROUGE-2: {metricas['rouge2']} | "
              f"ROUGE-L: {metricas['rougeL']} | BLEU: {metricas['bleu']}")

        for clave in acumulador_metricas:
            acumulador_metricas[clave] += metricas[clave]

        resultados_por_ejemplo.append({
            "ruta_imagen": ejemplo["ruta_imagen"],
            "referencia": ejemplo["referencia"],
            "generado": texto_generado,
            "metricas": metricas,
        })

    total_ejemplos_validos = len(resultados_por_ejemplo)
    if total_ejemplos_validos == 0:
        print("\nERROR: Ningún ejemplo pudo evaluarse correctamente.")
        return

    metricas_promedio = {
        clave: round(valor / total_ejemplos_validos, 4)
        for clave, valor in acumulador_metricas.items()
    }

    print("\n" + "=" * 70)
    print("RESUMEN DE MÉTRICAS PROMEDIO")
    print("=" * 70)
    print(f"Ejemplos evaluados: {total_ejemplos_validos}")
    print(f"ROUGE-1 promedio:   {metricas_promedio['rouge1']}")
    print(f"ROUGE-2 promedio:   {metricas_promedio['rouge2']}")
    print(f"ROUGE-L promedio:   {metricas_promedio['rougeL']}")
    print(f"BLEU promedio:      {metricas_promedio['bleu']}")

    os.makedirs(os.path.dirname(RUTA_REPORTE_SALIDA), exist_ok=True)
    with open(RUTA_REPORTE_SALIDA, mode="w", encoding="utf-8") as archivo_reporte:
        json.dump({
            "metricas_promedio": metricas_promedio,
            "resultados_por_ejemplo": resultados_por_ejemplo,
        }, archivo_reporte, ensure_ascii=False, indent=2)

    print(f"\nReporte detallado guardado en: {RUTA_REPORTE_SALIDA}")
    print("=" * 70)


if __name__ == "__main__":
    main()