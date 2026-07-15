"""
app_web_generativa.py
------------------------
Interfaz web local (Gradio) para el sistema de análisis dermatológico
asistido. Carga el modelo base Qwen2-VL-2B-Instruct en 4 bits y le adjunta
el adaptador LoRA entrenado con SFT (checkpoint_lora), generando informes
dermatológicos en español a partir de una imagen dermatoscópica.
"""

import os
import torch
from PIL import Image

import gradio as gr

from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from peft import PeftModel

# Importaciones para la generación del PDF estético
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


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

    with modelo.disable_adapter():
        with torch.no_grad():
            ids_generados = modelo.generate(
                **entradas_modelo,
                max_new_tokens=10,
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
# Función para generar PDF estético con ReportLab (Función 3 mejorada)
# ---------------------------------------------------------------------------

def guardar_informe_en_pdf(texto_informe, edad, sexo, localizacion):
    """
    Genera un archivo PDF estético estructurado profesionalmente,
    con formato de reporte clínico, incluyendo los metadatos del paciente.
    """
    if not texto_informe or "Por favor, sube una imagen" in texto_informe or "⚠️" in texto_informe:
        return None
    
    ruta_pdf = "informe_clinico_dermatologico.pdf"
    
    # Configuración del documento
    doc = SimpleDocTemplate(
        ruta_pdf,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Estilos personalizados estéticos
    estilo_titulo = ParagraphStyle(
        'TituloClinico',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1a365d'), # Azul oscuro institucional
        spaceAfter=15,
        alignment=1 # Centrado
    )
    
    estilo_subtitulo = ParagraphStyle(
        'SubtituloClinico',
        parent=styles['Normal'],
        fontSize=10,
        leading=12,
        textColor=colors.HexColor('#4a5568'),
        alignment=1,
        spaceAfter=20
    )

    estilo_encabezado_seccion = ParagraphStyle(
        'SeccionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2c5282'),
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    estilo_texto = ParagraphStyle(
        'TextoInforme',
        parent=styles['BodyText'],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor('#2d3748')
    )
    
    estilo_descargo = ParagraphStyle(
        'DescargoLegal',
        parent=styles['Normal'],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#e53e3e'), # Rojo discreto pero visible
        spaceBefore=30,
        alignment=1
    )

    # 1. Cabecera del Reporte
    story.append(Paragraph("REPORTE DE ANÁLISIS DERMATOLÓGICO ASISTIDO POR IA", estilo_titulo))
    story.append(Paragraph("Proyecto MGIA — Sistema Generativo Multimodal Local", estilo_subtitulo))
    story.append(Spacer(1, 10))
    
    # 2. Tabla con Datos del Paciente (Metadatos)
    datos_tabla = [
        [Paragraph("<b>PARÁMETRO CLÍNICO</b>", estilo_texto), Paragraph("<b>INFORMACIÓN REGISTRADA</b>", estilo_texto)],
        [Paragraph("Edad del Paciente", estilo_texto), Paragraph(f"{int(edad)} años" if edad else "No especificada", estilo_texto)],
        [Paragraph("Sexo del Paciente", estilo_texto), Paragraph(sexo.capitalize() if sexo else "No especificado", estilo_texto)],
        [Paragraph("Localización de la Lesión", estilo_texto), Paragraph(localizacion.capitalize() if localizacion else "No especificada", estilo_texto)],
    ]
    
    tabla_metadatos = Table(datos_tabla, colWidths=[200, 300])
    tabla_metadatos.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#edf2f7')),
        ('TEXTCOLOR', (0,0), (1,0), colors.HexColor('#2d3748')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7fafc')])
    ]))
    
    story.append(Paragraph("Metadatos del Paciente", estilo_encabezado_seccion))
    story.append(tabla_metadatos)
    story.append(Spacer(1, 15))
    
    # 3. Sección del Informe Clínico Generado
    story.append(Paragraph("Resultados del Análisis de Imagen", estilo_encabezado_seccion))
    
    # Reemplazar saltos de línea para que ReportLab los renderice correctamente
    parrafos_informe = texto_informe.split("\n")
    for p in parrafos_informe:
        if p.strip():
            story.append(Paragraph(p.strip(), estilo_texto))
            story.append(Spacer(1, 6))
            
    # 4. Descargo Legal de Responsabilidad Obligatorio
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "<b>Aviso legal importante:</b> Este informe clínico es generado por un sistema automatizado de inteligencia artificial "
        "con propósitos estrictamente académicos y de apoyo a la detección. No constituye un diagnóstico médico definitivo "
        "y debe ser evaluado, interpretado y validado en su totalidad por un dermatólogo colegiado y certificado.", 
        estilo_descargo
    ))
    
    # Construcción final del documento
    doc.build(story)
    return ruta_pdf


# ---------------------------------------------------------------------------
# 5. CONSTRUCCIÓN DE LA INTERFAZ (GRADIO)
# ---------------------------------------------------------------------------

def construir_interfaz(modelo, procesador):

    # Rutas locales a imágenes reales de tu dataset (HAM10000_images_part_1)
    # Si estas imágenes específicas no existen, cámbialas por el nombre de cualquier archivo .jpg que tengas ahí
    ruta_ejemplo_1 = os.path.join("datos", "crudos", "HAM10000_images_part_1", "ISIC_0024306.jpg")
    ruta_ejemplo_2 = os.path.join("datos", "crudos", "HAM10000_images_part_1", "ISIC_0024307.jpg")

    # Mecanismo de respaldo por seguridad (si no se encuentran los archivos del dataset para la demo)
    if not os.path.exists(ruta_ejemplo_1):
        print(f"⚠️ Imagen de ejemplo 1 no encontrada en: {ruta_ejemplo_1}. Creando marcador temporal.")
        carpeta_preparados = os.path.join("datos", "preparados")
        os.makedirs(carpeta_preparados, exist_ok=True)
        ruta_ejemplo_1 = os.path.join(carpeta_preparados, "ejemplo_espalda.jpg")
        if not os.path.exists(ruta_ejemplo_1):
            Image.new('RGB', (150, 150), color='#2c5282').save(ruta_ejemplo_1)
        
    if not os.path.exists(ruta_ejemplo_2):
        print(f"⚠️ Imagen de ejemplo 2 no encontrada en: {ruta_ejemplo_2}. Creando marcador temporal.")
        carpeta_preparados = os.path.join("datos", "preparados")
        os.makedirs(carpeta_preparados, exist_ok=True)
        ruta_ejemplo_2 = os.path.join(carpeta_preparados, "ejemplo_rostro.jpg")
        if not os.path.exists(ruta_ejemplo_2):
            Image.new('RGB', (150, 150), color='#2b6cb0').save(ruta_ejemplo_2)

    def manejar_analisis(imagen_subida, edad, sexo, localizacion):
        if imagen_subida is None:
            return "Por favor, sube una imagen dermatoscópica antes de analizar."

        imagen_pil = imagen_subida.convert("RGB")

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
                edad_entrada = gr.Number(label="Edad del paciente", precision=0, value=0)
                sexo_entrada = gr.Dropdown(
                    choices=["masculino", "femenino"], label="Sexo del paciente", value="masculino"
                )
                localizacion_entrada = gr.Dropdown(
                    choices=[
                        "la espalda", "la extremidad inferior", "el tronco",
                        "la extremidad superior", "el abdomen", "el rostro",
                        "el pecho", "el pie", "el cuello", "el cuero cabelludo",
                        "la mano", "la oreja",
                    ],
                    label="Localización de la lesión",
                    value="la espalda"
                )
                
                with gr.Row():
                    boton_analizar = gr.Button("Analizar imagen", variant="primary")
                    boton_limpiar = gr.ClearButton(
                        value="Limpiar campos",
                        variant="secondary"
                    )

            with gr.Column(scale=1):
                salida_informe = gr.Textbox(
                    label="Informe generado por el modelo",
                    lines=14,
                )
                archivo_descarga = gr.File(label="Archivo PDF del informe clínico", interactive=False)
                boton_descargar = gr.Button("⬇️ Generar reporte PDF", variant="secondary")

        # Ejemplos de prueba utilizando las rutas locales del dataset original
        gr.Markdown("### 💡 Ejemplos de prueba")
        gr.Examples(
            examples=[
                [ruta_ejemplo_1, 45, "masculino", "la espalda"],
                [ruta_ejemplo_2, 29, "masculino", "la mano"]
            ],
            inputs=[imagen_entrada, edad_entrada, sexo_entrada, localizacion_entrada],
            label="Selecciona un ejemplo para rellenar los campos automáticamente"
        )

        # Configurar el botón de Limpiar
        boton_limpiar.add([imagen_entrada, edad_entrada, sexo_entrada, localizacion_entrada, salida_informe, archivo_descarga])

        # Evento principal: Analizar
        boton_analizar.click(
            fn=manejar_analisis,
            inputs=[imagen_entrada, edad_entrada, sexo_entrada, localizacion_entrada],
            outputs=[salida_informe],
            show_progress="full"
        )

        # Evento adicional para crear el reporte estético en formato PDF
        boton_descargar.click(
            fn=guardar_informe_en_pdf,
            inputs=[salida_informe, edad_entrada, sexo_entrada, localizacion_entrada],
            outputs=[archivo_descarga]
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