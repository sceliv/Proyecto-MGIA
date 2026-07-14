"""
preparar_dataset.py
--------------------
Lee HAM10000_metadata.csv, localiza físicamente cada imagen en part_1 o part_2,
traduce las variables clínicas al español y genera un dataset conversacional
multimodal compatible con la plantilla de chat de Qwen2-VL, en formato JSONL.

VERSIÓN AMPLIADA (MÁXIMA VARIEDAD): cada componente del texto —el prompt del
usuario, la apertura del informe, la forma de nombrar el diagnóstico, la
naturaleza de la lesión, la localización, el método de confirmación y la
recomendación— cuenta con entre 15 y 20 variantes de redacción, escritas
variando estructura de frase, longitud y conectores (no solo sinónimos), para
sonar fluido y humano. Cada componente se elige de forma DETERMINISTA
(semilla derivada del image_id + un "salt" propio de ese componente), así:
  - La ejecución es 100% reproducible entre corridas.
  - Las combinaciones posibles entre componentes se multiplican, haciendo la
    repetición literal entre dos informes distintos prácticamente improbable.
  - El contenido clínico de fondo (diagnóstico real, nivel de riesgo real,
    siempre derivar a un profesional, nunca prescribir) nunca cambia, solo
    la forma de expresarlo.

Salida: datos/preparados/dataset_entrenamiento.jsonl
"""

import os
import csv
import json
import sys
import random

# ---------------------------------------------------------------------------
# Rutas base (relativas a la raíz del proyecto, asumiendo ejecución desde ahí)
# ---------------------------------------------------------------------------
RUTA_CSV = os.path.join("datos", "crudos", "HAM10000_metadata.csv")
RUTA_IMAGENES_PARTE_1 = os.path.join("datos", "crudos", "HAM10000_images_part_1")
RUTA_IMAGENES_PARTE_2 = os.path.join("datos", "crudos", "HAM10000_images_part_2")
RUTA_SALIDA_DIR = os.path.join("datos", "preparados")
RUTA_SALIDA_JSONL = os.path.join(RUTA_SALIDA_DIR, "dataset_entrenamiento.jsonl")

EXTENSION_IMAGEN = ".jpg"


# ---------------------------------------------------------------------------
# Diccionarios de traducción de variables clínicas (sin cambios de fondo)
# ---------------------------------------------------------------------------

DICCIONARIO_DIAGNOSTICOS = {
    "akiec": "Queratosis Actínica / Carcinoma Intraepitelial",
    "bcc": "Carcinoma Basocelular",
    "bkl": "Lesión Queratósica Benigna",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevus Melanocítico (Lunar Benigno)",
    "vasc": "Lesión Vascular",
}

DICCIONARIO_TIPO_DX = {
    "histo": "confirmación histopatológica (biopsia)",
    "follow_up": "seguimiento clínico",
    "consensus": "consenso de expertos",
    "confocal": "microscopía confocal",
}

DICCIONARIO_LOCALIZACION = {
    "back": "la espalda",
    "lower extremity": "la extremidad inferior",
    "trunk": "el tronco",
    "upper extremity": "la extremidad superior",
    "abdomen": "el abdomen",
    "face": "el rostro",
    "chest": "el pecho",
    "foot": "el pie",
    "neck": "el cuello",
    "scalp": "el cuero cabelludo",
    "hand": "la mano",
    "ear": "la oreja",
    "genital": "la zona genital",
    "acral": "la zona acral",
    "unknown": "localización no especificada",
}

DICCIONARIO_SEXO = {
    "male": "masculino",
    "female": "femenino",
    "unknown": "no especificado",
}


def traducir(diccionario, clave, valor_por_defecto="no especificado"):
    """Traduce una clave usando el diccionario dado, con fallback seguro."""
    if clave is None:
        return valor_por_defecto
    clave_normalizada = str(clave).strip().lower()
    return diccionario.get(clave_normalizada, valor_por_defecto)


# ---------------------------------------------------------------------------
# BANCOS DE VARIANTES — PROMPT DEL USUARIO (18 variantes)
# ---------------------------------------------------------------------------
# Cada variante es una función que recibe (fragmento_edad, sexo_es,
# localizacion_es) y retorna el texto completo del prompt. Todas piden lo
# mismo en el fondo, con estructura, longitud y tono distintos.

VARIANTES_PROMPT_USUARIO = [
    lambda edad, sexo, loc: (
        f"Analiza la siguiente imagen dermatoscópica. Corresponde a un "
        f"paciente de sexo {sexo}, {edad}, con la lesión localizada en {loc}. "
        f"Proporciona un informe dermatológico detallado en español, indicando "
        f"el diagnóstico más probable, su naturaleza (benigna o maligna) y una "
        f"recomendación clínica general."
    ),
    lambda edad, sexo, loc: (
        f"Por favor, examina esta imagen dermatoscópica correspondiente a un "
        f"paciente {sexo}, {edad}. La lesión se ubica en {loc}. Necesito un "
        f"informe en español que incluya el diagnóstico más probable, si la "
        f"lesión es benigna o maligna, y una recomendación clínica."
    ),
    lambda edad, sexo, loc: (
        f"Evalúa esta fotografía dermatoscópica. Datos del paciente: sexo "
        f"{sexo}, {edad}, lesión localizada en {loc}. Redacta un informe "
        f"dermatológico en español con diagnóstico probable, naturaleza de la "
        f"lesión y recomendación clínica general."
    ),
    lambda edad, sexo, loc: (
        f"Quisiera un análisis dermatológico de esta imagen. El paciente es "
        f"{sexo}, {edad}, y la lesión se encuentra en {loc}. Indica en "
        f"español el diagnóstico más probable, su naturaleza clínica y una "
        f"recomendación general para el paciente."
    ),
    lambda edad, sexo, loc: (
        f"Observa la imagen dermatoscópica adjunta y genera un informe "
        f"clínico en español. El paciente, de sexo {sexo}, {edad}, presenta "
        f"la lesión en {loc}. Incluye diagnóstico probable, naturaleza de la "
        f"lesión y recomendación."
    ),
    lambda edad, sexo, loc: (
        f"Se requiere una evaluación dermatoscópica de la imagen adjunta. "
        f"Paciente {sexo}, {edad}, con lesión en {loc}. El informe debe "
        f"estar en español e incluir diagnóstico probable, naturaleza "
        f"(benigna o maligna) y recomendación clínica."
    ),
    lambda edad, sexo, loc: (
        f"A partir de esta imagen dermatoscópica de un paciente {sexo} "
        f"({edad}), con la lesión en {loc}, elabora un informe dermatológico "
        f"en español que precise el diagnóstico más probable, su naturaleza "
        f"y una recomendación clínica adecuada."
    ),
    lambda edad, sexo, loc: (
        f"Realiza el análisis dermatológico de la siguiente imagen. Se trata "
        f"de un paciente {sexo}, {edad}, con lesión localizada en {loc}. "
        f"Responde en español, señalando el diagnóstico probable, si la "
        f"lesión es benigna o maligna, y una recomendación clínica."
    ),
    lambda edad, sexo, loc: (
        f"Adjunto una imagen dermatoscópica para su valoración. Paciente "
        f"{sexo}, {edad}. Lesión ubicada en {loc}. Agradecería un informe en "
        f"español que precise diagnóstico probable, naturaleza de la lesión "
        f"y una recomendación."
    ),
    lambda edad, sexo, loc: (
        f"Te comparto una fotografía dermatoscópica que requiere valoración. "
        f"El paciente es {sexo}, {edad}, y la lesión está en {loc}. "
        f"¿Podrías generar un informe en español con el diagnóstico probable, "
        f"su naturaleza y una recomendación clínica?"
    ),
    lambda edad, sexo, loc: (
        f"Solicito la evaluación de esta imagen dermatoscópica. Sexo del "
        f"paciente: {sexo}. Edad: {edad}. Localización de la lesión: {loc}. "
        f"El informe debe redactarse en español e incluir diagnóstico "
        f"probable, naturaleza y recomendación."
    ),
    lambda edad, sexo, loc: (
        f"¿Podrías revisar esta imagen dermatoscópica? Pertenece a un "
        f"paciente {sexo}, {edad}, con la lesión en {loc}. Me interesa "
        f"conocer el diagnóstico más probable, su naturaleza y una "
        f"recomendación clínica, todo en español."
    ),
    lambda edad, sexo, loc: (
        f"Imagen dermatoscópica para análisis. Paciente {sexo}, {edad}. "
        f"Localización: {loc}. Genera, en español, un informe con "
        f"diagnóstico probable, naturaleza de la lesión y recomendación."
    ),
    lambda edad, sexo, loc: (
        f"Necesito apoyo para interpretar esta imagen dermatoscópica de un "
        f"paciente {sexo}, {edad}, con la lesión localizada en {loc}. "
        f"Redacta el informe en español, cubriendo diagnóstico probable, "
        f"naturaleza y recomendación clínica."
    ),
    lambda edad, sexo, loc: (
        f"Aquí tienes una imagen dermatoscópica de un paciente {sexo} de "
        f"{edad}, con la lesión situada en {loc}. Elabora un informe en "
        f"español que indique el diagnóstico probable, la naturaleza de la "
        f"lesión y una recomendación clínica general."
    ),
    lambda edad, sexo, loc: (
        f"Analiza la imagen adjunta correspondiente a una lesión "
        f"dermatoscópica en {loc}, de un paciente {sexo}, {edad}. Se "
        f"requiere un informe en español con diagnóstico probable, "
        f"naturaleza de la lesión y recomendación."
    ),
    lambda edad, sexo, loc: (
        f"Ayúdame a interpretar esta fotografía dermatoscópica. Se trata de "
        f"un paciente {sexo}, {edad}, con la lesión en {loc}. Quisiera un "
        f"informe claro en español: diagnóstico probable, naturaleza y "
        f"recomendación."
    ),
    lambda edad, sexo, loc: (
        f"Presento para análisis una imagen dermatoscópica de un paciente "
        f"{sexo}, {edad}, con lesión en {loc}. Favor de responder en "
        f"español con el diagnóstico más probable, la naturaleza de la "
        f"lesión y una recomendación clínica."
    ),
]


# ---------------------------------------------------------------------------
# BANCOS DE VARIANTES — INFORME DEL ASISTENTE
# ---------------------------------------------------------------------------

# --- Apertura del informe (18 variantes) ---
VARIANTES_APERTURA = [
    "INFORME DERMATOLÓGICO\n\n",
    "INFORME DE ANÁLISIS DERMATOLÓGICO\n\n",
    "RESULTADO DEL ANÁLISIS DERMATOSCÓPICO\n\n",
    "INFORME CLÍNICO — ANÁLISIS DE IMAGEN DERMATOSCÓPICA\n\n",
    "EVALUACIÓN DERMATOLÓGICA ASISTIDA\n\n",
    "INFORME DE HALLAZGOS DERMATOSCÓPICOS\n\n",
    "RESUMEN DE EVALUACIÓN DERMATOLÓGICA\n\n",
    "ANÁLISIS DERMATOSCÓPICO ASISTIDO POR IA\n\n",
    "INFORME PRELIMINAR DE LESIÓN CUTÁNEA\n\n",
    "REPORTE DE ANÁLISIS DE IMAGEN CLÍNICA\n\n",
    "HALLAZGOS DEL ANÁLISIS DERMATOLÓGICO\n\n",
    "INFORME DE APOYO DIAGNÓSTICO DERMATOLÓGICO\n\n",
    "EVALUACIÓN ASISTIDA DE LESIÓN CUTÁNEA\n\n",
    "INFORME DE INTERPRETACIÓN DERMATOSCÓPICA\n\n",
    "RESULTADOS DE LA EVALUACIÓN DE IMAGEN\n\n",
    "INFORME TÉCNICO — DERMATOSCOPÍA DIGITAL\n\n",
    "ANÁLISIS DE LESIÓN CUTÁNEA ASISTIDO\n\n",
    "INFORME DE VALORACIÓN DERMATOLÓGICA\n\n",
]

# --- Forma de introducir el diagnóstico probable (18 variantes) ---
VARIANTES_INTRODUCCION_DIAGNOSTICO = [
    "Diagnóstico probable: {nombre_dx}.",
    "El diagnóstico más probable corresponde a: {nombre_dx}.",
    "Tras el análisis de la imagen, se observa un cuadro compatible con: {nombre_dx}.",
    "Hallazgo principal: {nombre_dx}.",
    "La lesión analizada es compatible con un diagnóstico de: {nombre_dx}.",
    "Impresión diagnóstica: {nombre_dx}.",
    "Con base en las características visuales observadas, se sugiere: {nombre_dx}.",
    "El patrón dermatoscópico identificado orienta hacia: {nombre_dx}.",
    "Diagnóstico sugerido por el análisis de imagen: {nombre_dx}.",
    "La evaluación de la lesión apunta a un cuadro de: {nombre_dx}.",
    "Se identifica un patrón compatible con: {nombre_dx}.",
    "El análisis visual de la lesión sugiere fuertemente: {nombre_dx}.",
    "Considerando la morfología observada, el diagnóstico probable es: {nombre_dx}.",
    "Interpretación diagnóstica preliminar: {nombre_dx}.",
    "Los rasgos dermatoscópicos observados son consistentes con: {nombre_dx}.",
    "A partir de los hallazgos visuales, se orienta el diagnóstico hacia: {nombre_dx}.",
    "Se estima, con base en la imagen, un cuadro de: {nombre_dx}.",
    "El sistema identifica como diagnóstico más probable: {nombre_dx}.",
]

# --- Naturaleza de la lesión, agrupada por nivel de riesgo (18 por grupo) ---
VARIANTES_NATURALEZA = {
    "benigna": [
        "Naturaleza de la lesión: benigna, sin signos de alarma evidentes.",
        "Se trata de una lesión de naturaleza benigna.",
        "Naturaleza de la lesión: benigna.",
        "Los hallazgos son compatibles con una lesión benigna.",
        "Naturaleza: benigna, de bajo riesgo clínico aparente.",
        "La lesión presenta características propias de un cuadro benigno.",
        "No se observan rasgos que sugieran malignidad; el cuadro parece benigno.",
        "Los patrones visuales identificados corresponden a una lesión benigna típica.",
        "Se trata, aparentemente, de una lesión sin comportamiento maligno.",
        "El aspecto general de la lesión es compatible con benignidad.",
        "No hay indicios visuales de un proceso maligno en esta lesión.",
        "La morfología observada corresponde a un patrón benigno habitual.",
        "En apariencia, se trata de una lesión de bajo riesgo y curso benigno.",
        "Los bordes, color y textura observados son consistentes con benignidad.",
        "Nada en las características visuales sugiere un comportamiento agresivo.",
        "El cuadro observado corresponde a una lesión benigna común en la práctica clínica.",
        "Se aprecia un patrón regular y homogéneo, propio de lesiones benignas.",
        "Los hallazgos no plantean sospecha de malignidad en esta evaluación.",
    ],
    "vigilancia": [
        "Naturaleza de la lesión: potencialmente premaligna, requiere seguimiento.",
        "Se trata de una lesión potencialmente premaligna que amerita vigilancia.",
        "Naturaleza de la lesión: de comportamiento incierto, sugiere seguimiento clínico.",
        "Los hallazgos son compatibles con una lesión que requiere control periódico.",
        "Naturaleza: premaligna o de bajo riesgo maligno, con necesidad de seguimiento.",
        "La lesión presenta características que justifican vigilancia dermatológica activa.",
        "Se observan rasgos que, sin ser concluyentes de malignidad, ameritan control.",
        "El cuadro es compatible con una lesión de potencial premaligno.",
        "No puede descartarse una evolución hacia malignidad sin seguimiento adecuado.",
        "Los patrones visuales sugieren cautela, sin ser definitivamente malignos.",
        "Se trata de un hallazgo con cierto grado de incertidumbre que exige control.",
        "La morfología observada plantea la necesidad de vigilancia dermatológica.",
        "Existe posibilidad de progresión, por lo que se sugiere monitoreo periódico.",
        "El aspecto de la lesión no es concluyente y amerita revisión especializada.",
        "Se identifican características que ameritan un control más estrecho que el habitual.",
        "La lesión podría evolucionar con el tiempo, por lo que se aconseja seguimiento.",
        "Los hallazgos no son alarmantes, pero tampoco descartables sin control clínico.",
        "El cuadro observado se encuentra en una zona intermedia que exige atención periódica.",
    ],
    "maligna_bajo_riesgo": [
        "Naturaleza de la lesión: maligna de bajo riesgo metastásico, requiere tratamiento.",
        "Se trata de una lesión maligna de crecimiento local, con bajo riesgo de diseminación.",
        "Naturaleza de la lesión: maligna, aunque de comportamiento habitualmente localizado.",
        "Los hallazgos son compatibles con malignidad de bajo riesgo sistémico.",
        "Naturaleza: maligna de bajo grado de agresividad, con indicación de tratamiento.",
        "La lesión presenta características malignas que requieren manejo médico, con riesgo metastásico bajo.",
        "Se observa un patrón maligno de crecimiento lento y comportamiento localizado.",
        "El cuadro corresponde a una neoplasia maligna de baja agresividad.",
        "Los rasgos observados sugieren malignidad, aunque de curso generalmente indolente.",
        "Se trata de un proceso maligno que rara vez compromete otros órganos.",
        "La lesión presenta signos de malignidad local, sin evidencia de diseminación.",
        "El patrón identificado es propio de una neoplasia maligna de bajo riesgo sistémico.",
        "Se aprecian características malignas, aunque con pronóstico habitualmente favorable si se trata a tiempo.",
        "Los hallazgos son compatibles con un tumor maligno de comportamiento localmente invasivo.",
        "Se trata de malignidad cutánea de bajo potencial metastásico.",
        "El cuadro observado corresponde a una lesión maligna que amerita tratamiento, con pronóstico generalmente bueno.",
        "La lesión muestra rasgos malignos, si bien su capacidad de diseminación es limitada.",
        "Se identifica un proceso maligno de crecimiento predominantemente local.",
    ],
    "maligna_alto_riesgo": [
        "Naturaleza de la lesión: maligna de alto riesgo, requiere atención prioritaria.",
        "Se trata de una lesión maligna con potencial de diseminación, de alta prioridad clínica.",
        "Naturaleza de la lesión: maligna y de comportamiento agresivo, requiere atención inmediata.",
        "Los hallazgos son compatibles con malignidad de alto riesgo, ameritando manejo urgente.",
        "Naturaleza: maligna de alto riesgo, con necesidad de evaluación especializada sin demora.",
        "La lesión presenta características de alto riesgo oncológico y requiere atención prioritaria.",
        "Se observan rasgos altamente sugestivos de malignidad agresiva.",
        "El patrón identificado es compatible con una neoplasia maligna de comportamiento invasivo.",
        "Los hallazgos plantean una sospecha seria de malignidad con potencial de diseminación.",
        "Se trata de un cuadro que podría comprometer otros tejidos si no se trata a tiempo.",
        "La lesión muestra signos de alarma compatibles con un proceso maligno avanzado.",
        "El aspecto observado es altamente sospechoso de un tumor maligno agresivo.",
        "Se identifican características que sugieren un riesgo oncológico significativo.",
        "Los rasgos visuales son preocupantes y compatibles con malignidad de comportamiento rápido.",
        "Este hallazgo reviste especial importancia por su potencial de diseminación.",
        "La lesión presenta un patrón que exige descartar malignidad de forma prioritaria.",
        "Se trata de un cuadro que, de confirmarse, requeriría manejo oncológico especializado.",
        "Los hallazgos son compatibles con un proceso maligno que no debe subestimarse.",
    ],
}

# Mapa fino de riesgo por dx (4 niveles: benigna, vigilancia,
# maligna_bajo_riesgo, maligna_alto_riesgo). Se usa TANTO para la
# "naturaleza" como para la "recomendación", garantizando coherencia
# clínica entre ambos componentes del informe.
MAPA_NATURALEZA_POR_DX = {
    "akiec": "vigilancia",
    "bcc": "maligna_bajo_riesgo",
    "bkl": "benigna",
    "df": "benigna",
    "mel": "maligna_alto_riesgo",
    "nv": "benigna",
    "vasc": "benigna",
}

# --- Forma de mencionar la localización (12 variantes) ---
VARIANTES_LOCALIZACION_FRASE = [
    "Localización evaluada: {localizacion}.",
    "La lesión se ubica en {localizacion}.",
    "Zona anatómica analizada: {localizacion}.",
    "Localización: {localizacion}.",
    "La imagen corresponde a una lesión situada en {localizacion}.",
    "Región anatómica comprometida: {localizacion}.",
    "El hallazgo se localiza en {localizacion}.",
    "Sitio de la lesión: {localizacion}.",
    "La lesión analizada se encuentra en {localizacion}.",
    "Ubicación anatómica: {localizacion}.",
    "Zona corporal evaluada: {localizacion}.",
    "La lesión fue identificada en {localizacion}.",
]

# --- Forma de mencionar el método de confirmación (12 variantes) ---
VARIANTES_METODO_CONFIRMACION = [
    "Método de confirmación diagnóstica de referencia: {metodo}.",
    "El diagnóstico de referencia fue establecido mediante: {metodo}.",
    "Confirmación diagnóstica basada en: {metodo}.",
    "Método de referencia utilizado: {metodo}.",
    "El diagnóstico original fue validado a través de: {metodo}.",
    "Referencia diagnóstica: {metodo}.",
    "Este caso fue confirmado clínicamente mediante: {metodo}.",
    "El método empleado para la confirmación fue: {metodo}.",
    "La validación diagnóstica de referencia se realizó por: {metodo}.",
    "Fuente de confirmación diagnóstica: {metodo}.",
    "El caso cuenta con confirmación mediante: {metodo}.",
    "Procedimiento de referencia para este diagnóstico: {metodo}.",
]

# --- Recomendaciones, agrupadas por nivel de riesgo (18 por grupo). Todas
# mantienen el mismo principio: nunca reemplazar el diagnóstico profesional,
# solo apoyar en la detección y orientar según el nivel de urgencia real. ---
VARIANTES_RECOMENDACION = {
    "benigna": [
        "Se sugiere validación por un dermatólogo certificado, complementando "
        "este análisis con evaluación clínica presencial.",
        "Aunque los hallazgos sugieren una lesión de bajo riesgo, se recomienda "
        "una revisión dermatológica de rutina para confirmar el diagnóstico.",
        "Se aconseja mantener un control periódico de la lesión y consultar a "
        "un dermatólogo si se observan cambios en tamaño, color o forma.",
        "Este análisis es orientativo; se recomienda una consulta dermatológica "
        "presencial para descartar cualquier duda diagnóstica.",
        "Se recomienda una evaluación de rutina con un especialista para "
        "confirmar estos hallazgos y descartar cualquier hallazgo adicional.",
        "Si bien no se observan signos de alarma, toda lesión cutánea debe ser "
        "valorada en algún momento por un profesional de la dermatología.",
        "Se sugiere fotografiar la lesión periódicamente y acudir a control "
        "dermatológico si se detecta algún cambio.",
        "Se recomienda una consulta dermatológica programada para validar este "
        "hallazgo, sin que ello implique urgencia.",
        "No se identifican señales de alarma; aun así, conviene que un "
        "dermatólogo confirme el diagnóstico en una próxima revisión.",
        "Este resultado no debe reemplazar el criterio médico; se sugiere "
        "comentarlo con un dermatólogo en la siguiente consulta de rutina.",
        "Puede continuar con sus controles habituales, e incluir esta lesión "
        "en la próxima revisión dermatológica general.",
        "Se aconseja observación regular y consulta profesional ante cualquier "
        "cambio perceptible en la lesión.",
        "Aunque el panorama es tranquilizador, la validación por un "
        "especialista sigue siendo el paso recomendado.",
        "Se sugiere incluir esta lesión en el próximo chequeo dermatológico "
        "anual, sin necesidad de adelantar la cita.",
        "Vale la pena llevar un registro visual de la lesión y compartirlo con "
        "su dermatólogo en la siguiente visita.",
        "Se recomienda no automedicarse ni intervenir la lesión, y esperar la "
        "valoración de un profesional en su próxima consulta.",
        "Este hallazgo parece de bajo riesgo, pero la confirmación definitiva "
        "corresponde siempre a un especialista certificado.",
        "Se sugiere mantener la piel protegida del sol y consultar a un "
        "dermatólogo si la lesión cambia de aspecto con el tiempo.",
    ],
    "vigilancia": [
        "Se sugiere validación por un dermatólogo certificado a la brevedad, "
        "complementando este análisis con evaluación clínica presencial y, de "
        "ser necesario, estudio histopatológico confirmatorio.",
        "Es importante que un especialista evalúe esta lesión en persona, ya "
        "que podría requerir seguimiento cercano o una biopsia confirmatoria.",
        "Se recomienda programar una cita dermatológica en un plazo razonable "
        "para un diagnóstico definitivo y descartar progresión de la lesión.",
        "Dada la naturaleza de este hallazgo, se aconseja no postergar la "
        "evaluación presencial con un dermatólogo certificado.",
        "Se sugiere acudir a control dermatológico en las próximas semanas, "
        "ya que este tipo de lesión amerita seguimiento cercano.",
        "Este hallazgo requiere confirmación profesional; se recomienda "
        "agendar una cita con un dermatólogo en un plazo breve.",
        "Se aconseja una evaluación clínica presencial y, de ser indicado por "
        "el especialista, un estudio histopatológico complementario.",
        "Se recomienda seguimiento dermatológico activo, dado que este tipo de "
        "lesión puede evolucionar con el tiempo.",
        "Conviene que un dermatólogo revise esta lesión pronto, ya que el "
        "análisis de imagen por sí solo no puede confirmar el diagnóstico.",
        "Se aconseja no dejar pasar mucho tiempo antes de la consulta, dado "
        "que este tipo de hallazgo se beneficia de una evaluación temprana.",
        "Sería recomendable una biopsia confirmatoria si el especialista lo "
        "considera pertinente tras la evaluación presencial.",
        "Este resultado amerita una consulta dedicada, no una revisión de "
        "rutina general, para descartar progresión.",
        "Se sugiere acudir a un centro dermatológico en las próximas dos o "
        "tres semanas para una valoración más detallada.",
        "El seguimiento cercano de esta lesión es importante; no se recomienda "
        "esperar al chequeo anual habitual.",
        "Se aconseja documentar cualquier cambio visible y llevar esa "
        "información a la consulta con el especialista.",
        "Este tipo de hallazgo suele beneficiarse de una segunda opinión "
        "profesional antes de descartar cualquier riesgo.",
        "Se recomienda priorizar esta consulta dermatológica frente a otras "
        "revisiones médicas no urgentes.",
        "Aunque no hay certeza de malignidad, la prudencia clínica sugiere no "
        "demorar la evaluación presencial.",
    ],
    "maligna_bajo_riesgo": [
        "Se recomienda buscar atención dermatológica en un plazo breve, ya "
        "que este hallazgo requiere evaluación clínica y, probablemente, "
        "tratamiento especializado.",
        "Es fundamental una consulta con un dermatólogo o cirujano "
        "especializado, dado que este tipo de lesión suele requerir "
        "tratamiento local.",
        "Se aconseja acudir a evaluación médica en las próximas semanas; el "
        "diagnóstico definitivo y el manejo deben ser determinados por un "
        "especialista.",
        "Se recomienda programar una consulta con un dermatólogo para definir "
        "el tratamiento más adecuado para esta lesión.",
        "Este hallazgo amerita atención médica, aunque no necesariamente de "
        "urgencia inmediata; se sugiere no postergar la consulta.",
        "Se aconseja evaluación especializada para definir el abordaje "
        "terapéutico más adecuado según el criterio del profesional tratante.",
        "Es recomendable buscar una cita con dermatología en un plazo de "
        "pocas semanas, ya que este tipo de lesión requiere tratamiento.",
        "El manejo de este tipo de hallazgo suele ser quirúrgico local; se "
        "sugiere consultar con un especialista para definir el procedimiento.",
        "Se aconseja no ignorar este resultado; aunque el riesgo sistémico es "
        "bajo, requiere intervención médica en algún momento cercano.",
        "Este hallazgo suele tratarse con buen pronóstico si se aborda a "
        "tiempo; se recomienda no demorar la consulta especializada.",
        "Se sugiere acudir a un dermatólogo para valorar opciones de "
        "tratamiento, como resección local u otras alternativas indicadas.",
        "Aunque el riesgo de diseminación es bajo, este hallazgo debe ser "
        "confirmado y tratado por un profesional certificado.",
        "Se recomienda una consulta dermatológica prioritaria, sin llegar a "
        "ser una urgencia médica inmediata.",
        "Este resultado justifica una cita médica en un plazo corto para "
        "iniciar el manejo adecuado según indique el especialista.",
        "Se aconseja evaluación presencial pronta; la mayoría de estos casos "
        "responde bien a un tratamiento oportuno.",
        "Conviene agendar la consulta especializada sin demoras prolongadas, "
        "dado que el tratamiento temprano mejora el pronóstico.",
        "Se sugiere buscar valoración dermatológica en la brevedad posible "
        "para definir el plan terapéutico correspondiente.",
        "Este hallazgo requiere seguimiento médico activo; se aconseja no "
        "dejarlo sin evaluar por más de unas pocas semanas.",
    ],
    "maligna_alto_riesgo": [
        "Se recomienda buscar atención dermatológica de forma prioritaria, ya "
        "que este hallazgo requiere evaluación clínica y confirmación "
        "histopatológica lo antes posible.",
        "Este resultado amerita una consulta médica urgente con un "
        "especialista certificado; el diagnóstico definitivo solo puede "
        "confirmarse mediante biopsia.",
        "Es fundamental acudir a un dermatólogo a la brevedad para una "
        "evaluación presencial; este análisis no reemplaza el diagnóstico "
        "clínico profesional.",
        "Se aconseja solicitar una cita médica prioritaria, dado que este "
        "tipo de hallazgo requiere confirmación y seguimiento especializado "
        "sin demora.",
        "Dada la seriedad potencial de este hallazgo, se recomienda no "
        "demorar la evaluación por un especialista en dermatología oncológica.",
        "Se sugiere acudir a un centro de salud con carácter prioritario para "
        "una valoración clínica inmediata y estudio confirmatorio.",
        "Este hallazgo requiere atención médica sin demora; se recomienda "
        "buscar una cita de urgencia con dermatología.",
        "Se aconseja no esperar a una cita de rutina: este tipo de hallazgo "
        "justifica una consulta prioritaria en los próximos días.",
        "Es indispensable una biopsia confirmatoria a la brevedad para "
        "establecer el diagnóstico definitivo y el plan de tratamiento.",
        "Se recomienda acudir a urgencias o a un centro especializado si no "
        "es posible obtener una cita dermatológica de forma inmediata.",
        "Este resultado no debe tomarse a la ligera; se sugiere buscar "
        "atención médica especializada en el menor tiempo posible.",
        "Se aconseja informar a un familiar o acompañante y buscar apoyo para "
        "gestionar la consulta médica con la mayor prontitud posible.",
        "La detección temprana es clave en estos casos; se recomienda "
        "encarecidamente no postergar la evaluación profesional.",
        "Se sugiere contactar a un dermatólogo oncológico o a un centro "
        "especializado en cáncer de piel lo antes posible.",
        "Este hallazgo requiere manejo interdisciplinario; se aconseja buscar "
        "evaluación especializada sin ninguna demora.",
        "Se recomienda priorizar esta consulta por encima de cualquier otro "
        "compromiso médico no urgente.",
        "Ante hallazgos de este tipo, la recomendación estándar es una "
        "evaluación clínica inmediata por parte de un especialista.",
        "Se aconseja acudir a valoración médica en un plazo máximo de pocos "
        "días, dado el nivel de prioridad de este hallazgo.",
    ],
}


def seleccionar_variante(banco_variantes, semilla_base, salt):
    """
    Selecciona una variante de forma determinista, combinando el image_id
    (semilla_base) con un 'salt' propio de cada componente del texto, para
    que la elección de la apertura, la naturaleza, la recomendación, etc.
    sean independientes entre sí (evitando que siempre coincidan las mismas
    combinaciones) mientras la corrida completa siga siendo reproducible.
    """
    generador_local = random.Random(f"{semilla_base}_{salt}")
    return generador_local.choice(banco_variantes)


def construir_prompt_usuario(edad, sexo_es, localizacion_es, image_id):
    """Construye el texto del prompt del usuario, eligiendo una variante
    de redacción de forma determinista según el image_id."""
    if edad is None or edad == "" or str(edad).lower() == "nan":
        fragmento_edad = "de edad no especificada"
    else:
        try:
            edad_int = int(float(edad))
            fragmento_edad = f"de {edad_int} años"
        except (ValueError, TypeError):
            fragmento_edad = "de edad no especificada"

    plantilla_prompt = seleccionar_variante(VARIANTES_PROMPT_USUARIO, image_id, "prompt")
    return plantilla_prompt(fragmento_edad, sexo_es, localizacion_es)


def construir_respuesta_asistente(dx_codigo, nombre_dx_es, tipo_dx_es, localizacion_es, image_id):
    """
    Construye el informe dermatológico combinando, de forma independiente,
    una variante para cada componente del texto (apertura, introducción del
    diagnóstico, naturaleza, localización, método de confirmación y
    recomendación). El contenido clínico de fondo (nivel de riesgo real
    asociado al dx) nunca cambia, solo la redacción de cada bloque.
    """
    dx_normalizado = str(dx_codigo).strip().lower()
    nivel_riesgo = MAPA_NATURALEZA_POR_DX.get(dx_normalizado, "vigilancia")

    texto_apertura = seleccionar_variante(VARIANTES_APERTURA, image_id, "apertura")

    plantilla_diagnostico = seleccionar_variante(
        VARIANTES_INTRODUCCION_DIAGNOSTICO, image_id, "diagnostico"
    )
    texto_diagnostico = plantilla_diagnostico.format(nombre_dx=nombre_dx_es)

    texto_naturaleza = seleccionar_variante(
        VARIANTES_NATURALEZA[nivel_riesgo], image_id, "naturaleza"
    )

    plantilla_localizacion = seleccionar_variante(
        VARIANTES_LOCALIZACION_FRASE, image_id, "localizacion"
    )
    texto_localizacion = plantilla_localizacion.format(localizacion=localizacion_es)

    plantilla_metodo = seleccionar_variante(
        VARIANTES_METODO_CONFIRMACION, image_id, "metodo"
    )
    texto_metodo = plantilla_metodo.format(metodo=tipo_dx_es)

    texto_recomendacion = seleccionar_variante(
        VARIANTES_RECOMENDACION[nivel_riesgo], image_id, "recomendacion"
    )

    return (
        f"{texto_apertura}"
        f"{texto_diagnostico}\n"
        f"{texto_naturaleza}\n"
        f"{texto_localizacion}\n"
        f"{texto_metodo}\n\n"
        f"Recomendación: {texto_recomendacion}"
    )


def localizar_imagen(image_id):
    """Busca el archivo físico de la imagen en part_1 o part_2."""
    nombre_archivo = f"{image_id}{EXTENSION_IMAGEN}"

    ruta_candidata_1 = os.path.join(RUTA_IMAGENES_PARTE_1, nombre_archivo)
    if os.path.isfile(ruta_candidata_1):
        return ruta_candidata_1

    ruta_candidata_2 = os.path.join(RUTA_IMAGENES_PARTE_2, nombre_archivo)
    if os.path.isfile(ruta_candidata_2):
        return ruta_candidata_2

    return None


def construir_registro_conversacional(ruta_imagen, edad, sexo_codigo, localizacion_codigo,
                                        dx_codigo, dx_type_codigo, image_id):
    """Arma el diccionario final en el formato de mensajes esperado por Qwen2-VL."""
    sexo_es = traducir(DICCIONARIO_SEXO, sexo_codigo)
    localizacion_es = traducir(DICCIONARIO_LOCALIZACION, localizacion_codigo)
    nombre_dx_es = traducir(DICCIONARIO_DIAGNOSTICOS, dx_codigo, "Diagnóstico no clasificado")
    tipo_dx_es = traducir(DICCIONARIO_TIPO_DX, dx_type_codigo, "no especificado")

    texto_prompt = construir_prompt_usuario(edad, sexo_es, localizacion_es, image_id)
    texto_respuesta = construir_respuesta_asistente(
        dx_codigo, nombre_dx_es, tipo_dx_es, localizacion_es, image_id
    )

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ruta_imagen},
                    {"type": "text", "text": texto_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": texto_respuesta},
                ],
            },
        ]
    }


def validar_entorno():
    """Verifica que el CSV y las carpetas de imágenes existan antes de procesar."""
    errores = []

    if not os.path.isfile(RUTA_CSV):
        errores.append(f"No se encontró el archivo de metadatos: {RUTA_CSV}")

    if not os.path.isdir(RUTA_IMAGENES_PARTE_1):
        errores.append(f"No se encontró la carpeta de imágenes: {RUTA_IMAGENES_PARTE_1}")

    if not os.path.isdir(RUTA_IMAGENES_PARTE_2):
        errores.append(f"No se encontró la carpeta de imágenes: {RUTA_IMAGENES_PARTE_2}")

    if errores:
        print("ERROR: No se puede continuar. Se encontraron los siguientes problemas:")
        for error in errores:
            print(f"  - {error}")
        sys.exit(1)


def main():
    print("=" * 70)
    print("PREPARACIÓN DE DATASET - HAM10000 -> Formato Qwen2-VL (JSONL)")
    print("VERSIÓN AMPLIADA: máxima variedad de redacción, determinista")
    print("=" * 70)

    validar_entorno()

    os.makedirs(RUTA_SALIDA_DIR, exist_ok=True)

    total_filas_csv = 0
    total_emparejados = 0
    total_no_encontrados = 0
    ids_no_encontrados = []

    try:
        with open(RUTA_CSV, mode="r", encoding="utf-8", newline="") as archivo_csv:
            lector = csv.DictReader(archivo_csv)

            columnas_requeridas = {"image_id", "dx", "dx_type", "age", "sex", "localization"}
            columnas_presentes = set(lector.fieldnames or [])
            columnas_faltantes = columnas_requeridas - columnas_presentes
            if columnas_faltantes:
                print(f"ERROR: El CSV no contiene las columnas esperadas: {columnas_faltantes}")
                sys.exit(1)

            with open(RUTA_SALIDA_JSONL, mode="w", encoding="utf-8") as archivo_salida:
                for fila in lector:
                    total_filas_csv += 1

                    image_id = fila.get("image_id", "").strip()
                    if not image_id:
                        total_no_encontrados += 1
                        continue

                    ruta_imagen = localizar_imagen(image_id)
                    if ruta_imagen is None:
                        total_no_encontrados += 1
                        ids_no_encontrados.append(image_id)
                        continue

                    registro = construir_registro_conversacional(
                        ruta_imagen=ruta_imagen,
                        edad=fila.get("age"),
                        sexo_codigo=fila.get("sex"),
                        localizacion_codigo=fila.get("localization"),
                        dx_codigo=fila.get("dx"),
                        dx_type_codigo=fila.get("dx_type"),
                        image_id=image_id,
                    )

                    archivo_salida.write(json.dumps(registro, ensure_ascii=False) + "\n")
                    total_emparejados += 1

    except Exception as error:
        print(f"ERROR inesperado durante el procesamiento del CSV: {error}")
        sys.exit(1)

    print("\n" + "-" * 70)
    print("RESUMEN DEL PROCESAMIENTO")
    print("-" * 70)
    print(f"Filas leídas del CSV:              {total_filas_csv}")
    print(f"Registros emparejados exitosamente: {total_emparejados}")
    print(f"Registros no encontrados:           {total_no_encontrados}")

    if ids_no_encontrados:
        muestra = ids_no_encontrados[:10]
        print(f"Ejemplos de image_id no localizados (máx. 10): {muestra}")

    print(f"\nArchivo generado en: {RUTA_SALIDA_JSONL}")
    print("=" * 70)


if __name__ == "__main__":
    main()