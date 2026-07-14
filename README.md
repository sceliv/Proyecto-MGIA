# Proyecto MGIA — Sistema de IA Generativa Multimodal para Análisis Dermatológico

Sistema local de Inteligencia Artificial Generativa Multimodal para el análisis dermatológico asistido (detección de melanomas y otras lesiones cutáneas), basado en el ajuste fino (QLoRA) del modelo de visión y lenguaje **Qwen2-VL-2B-Instruct** sobre el dataset clínico **HAM10000**, optimizado para ejecutarse íntegramente en hardware local con recursos limitados (GPU NVIDIA RTX 3050, 6GB VRAM).

> ⚠️ **Aviso importante:** este es un proyecto académico de apoyo a la detección. En ningún caso reemplaza el diagnóstico de un profesional de la salud. Cualquier resultado generado por el sistema debe ser validado por un dermatólogo certificado.

---

## 1. Descripción general

El sistema toma una imagen dermatoscópica como entrada y genera, en español, un informe clínico estructurado que incluye el diagnóstico más probable, la naturaleza de la lesión (benigna o maligna), y una recomendación orientada siempre a la derivación profesional — nunca a la prescripción de tratamiento.

El pipeline completo se compone de tres etapas secuenciales y una interfaz final:

1. **Preparación de datos** — traduce y estructura el dataset HAM10000 en formato conversacional multimodal.
2. **Entrenamiento (SFT con QLoRA)** — ajusta el modelo base sobre el dataset preparado, dentro del límite de 6GB de VRAM.
3. **Evaluación** — genera informes sobre imágenes de prueba y calcula métricas objetivas (ROUGE, BLEU).
4. **Aplicación web** — interfaz local (Gradio) para analizar imágenes nuevas de forma interactiva.

---

## 2. Estructura del proyecto

```
PROYECTO_MGIA/
├── venv/                                  # Entorno virtual local
├── datos/
│   ├── crudos/
│   │   ├── HAM10000_metadata.csv
│   │   ├── HAM10000_images_part_1/
│   │   └── HAM10000_images_part_2/
│   └── preparados/
│       └── dataset_entrenamiento.jsonl    # Generado por preparar_dataset.py
├── modelos/
│   ├── checkpoint_lora/                   # Adaptador LoRA entrenado (SFT)
│   └── resultados_entrenamiento/          # Checkpoints intermedios del entrenamiento
├── codigo/
│   ├── preparar_dataset.py
│   ├── entrenar_lora.py
│   └── evaluar_generacion.py
└── app_web_generativa.py
```

---

## 3. Requisitos

- **Python** 3.10 o superior
- **GPU NVIDIA** con al menos 6GB de VRAM y drivers CUDA actualizados
- **Sistema operativo:** Windows (probado en PowerShell) o Linux

### Dependencias principales

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft trl datasets accelerate bitsandbytes
pip install rouge_score sacrebleu
pip install gradio
pip install pillow
```

> En Windows, si `pip install` falla por permisos o compilación, asegúrate de tener instalado Microsoft C++ Build Tools y de estar dentro del entorno virtual (`venv`) activado.

### Activar el entorno virtual

```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

---

## 4. Obtención del dataset

Descarga el dataset **HAM10000** (disponible públicamente, por ejemplo en Kaggle o Harvard Dataverse) y colócalo dentro de `datos/crudos/` respetando exactamente esta estructura:

```
datos/crudos/HAM10000_metadata.csv
datos/crudos/HAM10000_images_part_1/   (imágenes .jpg)
datos/crudos/HAM10000_images_part_2/   (imágenes .jpg)
```

---

## 5. Orden de ejecución

Todos los comandos se ejecutan **desde la raíz del proyecto** (`PROYECTO_MGIA/`), con el entorno virtual activado.

### Paso 1 — Preparar el dataset

```powershell
python codigo/preparar_dataset.py
```

Lee el CSV de metadatos, localiza cada imagen física en `part_1` o `part_2`, traduce las variables clínicas al español con múltiples variantes de redacción (para evitar respuestas repetitivas), y genera `datos/preparados/dataset_entrenamiento.jsonl`.

**Salida esperada:** confirmación de 10,015 registros emparejados exitosamente.

### Paso 2 — Entrenar el adaptador LoRA (QLoRA)

```powershell
python codigo/entrenar_lora.py
```

Carga el modelo base cuantizado en 4 bits, entrena el adaptador LoRA sobre el dataset preparado, y guarda el resultado en `modelos/checkpoint_lora`.

- **Duración aproximada:** 5-7 horas en una RTX 3050 (6GB VRAM).
- El progreso se reporta en consola como `📢 [Progreso]: X / 10015 imágenes analizadas`.
- El entrenamiento guarda checkpoints automáticamente cada 50 pasos y **se puede reanudar** si se interrumpe: simplemente vuelve a ejecutar el mismo comando.

> ⚠️ **Importante:** si cambias el dataset (por ejemplo, tras modificar `preparar_dataset.py`) y quieres reentrenar desde cero, **debes borrar la carpeta de checkpoints previa** antes de volver a ejecutar el script, o el entrenamiento detectará que ya "terminó" y no procesará el nuevo dataset:
>
> ```powershell
> Remove-Item -Recurse -Force "modelos\resultados_entrenamiento"
> ```

### Paso 3 — Evaluar el modelo entrenado

```powershell
# Modo evaluación completa: genera informes sobre una muestra y calcula métricas
python codigo/evaluar_generacion.py

# Modo demo: genera un único informe sobre una imagen específica
python codigo/evaluar_generacion.py --imagen datos/crudos/HAM10000_images_part_1/ISIC_0027950.jpg
```

El modo evaluación guarda un reporte detallado en `modelos/reporte_evaluacion.json`, incluyendo el texto generado, la referencia, y las métricas ROUGE-1, ROUGE-2, ROUGE-L y BLEU por cada ejemplo.

### Paso 4 — Lanzar la aplicación web

```powershell
python app_web_generativa.py
```

Abre automáticamente una interfaz local en `http://127.0.0.1:7860`. Desde ahí puedes subir una imagen dermatoscópica, indicar edad, sexo y localización de la lesión, y obtener el informe generado por el modelo.

La aplicación incluye un **guardrail de verificación de dominio**: antes de generar cualquier informe, verifica (desactivando temporalmente el adaptador LoRA) que la imagen corresponda efectivamente a una lesión de piel humana, rechazando imágenes ajenas al dominio (por ejemplo, fotos de objetos, animales, o personas sin lesiones visibles).

---

## 6. Notas técnicas y decisiones de diseño relevantes

- **Cuantización de 4 bits (NF4 + double quant) con `bfloat16`** como tipo de cómputo, en lugar de `float16`, para evitar conflictos con el escalador de gradientes (`GradScaler`) en GPUs de arquitectura Ampere.
- **Batch size de 1 con acumulación de gradientes de 8**, para simular un batch efectivo de 8 dentro del límite de 6GB de VRAM.
- **Lazy loading real de imágenes**: el dataset ligero (JSONL) solo almacena rutas de archivo; las imágenes se abren con PIL únicamente en el momento del entrenamiento o la inferencia, nunca se cargan todas en memoria de antemano.
- **Optimizador `paged_adamw_8bit`** para reducir el consumo de VRAM de los estados del optimizador.
- **`model_max_length = 4096`** configurado explícitamente en el tokenizador, para evitar el truncamiento por defecto (1024 tokens) que colapsaría los tensores de visión de Qwen2-VL.
- **Variedad de redacción**: cada informe combina, de forma determinista (semilla basada en el `image_id`), variantes independientes de apertura, introducción del diagnóstico, naturaleza de la lesión, localización, método de confirmación y recomendación — evitando que el modelo memorice un único texto fijo por diagnóstico.
- **Principio clínico constante**: en ninguna variante de texto el sistema prescribe tratamiento; todas las recomendaciones derivan la decisión final a un profesional de la salud, con un tono de urgencia coherente con el nivel de riesgo real del diagnóstico.

---

## 7. Limitaciones conocidas

- La evaluación actual no parte de una división train/test independiente: el conjunto de evaluación proviene del mismo dataset usado en el entrenamiento, por lo que las métricas reflejan principalmente consistencia de formato, no capacidad de generalización a casos completamente nuevos.
- El guardrail de verificación de dominio depende del juicio del modelo base (sin ajustar) y puede fallar en casos visualmente ambiguos (por ejemplo, piel sana sin lesión visible).
- El modelo es una herramienta de apoyo a la detección; no constituye un dispositivo médico ni ha sido validado clínicamente.

---

## 8. Flujo resumido de un uso típico

```powershell
# 1) Activar entorno
.\venv\Scripts\Activate.ps1

# 2) Preparar datos (una sola vez, o cada vez que cambie preparar_dataset.py)
python codigo/preparar_dataset.py

# 3) Entrenar (una sola vez; horas de duración)
python codigo/entrenar_lora.py

# 4) Evaluar (opcional, para obtener métricas)
python codigo/evaluar_generacion.py

# 5) Usar la aplicación
python app_web_generativa.py
```