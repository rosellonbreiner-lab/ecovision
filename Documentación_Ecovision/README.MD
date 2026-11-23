# EcoVision — Clasificador de Residuos con Gemini Vision

Proyecto de **Redes Neuronales Artificiales (Unidad 05)**.  
Clasifica residuos en 3 contenedores (Blanco, Verde, Negro) según la **Resolución 2184 de 2019** de Colombia.

**Periodo del proyecto:** 2 al 22 de noviembre de 2026

---

## Motor de IA

- **Principal:** Google Gemini Vision (API multimodal). El modelo analiza la imagen y devuelve un JSON con la clase, el contenedor y la justificación normativa.
- **Respaldo offline:** MobileNetV2 (`ecovision.keras`) entrenado por el equipo con **86% de accuracy** en TrashNet, para cuando no hay API Key o falla la red.

---

## Requisitos

- Python 3.10 o 3.11
- API Key de Gemini (gratuita en https://aistudio.google.com/app/apikey)

---

## Instalación

```bash
git clone https://github.com/rosellonbreiner-lab/ecovision.git
cd ecovision
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Configuración

Copia `.env.example` a `.env` y pega tu API Key:

```
GEMINI_API_KEY=tu_api_key_aqui
```

---

## Ejecución

### App Streamlit (interfaz principal)

```bash
streamlit run app.py
```

Abre `http://localhost:8501`. Tiene 2 pestañas: cámara (foto) y subir archivo.

### Servidor web (video en tiempo real + enlace celular por QR)

```bash
python web_app.py
```

Abre `http://localhost:8000` en el navegador.

### Visor de escritorio OpenCV (bonus)

```bash
python live_cam.py
```

### Plan B (sin interfaz, sobre dataset local)

```bash
python imshow.py dataset/test
```

---

## Estructura del proyecto

```
ecovision/
├── app.py                  # Interfaz Streamlit (pestañas cámara + subir)
├── web_app.py              # Servidor FastAPI (video en vivo + QR celular)
├── gemini_vision.py        # Integración Gemini Vision
├── classifier.py           # Clasificador local (fallback MobileNetV2)
├── rules.py                # Lógica de contenedores (Res. 2184)
├── live_cam.py             # Visor OpenCV con HUD
├── imshow.py               # Plan B sobre carpeta local
├── train_model.py          # Entrenamiento MobileNetV2
├── remap.py                # Remapeo de TrashNet a 4 clases
├── requirements.txt
├── model/ecovision.keras   # Modelo local entrenado
└── .env                    # GEMINI_API_KEY
```

---

## Separación IA / lógica

- **Gemini Vision** (IA): recibe la imagen, devuelve clase final + contenedor + instrucción.
- **`rules.py`** (lógica): aplica umbral de confianza y mapea a la caneca oficial.

La IA no decide el contenedor. Solo clasifica. La lógica de decisión está en `rules.py` según la Resolución 2184 de 2019.

---

## Resultados principales

- Gemini Vision: 8/10 aciertos en pruebas documentadas (ver `PRUEBAS.MD`).
- Latencia promedio: 420 ms.
- Modelo local de respaldo: 86% accuracy, macro-F1 0.84 en test.

---

## Autores

- Joás Alvarez
- Nestor Granadillo
- Luis Miranda
- Breiner Rosellón
- Jesús Cárdenas
- Jesús Silva
- Ramsés Sepúlveda

Curso: Redes Neuronales Artificiales · Noveno semestre · 2026