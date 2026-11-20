# EcoVision — Clasificador Visual de Residuos en Tiempo Real

Proyecto del curso **Redes Neuronales Artificiales (Unidad 05)**.  
Clasifica residuos visualmente en tiempo real y asigna el contenedor correspondiente según el código de colores de Colombia (**Resolución 2184 de 2019**).

---

## 🌟 Novedades y Mejoras Implementadas

1. **Cámara en Tiempo Real con Interfaz Web**:
   - Detección continua a 60 FPS con retícula de enfoque.
   - Modo espejo horizontal para movimiento natural (con atajo tecla `M`).
   - Iluminación dinámica de los 3 contenedores: **Blanco**, **Verde** y **Negro**.
   - Asistente de voz en español que anuncia la caneca recomendada.

2. **Detección con Google Gemini Vision AI (Multimodal)**:
   - Identificación de objetos con visión multimodal avanzada (marcas, estado del residuo, grado de suciedad).
   - Justificación técnica y normativa de disposición para cada residuo.
   - Pool de conexiones HTTP persistente con tiempo de respuesta optimizado (300ms - 600ms).

3. **Enlace con Celular como Cámara HD (`/remote`)**:
   - Escanea el código QR desde tu smartphone para usar su cámara trasera en alta definición.
   - Transmisión en tiempo real vía WebSocket a la interfaz de escritorio.
   - Soporte para encendido de linterna desde el navegador móvil.

4. **Modo Local Offline (MobileNetV2)**:
   - Respaldo automático mediante red neuronal convolucional local sin necesidad de internet.

---

## 🎨 Código de Contenedores (Resolución 2184 de 2019)

| Contenedor | Residuos Permitidos | Ejemplos |
|---|---|---|
| ⚪ **Blanco (Aprovechables)** | Plástico limpio, metales, papel, cartón, vidrio | Botellas PET, latas de aluminio, cajas secas, frascos |
| 🟢 **Verde (Orgánicos)** | Desechos orgánicos aprovechables | Restos de comida, cáscaras de fruta, posos de café |
| ⚫ **Negro (No aprovechables)** | Residuos ordinarios y contaminados | Servilletas usadas, papel higiénico, cajas grasosas, icopor |

---

## 🚀 Instalación y Ejecución

### 1. Instalación de dependencias
```bash
pip install -r requirements.txt
```

### 2. Variables de entorno (Opcional para modo Gemini Vision)
Copia `.env.example` a `.env` y configura tu API Key:
```env
GEMINI_API_KEY=tu_api_key_aqui
```
*(Puedes obtener una clave gratuita en [Google AI Studio](https://aistudio.google.com/app/apikey) o ingresarla directamente en la interfaz).*

### 3. Iniciar la aplicación

- **Servidor Web Ultrarrápido (Recomendado)**:
  ```bash
  python web_app.py
  ```
  O haz doble clic en `iniciar_web.bat`.  
  Abre tu navegador en: `http://localhost:8000`

- **Visor de Escritorio OpenCV**:
  ```bash
  python live_cam.py
  ```
  O haz doble clic en `iniciar_opencv.bat`.

- **Interfaz Streamlit**:
  ```bash
  streamlit run app.py
  ```
  O haz doble clic en `iniciar_streamlit.bat`.

---

## 👥 Autores
- Joás Alvarez
- Nestor Granadillo
- Luis Miranda
- Breiner Rosellón
- Jesús Cárdenas
- Jesús Silva
- Ramsés Sepúlveda
