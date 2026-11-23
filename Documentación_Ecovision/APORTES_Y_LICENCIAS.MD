# Registro de Aportes Individuales, Licencias y Créditos

**Proyecto:** EcoVision — Clasificador de Residuos con Gemini Vision  
**Curso:** Redes Neuronales Artificiales - Unidad 05  
**Periodo:** 2 al 22 de noviembre de 2026

---

## 1. Registro de aportes individuales

| Integrante | Rol principal | Aportes concretos |
|---|---|---|
| Joás Alvarez | Líder de proyecto | Coordinación general, diseño del `SYSTEM_PROMPT` para Gemini, revisión de la separación IA/lógica en `rules.py`, integración final entre módulos |
| Nestor Granadillo | Desarrollo backend | Implementación de `web_app.py` (FastAPI), WebSocket para streaming del celular, enlace QR, manejo de errores HTTP |
| Luis Miranda | Integración de IA | Implementación de `gemini_vision.py`, optimización de latencia con sesión HTTP persistente, manejo de fallback a modelo local |
| Breiner Rosellón | Frontend y UX | Implementación de `app.py` (Streamlit), interfaz de `web_app.py` (HTML/CSS/JS), diseño visual de la caneca asignada y el historial |
| Jesús Cárdenas | Datos y entrenamiento | Remapeo de TrashNet (`remap.py`), entrenamiento del modelo MobileNetV2 (`train_model.py`), evaluación con matriz de confusión |
| Jesús Silva | QA y validación | Ejecución de las 10 pruebas documentadas (`PRUEBAS.MD`), análisis de errores, ajuste del umbral en `rules.py`, verificación del Plan B (`imshow.py`) |
| Ramsés Sepúlveda | Documentación | Redacción del `README.MD`, `INFORME_TECNICO.MD`, `APORTES_Y_LICENCIAS.MD`, coordinación de la sustentación oral |

**Nota:** todos los integrantes pueden explicar el proyecto completo, no solo su parte (requisito de la guía para la sustentación oral).

---

## 2. Licencias de software utilizado

| Recurso | Tipo | Licencia | Uso en el proyecto |
|---|---|---|---|
| TensorFlow / Keras | Framework de deep learning | Apache 2.0 | Entrenamiento del modelo local de respaldo |
| MobileNetV2 | Arquitectura de red neuronal | Apache 2.0 | Base del modelo `ecovision.keras` |
| Streamlit | Framework de interfaz | Apache 2.0 | Interfaz principal (`app.py`) |
| FastAPI | Framework web | MIT | Servidor de video en vivo (`web_app.py`) |
| OpenCV | Visión por computador | Apache 2.0 | Procesamiento de frames (`live_cam.py`) |
| Pillow | Procesamiento de imágenes | HPND | Manipulación de imágenes |
| NumPy | Cómputo numérico | BSD 3-Clause | Arrays y operaciones matriciales |
| Gemini API | API de IA multimodal | Términos de servicio de Google | Motor principal de clasificación |
| TrashNet | Dataset de residuos | MIT | Entrenamiento del modelo local |

---

## 3. Créditos y atribución de recursos externos

- **Dataset TrashNet:** creado por Gary Thung y Mindy Yang. Disponible en https://github.com/garythung/trashnet. Usado bajo licencia MIT para el entrenamiento del modelo local de respaldo.

- **MobileNetV2:** arquitectura propuesta por Sandler et al. (2018) en "MobileNetV2: Inverted Residuals and Linear Bottlenecks". Pesos preentrenados en ImageNet disponibles en TensorFlow/Keras bajo Apache 2.0.

- **Google Gemini:** API multimodal usada como motor principal de clasificación, según indicación de la cátedra. El equipo no entrena este modelo; se accede vía API REST con prompt engineering estructurado.

- **Resolución 2184 de 2019 (Colombia):** normativa que define el código de colores para la separación de residuos (Blanco, Verde, Negro). Usada como base de las reglas en `rules.py` y del `SYSTEM_PROMPT` enviado a Gemini.

- **Repositorio de referencia:** `arpanpramanik2003/smart-waste-classification` (Apache 2.0) fue consultado durante la fase de exploración de transfer learning con MobileNetV2. No se copió código directamente; se usó como referencia conceptual para el pipeline de entrenamiento.

---

## 4. Declaración de originalidad

El código de este repositorio fue desarrollado por los integrantes del equipo listados arriba durante el periodo del 2 al 22 de noviembre de 2026. Las librerías, datasets, arquitecturas y APIs externas utilizadas se declaran en las secciones anteriores con sus respectivas licencias. No se ha copiado código de otros equipos sin atribución.

**Fecha de la declaración:** 22 de noviembre de 2026

**Firmas del equipo:**

- Joás Alvarez
- Nestor Granadillo
- Luis Miranda
- Breiner Rosellón
- Jesús Cárdenas
- Jesús Silva
- Ramsés Sepúlveda