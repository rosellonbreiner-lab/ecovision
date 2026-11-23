# EcoVision — Informe Técnico

**Curso:** Redes Neuronales Artificiales - Unidad 05  
**Equipo:** Joás Alvarez, Nestor Granadillo, Luis Miranda, Breiner Rosellón, Jesús Cárdenas, Jesús Silva, Ramsés Sepúlveda  
**Periodo:** 2 al 22 de noviembre de 2026  
**Repositorio:** https://github.com/rosellonbreiner-lab/ecovision

---

## 1. Problema

En espacios universitarios se depositan residuos en contenedores incorrectos por desconocimiento o clasificación visual imprecisa. Un asistente visual automático puede reducir este error mostrando, en el momento, la categoría del residuo y la caneca correspondiente.

**Alcance:** clasificación en 3 contenedores oficiales (Blanco, Verde, Negro) según la Resolución 2184 de 2019.  
**Fuera de alcance:** diagnóstico de residuos contaminados, materiales compuestos, residuos peligrosos o electrónicos.

---

## 2. Arquitectura

```
Imagen (cámara o archivo)
        |
        v
Google Gemini Vision API (modelo multimodal)
        |
        v
JSON estructurado: clase_final, contenedor, instruccion, justificacion
        |
        v
rules.py (validación de umbral y mapeo final a caneca)
        |
        v
Interfaz (Streamlit / FastAPI / OpenCV)
```

**Motor principal:** Gemini Vision, indicado por la cátedra.  
**Motor de respaldo:** MobileNetV2 entrenado por el equipo, activado si no hay API Key o falla la red.

---

## 3. Componente de IA: Gemini Vision

- **Qué es:** modelo multimodal de Google que procesa imágenes y texto, disponible como API REST.
- **Entrada:** imagen JPEG en base64 (redimensionada a 384 px de lado máximo para reducir latencia).
- **Salida:** JSON con la clasificación y justificación normativa.
- **Prompt engineering:** el `SYSTEM_PROMPT` inyecta las reglas de la Resolución 2184 (qué va en cada caneca) y restringe al modelo a responder únicamente con un JSON válido usando `response_mime_type: "application/json"`.
- **Modelos usados:** `gemini-2.5-flash` como principal, `gemini-1.5-flash` como respaldo.
- **Limitaciones:** Gemini no expone probabilidades calibradas. El campo `confianza` es una estimación cualitativa del modelo generativo, no una probabilidad softmax. La decisión final de aceptar o rechazar la toma `rules.py`.

---

## 4. Separación IA vs. lógica (crítico)

| Componente | Tipo | Entrada | Salida |
|---|---|---|---|
| Gemini Vision | IA (API) | Imagen JPEG | JSON con clase + contenedor + instrucción |
| rules.py | Lógica | (clase, confianza) | (clase final, caneca asignada) |

La IA de Google hace la percepción. Nosotros hacemos la lógica de decisión y el mapeo a la normativa colombiana. La IA no conoce la Resolución 2184; se la inyectamos por prompt.

---

## 5. Motor de respaldo: MobileNetV2

Entrenado por el equipo con TrashNet (2.527 imágenes, 6 clases remapeadas a 4). Configuración:

- Base: MobileNetV2 preentrenada en ImageNet, base congelada.
- Cabeza: GAP + Dropout(0.3) + Dense(128, ReLU) + Dropout(0.3) + Dense(4, softmax).
- Optimizador: Adam(1e-4), 15 épocas con early stopping.
- Data augmentation: rotación 20°, zoom 15%, flip horizontal.

**Resultados en test (377 imágenes no usadas en entrenamiento):**

- **Accuracy:** 0.86
- **Macro-F1:** 0.84

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| botella_plastica | 0.85 | 0.79 | 0.82 | 72 |
| lata | 0.77 | 0.90 | 0.83 | 61 |
| no_identificado | 0.79 | 0.74 | 0.76 | 95 |
| papel_carton | 0.95 | 0.96 | 0.96 | 149 |

**Rol:** este modelo no es el motor principal. Se usa como respaldo offline cuando Gemini no está disponible. Se conserva en el repositorio como evidencia del entrenamiento propio realizado por el equipo.

---

## 6. Resultados experimentales

La tabla completa de las 10 pruebas (con condiciones adversas) se encuentra en `PRUEBAS.MD`.

**Métricas de Gemini durante las pruebas:**

- Latencia promedio: 420 ms
- Aciertos: 8/10
- Rechazos correctos (confianza baja): 2/10

**Análisis de errores:**

- **Luz tenue (caso 5):** Gemini confundió una botella oscura con "no identificado". Mitigado por el umbral de `rules.py`, que fuerza el rechazo y pide repetir la foto.
- **Residuos contaminados (caso 9):** Gemini clasificó una caja de pizza grasosa como "papel/cartón" (Blanco), cuando debería ir a Negro. Limitación conocida del prompt actual: no distingue entre papel limpio y contaminado.
- **Múltiples objetos (caso 8):** Gemini detectó la botella e ignoró la lata. Se añadió una regla al prompt para rechazar escenas con más de un residuo; tras el cambio, el caso 8 se maneja correctamente.

---

## 7. Ética, privacidad y límites

- **Datos:** no se capturan personas, solo residuos. Las imágenes se envían a la API de Google como servicio externo; no se almacenan en el repositorio.
- **Límites:** no sustituye a un experto en gestión de residuos; no es normativa oficial; depende de internet para funcionar con Gemini.
- **Licencias:** TensorFlow (Apache 2.0), Streamlit (Apache 2.0), FastAPI (MIT), OpenCV (Apache 2.0), TrashNet (MIT), Gemini API (términos de Google). Detalle completo en `APORTES_Y_LICENCIAS.MD`.
- **No se usan datos biométricos** ni se captan terceros sin autorización.

---

## 8. Conclusiones y trabajo futuro

**Logros:**

- Sistema funcional con Gemini Vision como motor principal.
- 3 modos de entrada: Streamlit (foto), FastAPI (video en vivo + QR celular), OpenCV (HUD local).
- 10 pruebas documentadas incluyendo condiciones adversas.
- Modelo local de respaldo con 86% de accuracy.
- Separación clara entre IA (Gemini) y lógica (`rules.py`).

**Trabajo futuro:**

- Fine-tuning del modelo local con fotos propias del campus.
- Integración de YOLO para detectar múltiples objetos automáticamente.
- Ampliar el prompt de Gemini con reglas específicas para residuos contaminados.
- Despliegue en la nube (Fly.io o Render).

---

## 9. Referencias

- Gemini API: https://ai.google.dev/gemini-api/docs
- Resolución 2184 de 2019 (Colombia)
- TrashNet: https://github.com/garythung/trashnet
- MobileNetV2: https://arxiv.org/abs/1801.04381
- Streamlit: https://docs.streamlit.io/
- FastAPI: https://fastapi.tiangolo.com/