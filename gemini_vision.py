"""
gemini_vision.py — Módulo de Clasificación Visual Ultrarrápido con Google Gemini Vision
Optimizado con pool de conexiones HTTP persistentes, compresión acelerada y mínima latencia (300-500ms).
"""

import os
import io
import json
import base64
import requests
from requests.adapters import HTTPAdapter
from PIL import Image

# Sesión HTTP persistente con Keep-Alive para evitar renegociar TLS/SSL en cada fotograma
_session = requests.Session()
_adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=1)
_session.mount("https://", _adapter)

SYSTEM_PROMPT = """Eres EcoVision, un clasificador visual de residuos para Colombia bajo la Resolución 2184 de 2019.
Identifica el residuo principal frente a la cámara y clasifícalo en su caneca oficial:

1. BLANCO (Aprovechables limpios y secos):
   - Plástico: Botellas PET, envases plásticos limpios, tapas, bolsas limpias.
   - Metales: Latas de aluminio de gaseosa/cerveza, hojalata, conservas limpias.
   - Papel y Cartón: Cajas de cartón secas, hojas de papel, cuadernos, periódicos (SIN grasa ni comida).
   - Vidrio: Botellas y frascos limpios.
   - Envases Tetra Pak limpios y escurridos.

2. VERDE (Orgánicos aprovechables):
   - Restos de comida, cáscaras de frutas, verduras, posos de café, residuos vegetales.

3. NEGRO (No aprovechables / Ordinarios):
   - Servilletas usadas, papel higiénico, toallas de papel usadas.
   - Cajas o papeles engrasados (ej. cajas de pizza sucias, papel con aceite).
   - Paquetes de papas/snacks (empaques metalizados tipo galletas o Doritos).
   - Icopor (poliestireno expandido), plásticos de un solo uso muy sucios.
   - Si no hay objeto claro, solo se ve la persona o la habitación vacía.

Responde ÚNICAMENTE un JSON válido (sin formato markdown ni texto adicional) con:
{
  "detectado": true,
  "nombre_comun": "Nombre descriptivo claro (ej: Botella PET de Coca-Cola)",
  "material": "Material detectado (ej: Plástico PET #1)",
  "clase_final": "botella_plastica" | "lata" | "papel_carton" | "vidrio" | "organico" | "no_identificado",
  "contenedor": "Blanco" | "Verde" | "Negro",
  "tipo_contenedor": "Aprovechables Limpios" | "Orgánicos Aprovechables" | "No Aprovechables",
  "color_hex": "#3B82F6",
  "icono": "🍶",
  "confianza": 0.95,
  "instruccion": "Acción inmediata de preparación y disposición (máximo 15 palabras)",
  "por_que": "Justificación concisa según la norma colombiana (máximo 20 palabras)"
}
"""

def optimizar_imagen(imagen_b64: str, max_size=384, calidad=65) -> str:
    """
    Comprime y redimensiona la imagen velozmente usando BILINEAR.
    Reduce el payload a ~12KB para acelerar la subida por red.
    """
    try:
        if "," in imagen_b64:
            imagen_b64 = imagen_b64.split(",", 1)[1]
        raw_bytes = base64.b64decode(imagen_b64)
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=calidad, optimize=False)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        return imagen_b64

def analizar_con_gemini(imagen_b64: str, api_key: str = None, modelo: str = "gemini-3.5-flash-lite"):
    """
    Envía la imagen optimizada a Gemini Vision reutilizando la conexión HTTP existente.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return {
            "error": "Falta la API Key de Gemini. Configúrala en la interfaz o usa el modo local.",
            "modo": "sin_api_key"
        }

    # Reducir peso del frame a alta velocidad
    img_opt = optimizar_imagen(imagen_b64, max_size=384, calidad=65)

    # Modelos ultrarrápidos con soporte de visión
    modelos_a_probar = [
        "gemini-2.5-flash",
        "gemini-3.5-flash-lite"
    ]

    ultimo_error = None

    for mod in modelos_a_probar:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{mod}:generateContent?key={key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": SYSTEM_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_opt
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.0,
                "max_output_tokens": 200
            }
        }

        try:
            # Timeout corto de 3.5s para no congelar la UI si la red falla
            resp = _session.post(url, json=payload, timeout=3.5)
            if resp.status_code == 200:
                data = resp.json()
                text_response = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                
                # Quitar delimitadores markdown si los hubiera
                if text_response.startswith("```json"):
                    text_response = text_response[7:]
                if text_response.startswith("```"):
                    text_response = text_response[3:]
                if text_response.endswith("```"):
                    text_response = text_response[:-3]
                text_response = text_response.strip()

                parsed = json.loads(text_response)
                parsed["tipo_modelo"] = f"Gemini Vision ({mod})"
                parsed["es_gemini"] = True
                return parsed
            else:
                ultimo_error = f"Error {resp.status_code}: {resp.text}"
        except Exception as e:
            ultimo_error = str(e)

    return {
        "error": f"Error al consultar Gemini API: {ultimo_error}",
        "modo": "error_api"
    }
