# rules.py — LOGICA CONVENCIONAL Y NORMATIVA DE RESIDUOS (Resolución 2184 de 2019)
"""
Módulo de reglas de negocio para EcoVision.
Separa estrictamente la inferencia de la red neuronal de la asignación
de contenedores y recomendaciones prácticas de separación de residuos.
"""

CONTENEDORES = {
    "botella_plastica": "Blanco (aprovechables)",
    "lata":             "Blanco (aprovechables)",
    "papel_carton":     "Blanco (aprovechables)",
    "vidrio":           "Blanco (aprovechables)",
    "organico":         "Verde (orgánicos aprovechables)",
    "no_identificado":  "Negro (no aprovechables)",
}

DETALLES_RESIDUO = {
    "botella_plastica": {
        "nombre_comun": "Botella Plástica (PET)",
        "contenedor": "Blanco",
        "tipo_contenedor": "Aprovechables Limpios",
        "color_hex": "#F3F4F6",       # Blanco / Gris claro
        "color_borde": "#9CA3AF",
        "color_badge": "#2563EB",     # Azul distintivo para reciclaje
        "icono": "🍶",
        "instruccion": "Vacía los líquidos restantes, enjuágala si es posible y aplástala para optimizar espacio.",
        "aprovechable": True
    },
    "lata": {
        "nombre_comun": "Lata de Aluminio / Metal",
        "contenedor": "Blanco",
        "tipo_contenedor": "Aprovechables Limpios",
        "color_hex": "#F3F4F6",
        "color_borde": "#9CA3AF",
        "color_badge": "#0284C7",
        "icono": "🥫",
        "instruccion": "Drena cualquier residuo líquido o alimento y deposítala limpia y seca.",
        "aprovechable": True
    },
    "papel_carton": {
        "nombre_comun": "Papel y Cartón",
        "contenedor": "Blanco",
        "tipo_contenedor": "Aprovechables Limpios",
        "color_hex": "#F3F4F6",
        "color_borde": "#9CA3AF",
        "color_badge": "#4F46E5",
        "icono": "📦",
        "instruccion": "Debe estar limpio, seco y sin grasa de comida. Pliega las cajas para ahorrar espacio.",
        "aprovechable": True
    },
    "vidrio": {
        "nombre_comun": "Botella o Frasco de Vidrio",
        "contenedor": "Blanco",
        "tipo_contenedor": "Aprovechables Limpios",
        "color_hex": "#F3F4F6",
        "color_borde": "#9CA3AF",
        "color_badge": "#0D9488",
        "icono": "🍾",
        "instruccion": "Enjuaga con agua, retira la tapa metálica o plástica y deposítalo con cuidado sin romperlo.",
        "aprovechable": True
    },
    "organico": {
        "nombre_comun": "Residuo Orgánico / Restos de Comida",
        "contenedor": "Verde",
        "tipo_contenedor": "Orgánicos Aprovechables",
        "color_hex": "#10B981",       # Verde esmeralda
        "color_borde": "#059669",
        "color_badge": "#059669",
        "icono": "🍎",
        "instruccion": "Cáscaras de frutas, verduras, posos de café y restos de comida aptos para compostaje.",
        "aprovechable": True
    },
    "no_identificado": {
        "nombre_comun": "No Identificado / Desecho Ordinario",
        "contenedor": "Negro",
        "tipo_contenedor": "No Aprovechables",
        "color_hex": "#1F2937",       # Negro grafito
        "color_borde": "#111827",
        "color_badge": "#374151",
        "icono": "🗑️",
        "instruccion": "Papel higiénico, servilletas usadas, plásticos contaminados o residuos no aprovechables.",
        "aprovechable": False
    }
}

def recomendar(clase, confianza, umbral=0.55):
    """
    Función de compatibilidad original de EcoVision.
    Mapea la predicción del modelo y aplica umbral de rechazo.
    """
    if confianza < umbral:
        clase = "no_identificado"
    if clase not in CONTENEDORES:
        clase = "no_identificado"
    return clase, CONTENEDORES[clase]

def obtener_info_completa(clase, confianza, umbral=0.55):
    """
    Retorna toda la información requerida por la interfaz gráfica en tiempo real.
    """
    clase_final, nombre_contenedor = recomendar(clase, confianza, umbral)
    info = DETALLES_RESIDUO.get(clase_final, DETALLES_RESIDUO["no_identificado"]).copy()
    info["clase_original"] = clase
    info["clase_final"] = clase_final
    info["confianza"] = confianza
    info["nombre_contenedor"] = nombre_contenedor
    info["es_confiable"] = confianza >= umbral
    return info
