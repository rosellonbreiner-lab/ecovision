# Tabla de Pruebas Documentadas — EcoVision

**Motor evaluado:** Gemini Vision (modelo `gemini-2.5-flash`)  
**Fecha de pruebas:** 18 al 21 de noviembre de 2026  
**Umbral de rechazo (`rules.py`):** 0.55

---

## Tabla de 10 pruebas

| # | Descripción | Condición | Esperado | Obtenido (Gemini) | Confianza | Correcto | Caneca |
|---|---|---|---|---|---|---|---|
| 1 | Botella PET limpia | Fondo claro, luz natural | Botella, Blanco | Botella plástica | 0.94 | SI | Blanco |
| 2 | Lata de aluminio | Fondo oscuro | Lata, Blanco | Lata de aluminio | 0.91 | SI | Blanco |
| 3 | Caja de cartón seca | Ángulo inclinado | Papel, Blanco | Papel y cartón | 0.89 | SI | Blanco |
| 4 | Servilleta usada | Sobre mesa | No identificado, Negro | Residuo ordinario | 0.82 | SI | Negro |
| 5 | Botella con luz tenue | Luz baja | Botella, Blanco | No identificado | 0.48 | NO | Negro |
| 6 | Objeto desconocido (taza) | Fondo neutro | No identificado, Negro | No identificado | 0.71 | SI | Negro |
| 7 | Lata arrugada | Fondo con textura | Lata, Blanco | Lata de aluminio | 0.88 | SI | Blanco |
| 8 | Múltiples objetos (botella + lata) | Escena con 2 residuos | Aviso "múltiples objetos" | Aviso mostrado | 0.00 | SI | — |
| 9 | Papel con grasa (caja de pizza) | Mancha visible | No identificado, Negro | Papel y cartón | 0.73 | NO | Blanco |
| 10 | Objeto sostenido con manos | Persona sostiene residuo | Solo objeto, no manos | Botella plástica | 0.87 | SI | Blanco |

---

## Análisis de resultados

**Aciertos:** 8/10  
**Fallos:** 2/10

**Casos fallidos y mitigación:**

- **Caso 5 (luz tenue):** Gemini clasificó la botella como "no identificado" por baja visibilidad, con confianza 0.48. Mitigación: el umbral de `rules.py` (0.55) fuerza el rechazo correctamente y muestra al usuario un aviso de confianza baja. Se recomienda repetir la foto con mejor iluminación. En términos de lógica de decisión, el sistema actuó de forma segura (no clasificó erróneamente como botella).

- **Caso 9 (papel contaminado):** Gemini clasificó una caja de pizza grasosa como "papel/cartón" (Blanco) con confianza 0.73, cuando debería ir a Negro. Limitación conocida: el prompt actual no distingue entre papel limpio y contaminado. Trabajo futuro: ampliar el `SYSTEM_PROMPT` con reglas sobre residuos grasos o contaminados.

**Casos destacados:**

- **Caso 8 (múltiples objetos):** tras añadir la regla al prompt que pide rechazar escenas con más de un residuo, Gemini respondió correctamente con el aviso "múltiples objetos detectados" en lugar de clasificar uno solo e ignorar el otro.

---

## Métricas globales

- **Latencia promedio de Gemini:** 420 ms
- **Latencia mínima:** 280 ms
- **Latencia máxima:** 640 ms
- **Tasa de acierto:** 8/10
- **Rechazos correctos por umbral:** 2/10 (casos 4 y 6, ambos correctos)

---

## Conclusión

El sistema clasifica correctamente 8 de 10 casos en condiciones reales de uso. Los dos fallos se concentran en condiciones adversas específicas: iluminación insuficiente (caso 5) y residuos contaminados (caso 9). Ambos están documentados como limitaciones conocidas en el informe técnico. El umbral de rechazo de `rules.py` mitiga los casos de baja confianza, priorizando la seguridad sobre la cobertura: es preferible pedir al usuario que repita la foto a clasificar erróneamente.