"""
live_cam.py — Detección de Residuos en Tiempo Real con Cámara Local (OpenCV HUD)
EcoVision - Unidad 05
"""

import cv2
import time
import numpy as np
from PIL import Image
from classifier import EcoVisionClassifier

def main():
    print("==================================================")
    print("  🌱 EcoVision - Visor de Cámara en Tiempo Real")
    print("  Presiona 'q' o 'ESC' en la ventana para salir")
    print("  Presiona 's' para cambiar sensibilidad (umbral)")
    print("==================================================")

    classifier = EcoVisionClassifier()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara web local.")
        print("   Por favor verifica que ninguna otra app la esté usando.")
        return

    # Ajustar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    umbral = 0.50
    fps_time = time.time()
    last_pred_time = 0
    cached_info = None

    # Colores BGR para OpenCV
    COLOR_MAP = {
        "Blanco": (240, 240, 240),      # Blanco/Gris claro
        "Verde":  (50, 205, 50),        # Verde lima
        "Negro":  (60, 60, 60),          # Gris oscuro / Negro
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Aviso: No se pudo leer el fotograma de la cámara.")
            break

        # Espejo horizontal para sensación natural
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Inferencia cada 150 ms para mantener alta tasa de refresco
        now = time.time()
        if now - last_pred_time > 0.15 or cached_info is None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            cached_info = classifier.predict_image(pil_img, umbral=umbral)
            last_pred_time = now

        # Calcular FPS
        fps = 1.0 / (time.time() - fps_time + 1e-6)
        fps_time = time.time()

        # ================= DIBUJAR HUD FUTURISTA =================
        # Zona central de escaneo
        box_w, box_h = int(w * 0.55), int(h * 0.65)
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h

        contenedor = cached_info.get("contenedor", "Negro")
        bgr_color = COLOR_MAP.get(contenedor, (200, 200, 200))

        # Esquinas estilizadas en la zona de enfoque
        corner_len = 25
        thickness = 3
        # Sup-izq
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), bgr_color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), bgr_color, thickness)
        # Sup-der
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), bgr_color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), bgr_color, thickness)
        # Inf-izq
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), bgr_color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), bgr_color, thickness)
        # Inf-der
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), bgr_color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), bgr_color, thickness)

        # Panel superior de información (fondo semi-transparente)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (20, 24, 33), -1)
        # Panel inferior de instrucciones
        cv2.rectangle(overlay, (0, h - 55), (w, h), (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Título y FPS
        cv2.putText(frame, "EcoVision AI Live", (18, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 220, 130), 2)
        cv2.putText(frame, f"FPS: {fps:4.1f} | Latencia: {cached_info.get('tiempo_ms', 0):.0f}ms", (w - 240, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        # Resultado principal
        nombre_obj = cached_info.get("nombre_comun", cached_info.get("clase_final", "Desconocido"))
        conf = cached_info.get("confianza", 0.0)
        tipo_cont = cached_info.get("tipo_contenedor", "")
        
        cv2.putText(frame, f"OBJETO: {nombre_obj.upper()} ({conf*100:.1f}%)", (18, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"CONTENEDOR: {contenedor.upper()} [{tipo_cont}]", (18, 75),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, bgr_color, 2)

        # Instrucción inferior
        instruccion = cached_info.get("instruccion", "")
        cv2.putText(frame, f">> {instruccion[:75]}", (18, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        cv2.imshow("EcoVision - Camara en Tiempo Real (Res. 2184)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' o ESC
            break
        elif key == ord('s'):
            umbral = 0.40 if umbral >= 0.60 else (umbral + 0.10)
            print(f"Nuevo umbral de confianza: {umbral:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
