import io
import streamlit as st
from PIL import Image
from classifier import EcoVisionClassifier
from rules import obtener_info_completa

st.set_page_config(
    page_title="EcoVision - Clasificador de Residuos",
    page_icon="🌱",
    layout="wide"
)

# Cargar clasificador en caché de Streamlit
@st.cache_resource(show_spinner="Cargando modelo neuronal...")
def get_classifier():
    return EcoVisionClassifier()

classifier = get_classifier()

# Encabezado estilizado
st.markdown("""
<div style="background: linear-gradient(135deg, #064e3b 0%, #0f172a 100%); padding: 1.8rem; border-radius: 1rem; margin-bottom: 1.5rem; border: 1px solid #059669;">
    <h1 style="color: #34d399; margin: 0; font-size: 2.2rem; font-weight: 800;">🌱 EcoVision</h1>
    <p style="color: #cbd5e1; margin: 0.4rem 0 0 0; font-size: 1.05rem;">
        Clasificador de Residuos con Red Neuronal Convolucional (MobileNetV2) y asignación según la <b>Resolución 2184 de 2019</b>.
    </p>
</div>
""", unsafe_allow_html=True)

# Pestañas de modo
tab_cam, tab_upload, tab_live = st.tabs(["📸 Foto con Cámara", "📁 Subir Archivo", "⚡ Cámara en Tiempo Real (Live 60 FPS)"])

def mostrar_resultado(img: Image.Image, umbral: float):
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("🖼️ Imagen Analizada")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("📊 Diagnóstico y Clasificación")
        with st.spinner("Ejecutando inferencia neuronal..."):
            res = classifier.predict_image(img, umbral=umbral)

        # Contenedor estilizado
        color_badge = res.get("color_badge", "#2563EB")
        nombre_obj = res.get("nombre_comun", res.get("clase_final"))
        contenedor = res.get("contenedor", "Negro")
        tipo_cont = res.get("tipo_contenedor", "")
        icono = res.get("icono", "♻️")
        conf = res.get("confianza", 0.0)

        # Card del Contenedor Asignado
        bg_bin = "#1e293b"
        border_bin = "#3b82f6" if contenedor == "Blanco" else ("#10b981" if contenedor == "Verde" else "#6b7280")
        
        st.markdown(f"""
        <div style="background: {bg_bin}; border: 2px solid {border_bin}; border-radius: 1rem; padding: 1.25rem; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="background: {color_badge}; color: white; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">
                        Contenedor {contenedor}
                    </span>
                    <h2 style="color: white; margin: 0.4rem 0 0 0; font-size: 1.5rem;">{nombre_obj}</h2>
                    <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">{tipo_cont}</p>
                </div>
                <div style="font-size: 3rem;">{icono}</div>
            </div>
            <div style="margin-top: 1rem; padding: 0.75rem; background: rgba(0,0,0,0.3); border-radius: 0.5rem; border-left: 4px solid {border_bin};">
                <p style="color: #e2e8f0; margin: 0; font-size: 0.9rem;"><b>💡 Instrucción:</b> {res.get('instruccion')}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Confianza", f"{conf * 100:.1f}%")
        m2.metric("Latencia", f"{res.get('tiempo_ms', 0):.0f} ms")
        m3.metric("Modelo", res.get("tipo_modelo", "MobileNetV2"))

        # Desglose de probabilidades
        st.markdown("#### 🎯 Distribución de Probabilidades Softmax")
        probs = res.get("probabilidades", {})
        for cls_name, p in probs.items():
            label_fmt = cls_name.replace("_", " ").title()
            st.progress(float(p), text=f"{label_fmt}: {p*100:.1f}%")

with tab_cam:
    st.write("Toma una fotografía directa de un residuo usando la cámara de tu equipo:")
    cam_picture = st.camera_input("Capturar con Cámara")
    umbral_cam = st.slider("Umbral de confianza", 0.30, 0.90, 0.50, 0.05, key="slider_cam")

    if cam_picture is not None:
        pil_img = Image.open(io.BytesIO(cam_picture.read())).convert("RGB")
        mostrar_resultado(pil_img, umbral_cam)

with tab_upload:
    st.write("Sube una imagen desde tus archivos locales (JPG, PNG):")
    uploaded = st.file_uploader("Elige una imagen de residuo", type=["jpg", "jpeg", "png"])
    umbral_upload = st.slider("Umbral de confianza", 0.30, 0.90, 0.50, 0.05, key="slider_upload")

    if uploaded is not None:
        pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        mostrar_resultado(pil_img, umbral_upload)

with tab_live:
    st.markdown("""
    ### ⚡ Modo Video en Tiempo Real Continuo (30-60 FPS)
    Para detección continua frame a frame sin necesidad de tomar fotos individuales, EcoVision incluye el nuevo servidor interactivo de alta velocidad:
    
    1. **Servidor Web Ultrarrápido:**
       ```bash
       python web_app.py
       ```
       *Abre una interfaz futurista en tu navegador en `http://localhost:8000` con streaming continuo de cámara, síntesis de voz automática ("Botella plástica - Contenedor Blanco") e iluminación en tiempo real de los 3 contenedores.*

    2. **Visor de Escritorio Nativo OpenCV:**
       ```bash
       python live_cam.py
       ```
       *Abre una ventana local de OpenCV con HUD sobrepuesto, recuadro de escaneo y FPS en vivo.*
    """)

st.markdown("---")
st.caption("EcoVision 2.0 • Redes Neuronales Artificiales • MobileNetV2 + Resolución 2184 de 2019 de Colombia")
