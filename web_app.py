"""
web_app.py — EcoVision AI con Enlace a Celular y Visión Ultrarrápida
Características:
1. Soporte para cámara del PC o Celular vinculado por Wi-Fi (vía QR /remote)
2. Control de volteo horizontal (Efecto espejo ON/OFF)
3. Latencia reducida con compresión de imágenes y streaming optimizado
4. Clasificación multimodal Gemini Vision + Fallback local MobileNetV2
5. Asignación a Canecas Blanca, Verde y Negra (Resolución 2184 de 2019)
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json
import base64
import time
import socket
import asyncio
from typing import Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from classifier import EcoVisionClassifier
from gemini_vision import analizar_con_gemini

# Detección dinámica de IP local para enlazar celular
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

LOCAL_IP = get_local_ip()
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

def load_env_file():
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")

def save_env_key(key: str, value: str):
    lines = []
    found = False
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}=") or line.strip() == key:
            lines[i] = f"{key}={value}\n"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}\n")
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

load_env_file()

app = FastAPI(title="EcoVision - Clasificador Inteligente con IA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = EcoVisionClassifier()

# Gestor de conexiones para el enlace de cámara del celular
class PhoneStreamManager:
    def __init__(self):
        self.phone_sockets: List[WebSocket] = []
        self.desktop_sockets: List[WebSocket] = []
        self.last_frame: Optional[str] = None
        self.has_phone_connected: bool = False

    async def connect_phone(self, ws: WebSocket):
        await ws.accept()
        self.phone_sockets.append(ws)
        self.has_phone_connected = True
        await self.broadcast_status_to_desktop(True)

    def disconnect_phone(self, ws: WebSocket):
        if ws in self.phone_sockets:
            self.phone_sockets.remove(ws)
        self.has_phone_connected = len(self.phone_sockets) > 0

    async def connect_desktop(self, ws: WebSocket):
        await ws.accept()
        self.desktop_sockets.append(ws)
        await ws.send_json({"type": "phone_status", "connected": self.has_phone_connected})

    def disconnect_desktop(self, ws: WebSocket):
        if ws in self.desktop_sockets:
            self.desktop_sockets.remove(ws)

    async def relay_frame_to_desktop(self, frame_b64: str):
        self.last_frame = frame_b64
        dead_sockets = []
        msg = json.dumps({"type": "frame", "image": frame_b64})
        for ws in self.desktop_sockets:
            try:
                await ws.send_text(msg)
            except Exception:
                dead_sockets.append(ws)
        for ws in dead_sockets:
            self.disconnect_desktop(ws)

    async def broadcast_status_to_desktop(self, status: bool):
        msg = json.dumps({"type": "phone_status", "connected": status})
        for ws in self.desktop_sockets:
            try:
                await ws.send_text(msg)
            except Exception:
                pass

stream_manager = PhoneStreamManager()

class PredictRequest(BaseModel):
    image: str
    umbral: float = 0.50
    engine: str = "gemini"  # "gemini" | "local"
    api_key: Optional[str] = None

class SaveKeyRequest(BaseModel):
    api_key: str

@app.post("/api/save_key")
async def save_api_key(req: SaveKeyRequest):
    try:
        key = req.api_key.strip()
        os.environ["GEMINI_API_KEY"] = key
        save_env_key("GEMINI_API_KEY", key)
        return {"status": "ok", "message": "API Key guardada correctamente."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/config")
async def get_config():
    key = os.getenv("GEMINI_API_KEY", "")
    has_key = bool(key and len(key) > 5)
    masked = f"{key[:6]}...{key[-4:]}" if has_key and len(key) > 10 else ("***" if has_key else "")
    return {
        "has_gemini_key": has_key,
        "masked_key": masked,
        "default_engine": "gemini" if has_key else "local",
        "local_ip": LOCAL_IP,
        "remote_url": f"http://{LOCAL_IP}:8000/remote"
    }

@app.post("/api/predict")
async def predict_frame(req: PredictRequest):
    t0 = time.time()
    effective_key = req.api_key or os.getenv("GEMINI_API_KEY")

    # MODO 1: GEMINI VISION AI
    if req.engine == "gemini" and effective_key:
        try:
            res = await asyncio.to_thread(analizar_con_gemini, req.image, effective_key)
            if "error" not in res and res.get("detectado") is not False:
                res["tiempo_ms"] = round((time.time() - t0) * 1000, 1)
                return res
            elif "error" in res:
                print(f"[EcoVision] Fallback a local por: {res.get('error')}")
                fallback = await asyncio.to_thread(classifier.predict_base64, req.image, req.umbral)
                fallback["aviso_fallback"] = res.get("error")
                fallback["tiempo_ms"] = round((time.time() - t0) * 1000, 1)
                return fallback
        except Exception as e:
            print(f"[EcoVision] Error en Gemini: {e}")

    # MODO 2: LOCAL MOBILENETV2
    local_res = await asyncio.to_thread(classifier.predict_base64, req.image, req.umbral)
    local_res["tiempo_ms"] = round((time.time() - t0) * 1000, 1)
    local_res["material"] = "Clasificado por Red Neuronal Local"
    local_res["por_que"] = "Clasificación local basada en MobileNetV2 y reglas de negocio según la Resolución 2184 de 2019."
    return local_res

# WebSocket para recibir video desde el celular y retransmitir a la PC
@app.websocket("/ws/phone_cam")
async def phone_websocket_endpoint(websocket: WebSocket):
    role = websocket.query_params.get("role", "phone")
    if role == "phone":
        await stream_manager.connect_phone(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                if data:
                    await stream_manager.relay_frame_to_desktop(data)
        except WebSocketDisconnect:
            stream_manager.disconnect_phone(websocket)
            await stream_manager.broadcast_status_to_desktop(False)
    else:
        # Rol desktop listener
        await stream_manager.connect_desktop(websocket)
        try:
            while True:
                # Mantener conexión viva con ping
                await websocket.receive_text()
        except WebSocketDisconnect:
            stream_manager.disconnect_desktop(websocket)

# ==================== INTERFAZ MÓVIL (/remote) ====================
REMOTE_HTML = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>EcoVision — Cámara Móvil HD</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/lucide@latest"></script>
</head>
<body class="bg-black text-white h-screen flex flex-col justify-between overflow-hidden">
  
  <!-- Encabezado Móvil -->
  <header class="p-3 bg-gray-950/80 backdrop-blur-md flex items-center justify-between z-10 border-b border-gray-800">
    <div class="flex items-center space-x-2">
      <div class="w-3 h-3 rounded-full bg-emerald-400 animate-pulse"></div>
      <span class="font-bold text-sm text-emerald-400">EcoVision Móvil</span>
    </div>
    <div class="flex items-center space-x-2">
      <button onclick="toggleTorch()" id="torchBtn" class="px-3 py-1 bg-gray-800 rounded-lg text-xs font-semibold flex items-center space-x-1 border border-gray-700">
        <i data-lucide="zap" class="w-3.5 h-3.5 text-yellow-400"></i>
        <span>Linterna</span>
      </button>
      <span id="connStatus" class="text-xs px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 font-mono">Conectado</span>
    </div>
  </header>

  <!-- Visor de Cámara / Modo Video Continuo -->
  <div id="phoneVideoContainer" class="relative flex-1 flex items-center justify-center overflow-hidden">
    <video id="phoneVideo" autoplay playsinline muted class="w-full h-full object-cover"></video>
    <canvas id="sendCanvas" class="hidden"></canvas>
    
    <!-- Retícula guía en móvil -->
    <div class="absolute inset-x-8 inset-y-16 border-2 border-dashed border-emerald-400/70 rounded-2xl pointer-events-none flex items-center justify-center">
      <span class="text-xs font-bold text-emerald-300 bg-black/60 px-3 py-1 rounded-full uppercase tracking-wider">Apunta el residuo aquí</span>
    </div>
  </div>

  <!-- Vista Modo Foto Nativa (Funciona 100% sobre HTTP sin restricciones) -->
  <div id="fallbackView" class="hidden flex-1 flex flex-col items-center justify-center p-6 text-center space-y-4">
    <div class="w-20 h-20 rounded-3xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shadow-lg shadow-emerald-500/10">
      <i data-lucide="camera" class="w-10 h-10"></i>
    </div>
    <div>
      <h2 class="text-lg font-bold text-white">Cámara Móvil HD Conectada</h2>
      <p class="text-xs text-gray-400 mt-1 max-w-xs">Toma fotos de alta resolución de cualquier residuo para enviarlas instantáneamente a la pantalla del computador.</p>
    </div>

    <!-- Botón Grande de Captura con Cámara Nativa -->
    <label class="w-full max-w-xs py-4 px-6 rounded-2xl bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-white font-bold text-sm shadow-xl shadow-emerald-600/30 flex items-center justify-center space-x-2 cursor-pointer active:scale-95 transition">
      <i data-lucide="aperture" class="w-5 h-5"></i>
      <span>Tomar Foto del Residuo</span>
      <input type="file" id="nativePhotoInput" accept="image/*" capture="environment" class="hidden" onchange="handleNativePhoto(this)">
    </label>

    <!-- Miniatura de última foto enviada -->
    <div id="lastSentBox" class="hidden flex items-center space-x-2 p-2 rounded-xl bg-gray-900 border border-emerald-500/30">
      <img id="lastSentThumb" class="w-10 h-10 rounded-lg object-cover border border-gray-700" alt="Última foto">
      <div class="text-left text-xs">
        <p class="text-emerald-400 font-bold">¡Foto enviada al PC!</p>
        <p class="text-[10px] text-gray-500">Analizando en pantalla...</p>
      </div>
    </div>

    <!-- Guía para video continuo -->
    <div class="p-3 rounded-xl bg-gray-950/80 border border-gray-800 text-[11px] text-gray-400 text-left max-w-xs space-y-1">
      <p class="font-bold text-cyan-300">💡 ¿Prefieres transmisión de video en vivo?</p>
      <p class="text-[10px] leading-relaxed">Chrome en Android exige habilitar la red local: escribe <b>chrome://flags</b> en tu celular, busca <i>Insecure origins</i>, añade <b>http://{LOCAL_IP}:8000</b> y reinicia Chrome.</p>
    </div>
  </div>

  <!-- Barra de Acciones Móvil -->
  <footer class="p-4 bg-gray-950/90 backdrop-blur-md flex flex-col space-y-2 z-10 border-t border-gray-800">
    <div class="flex items-center justify-between text-xs text-gray-400">
      <span>Enlazado con EcoVision PC</span>
      <span id="fpsCounter" class="font-mono text-cyan-400 font-bold">Listo</span>
    </div>
  </footer>

  <script>
    lucide.createIcons();
    const video = document.getElementById("phoneVideo");
    const canvas = document.getElementById("sendCanvas");
    const ctx = canvas.getContext("2d");
    let ws = null;
    let track = null;
    let isTorchOn = false;
    let framesSent = 0;
    let lastFpsTime = performance.now();

    function connectWebSocket() {{
      const loc = window.location;
      const wsUrl = (loc.protocol === "https:" ? "wss://" : "ws://") + loc.host + "/ws/phone_cam?role=phone";
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {{
        document.getElementById("connStatus").innerText = "Conectado";
        document.getElementById("connStatus").className = "text-xs px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 font-mono";
      }};

      ws.onclose = () => {{
        document.getElementById("connStatus").innerText = "Reconectando...";
        document.getElementById("connStatus").className = "text-xs px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 font-mono";
        setTimeout(connectWebSocket, 2000);
      }};
    }}

    async function initPhoneCamera() {{
      connectWebSocket();

      // Si el navegador móvil bloquea getUserMedia por ser HTTP
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
        document.getElementById("phoneVideoContainer").classList.add("hidden");
        document.getElementById("fallbackView").classList.remove("hidden");
        document.getElementById("torchBtn").classList.add("hidden");
        return;
      }}

      try {{
        const stream = await navigator.mediaDevices.getUserMedia({{
          video: {{
            facingMode: {{ ideal: "environment" }},
            width: {{ ideal: 1280 }},
            height: {{ ideal: 720 }}
          }}
        }});
        video.srcObject = stream;
        track = stream.getVideoTracks()[0];
        startStreaming();
      }} catch (err) {{
        document.getElementById("phoneVideoContainer").classList.add("hidden");
        document.getElementById("fallbackView").classList.remove("hidden");
        document.getElementById("torchBtn").classList.add("hidden");
      }}
    }}

    function handleNativePhoto(input) {{
      if (input.files && input.files[0]) {{
        const file = input.files[0];
        const reader = new FileReader();
        reader.onload = (e) => {{
          const img = new Image();
          img.onload = () => {{
            // Redimensionar para envío instantáneo sin retrasos
            const maxDim = 800;
            let w = img.width;
            let h = img.height;
            if (w > maxDim || h > maxDim) {{
              if (w > h) {{
                h = Math.round((h * maxDim) / w);
                w = maxDim;
              }} else {{
                w = Math.round((w * maxDim) / h);
                h = maxDim;
              }}
            }}
            canvas.width = w;
            canvas.height = h;
            ctx.drawImage(img, 0, 0, w, h);
            const b64 = canvas.toDataURL("image/jpeg", 0.75);

            if (ws && ws.readyState === WebSocket.OPEN) {{
              ws.send(b64);
            }}

            document.getElementById("lastSentThumb").src = b64;
            document.getElementById("lastSentBox").classList.remove("hidden");
            document.getElementById("fpsCounter").innerText = "Foto enviada";
          }};
          img.src = e.target.result;
        }};
        reader.readAsDataURL(file);
      }}
    }}

    function startStreaming() {{
      setInterval(() => {{
        if (!ws || ws.readyState !== WebSocket.OPEN || video.videoWidth === 0) return;
        canvas.width = 640;
        canvas.height = 480;
        ctx.drawImage(video, 0, 0, 640, 480);
        const b64 = canvas.toDataURL("image/jpeg", 0.65);
        ws.send(b64);
        
        framesSent++;
        const now = performance.now();
        if (now - lastFpsTime >= 1000) {{
          document.getElementById("fpsCounter").innerText = framesSent + " FPS";
          framesSent = 0;
          lastFpsTime = now;
        }}
      }}, 65);
    }}

    async function toggleTorch() {{
      if (!track) return;
      try {{
        isTorchOn = !isTorchOn;
        await track.applyConstraints({{ advanced: [{{ torch: isTorchOn }}] }});
        document.getElementById("torchBtn").className = isTorchOn ? "px-3 py-1 bg-yellow-500 text-black rounded-lg text-xs font-bold flex items-center space-x-1" : "px-3 py-1 bg-gray-800 rounded-lg text-xs font-semibold flex items-center space-x-1 border border-gray-700";
      }} catch (e) {{
        alert("Tu dispositivo no soporta linterna desde el navegador.");
      }}
    }}

    window.addEventListener("DOMContentLoaded", initPhoneCamera);
  </script>
</body>
</html>
"""

# ==================== INTERFAZ PRINCIPAL DE ESCRITORIO ====================
DESKTOP_HTML = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EcoVision — Clasificador de Residuos AI con Celular y Webcam</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="https://unpkg.com/lucide@latest"></script>
  <style>
    body {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
    .glass-card {{
      background: rgba(15, 23, 42, 0.78);
      backdrop-filter: blur(20px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .scan-reticle {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 70%;
      height: 70%;
      border: 2px dashed rgba(14, 165, 233, 0.7);
      border-radius: 1.5rem;
      pointer-events: none;
      box-shadow: 0 0 35px rgba(14, 165, 233, 0.2);
      animation: pulse-reticle 2.2s infinite ease-in-out;
    }}
    @keyframes pulse-reticle {{
      0%, 100% {{ border-color: rgba(14, 165, 233, 0.6); box-shadow: 0 0 25px rgba(14, 165, 233, 0.2); }}
      50% {{ border-color: rgba(16, 185, 129, 0.9); box-shadow: 0 0 45px rgba(16, 185, 129, 0.4); }}
    }}
    .scan-line {{
      position: absolute;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, transparent, #38bdf8, #10b981, transparent);
      box-shadow: 0 0 15px #38bdf8;
      animation: scan-move 2.4s infinite linear;
    }}
    @keyframes scan-move {{
      0% {{ top: 0%; opacity: 0; }}
      15% {{ opacity: 1; }}
      85% {{ opacity: 1; }}
      100% {{ top: 100%; opacity: 0; }}
    }}
    .video-mirror {{
      transform: scaleX(-1);
    }}
    .bin-active-blanco {{
      border-color: #60a5fa !important;
      box-shadow: 0 0 40px rgba(96, 165, 250, 0.5);
      transform: translateY(-4px) scale(1.03);
    }}
    .bin-active-verde {{
      border-color: #34d399 !important;
      box-shadow: 0 0 40px rgba(52, 211, 153, 0.5);
      transform: translateY(-4px) scale(1.03);
    }}
    .bin-active-negro {{
      border-color: #9ca3af !important;
      box-shadow: 0 0 40px rgba(156, 163, 175, 0.5);
      transform: translateY(-4px) scale(1.03);
    }}
  </style>
</head>
<body class="bg-[#080c14] text-gray-100 min-h-screen flex flex-col selection:bg-emerald-500 selection:text-white">

  <!-- TOPBAR -->
  <header class="border-b border-gray-800/80 bg-[#0d1322]/90 sticky top-0 z-50 backdrop-blur-xl">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      
      <!-- LOGO -->
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 via-teal-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/25">
          <i data-lucide="sparkles" class="w-6 h-6 text-white"></i>
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <span class="text-xl font-black tracking-tight bg-gradient-to-r from-emerald-400 via-cyan-300 to-blue-400 bg-clip-text text-transparent">EcoVision AI</span>
            <span class="text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">Ultra Speed</span>
          </div>
          <p class="text-xs text-gray-400">Resolución 2184 de 2019 • Canecas Blanca, Verde y Negra</p>
        </div>
      </div>

      <!-- SELECTOR DE MOTOR, VINCULAR CELULAR & CONTROLES -->
      <div class="flex items-center space-x-2 sm:space-x-3">
        
        <!-- Botón Enlazar Celular -->
        <button onclick="openPhoneModal()" class="px-3 py-1.5 rounded-xl bg-gradient-to-r from-blue-600/30 to-cyan-600/30 hover:from-blue-600/40 hover:to-cyan-600/40 text-cyan-300 border border-cyan-500/40 text-xs font-bold flex items-center space-x-1.5 transition shadow-sm">
          <i data-lucide="smartphone" class="w-4 h-4 text-cyan-400"></i>
          <span id="phoneBtnText">Enlazar Celular</span>
          <span id="phoneDot" class="w-2 h-2 rounded-full bg-gray-500"></span>
        </button>

        <!-- Selector de Motor (Gemini vs Local) -->
        <div class="hidden sm:flex items-center bg-gray-900/90 p-1 rounded-xl border border-gray-800 text-xs">
          <button id="btnEngineGemini" onclick="setEngine('gemini')" class="px-3 py-1.5 rounded-lg font-semibold flex items-center space-x-1.5 transition bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
            <i data-lucide="bot" class="w-3.5 h-3.5"></i>
            <span>Gemini Vision</span>
          </button>
          <button id="btnEngineLocal" onclick="setEngine('local')" class="px-3 py-1.5 rounded-lg font-semibold flex items-center space-x-1.5 text-gray-400 hover:text-white transition">
            <i data-lucide="cpu" class="w-3.5 h-3.5"></i>
            <span>Local</span>
          </button>
        </div>

        <!-- Botón Configurar API Key -->
        <button onclick="openKeyModal()" class="px-3 py-1.5 rounded-xl bg-gray-800 hover:bg-gray-700 text-xs font-semibold text-gray-200 border border-gray-700 flex items-center space-x-1.5 transition">
          <i data-lucide="key" class="w-4 h-4 text-amber-400"></i>
          <span id="keyStatusText">API Key</span>
        </button>

        <!-- Toggle Voz -->
        <button id="audioToggleBtn" onclick="toggleAudio()" class="px-3 py-1.5 rounded-xl bg-gray-800 hover:bg-gray-700 text-xs font-semibold text-gray-200 border border-gray-700 flex items-center space-x-1.5 transition">
          <i data-lucide="volume-2" class="w-4 h-4 text-emerald-400" id="audioIcon"></i>
          <span id="audioStatusText" class="hidden sm:inline">Voz: ON</span>
        </button>

      </div>
    </div>
  </header>

  <!-- CONTENIDO PRINCIPAL -->
  <main class="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-12 gap-6">

    <!-- COLUMNA IZQUIERDA: CÁMARA (7 columnas) -->
    <div class="lg:col-span-7 flex flex-col space-y-4">
      
      <div class="glass-card rounded-2xl p-4 border border-gray-800/80 shadow-2xl flex flex-col relative overflow-hidden">
        
        <!-- Video Viewport -->
        <div class="relative bg-black rounded-xl overflow-hidden aspect-[4/3] flex items-center justify-center border border-gray-800 shadow-inner">
          
          <!-- Elemento de video del PC -->
          <video id="videoElement" autoplay playsinline muted class="w-full h-full object-cover video-mirror"></video>
          
          <!-- Imagen recibida desde el celular (cuando esté activo) -->
          <img id="phoneImgElement" class="w-full h-full object-cover hidden" alt="Feed celular">
          
          <canvas id="hiddenCanvas" width="640" height="480" class="hidden"></canvas>

          <!-- Retícula animada de enfoque -->
          <div id="reticle" class="scan-reticle">
            <div class="scan-line"></div>
            <div class="absolute top-3 left-3 text-[10px] uppercase font-bold tracking-wider text-cyan-400 bg-gray-950/80 px-2.5 py-1 rounded-lg border border-cyan-500/30 flex items-center space-x-1.5 shadow-lg">
              <span class="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
              <span id="sourceLabel">Cámara PC (Modo Espejo)</span>
            </div>
          </div>

          <!-- Overlay Badge Dentro del Video -->
          <div class="absolute bottom-4 left-4 right-4 flex items-center justify-between pointer-events-none">
            <div class="bg-gray-950/85 backdrop-blur-md px-3.5 py-2 rounded-xl border border-gray-700/80 flex items-center space-x-3 shadow-xl">
              <span id="liveIcon" class="text-2xl">🔍</span>
              <div>
                <p class="text-[10px] uppercase tracking-wider text-gray-400 font-bold" id="liveLabel">Listo para escanear</p>
                <p class="text-sm font-bold text-white leading-tight" id="liveObject">Coloca un residuo frente a la cámara</p>
              </div>
            </div>
            <div class="bg-gray-950/85 backdrop-blur-md px-3 py-1.5 rounded-xl border border-gray-700/80 text-right shadow-xl">
              <p class="text-[10px] text-gray-400">Tiempo Respuesta</p>
              <p class="text-xs font-mono font-bold text-cyan-400" id="latencyBadge">-- ms</p>
            </div>
          </div>

          <!-- Overlay Cámara Desactivada -->
          <div id="camInactiveOverlay" class="absolute inset-0 bg-gray-950/90 flex flex-col items-center justify-center text-center p-6 space-y-3 z-20">
            <div class="w-16 h-16 rounded-2xl bg-gray-800/80 flex items-center justify-center text-gray-400 border border-gray-700">
              <i data-lucide="camera-off" class="w-8 h-8 text-gray-400"></i>
            </div>
            <h3 class="text-lg font-bold text-white">Cámara Web en Pausa</h3>
            <p class="text-xs text-gray-400 max-w-sm">Activa la cámara o vincula tu celular para detectar cualquier residuo con alta resolución.</p>
            <div class="flex items-center space-x-3 mt-2">
              <button onclick="startCamera()" class="px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-white font-semibold text-xs shadow-lg shadow-emerald-600/30 transition flex items-center space-x-2">
                <i data-lucide="play" class="w-4 h-4"></i>
                <span>Activar Webcam PC</span>
              </button>
              <button onclick="openPhoneModal()" class="px-4 py-2.5 rounded-xl bg-blue-600/30 hover:bg-blue-600/50 text-cyan-300 border border-cyan-500/40 font-semibold text-xs transition flex items-center space-x-2">
                <i data-lucide="smartphone" class="w-4 h-4"></i>
                <span>Usar Celular</span>
              </button>
            </div>
          </div>
        </div>

        <!-- Barra de Controles Inferior -->
        <div class="mt-4 flex flex-wrap items-center justify-between gap-3">
          
          <div class="flex items-center space-x-2">
            <!-- Botón Escanear Ahora -->
            <button id="scanNowBtn" onclick="triggerScan(true)" class="px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-white text-xs font-bold flex items-center space-x-2 transition shadow-lg shadow-emerald-600/25">
              <i data-lucide="scan" class="w-4 h-4"></i>
              <span>Escanear Ahora (Espacio)</span>
            </button>
            
            <!-- Botón Voltear Cámara / Espejo Horizontal -->
            <button id="mirrorBtn" onclick="toggleMirror()" class="px-3 py-2.5 rounded-xl bg-cyan-600/20 hover:bg-cyan-600/30 text-cyan-300 border border-cyan-500/30 text-xs font-semibold flex items-center space-x-1.5 transition" title="Cambia la orientación si al moverte a la derecha se mueve al revés">
              <i data-lucide="flip-horizontal" class="w-4 h-4"></i>
              <span id="mirrorStatusText">Espejo: ON</span>
            </button>

            <!-- Botón Pausar Cámara -->
            <button id="camToggleBtn" onclick="toggleCamera()" class="px-3.5 py-2.5 rounded-xl bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs font-semibold flex items-center space-x-1.5 transition border border-gray-700">
              <i data-lucide="camera" class="w-4 h-4"></i>
              <span id="camToggleText">Pausar</span>
            </button>
          </div>

          <!-- Selector de Fuente (PC vs Celular) -->
          <div class="flex items-center space-x-2 bg-gray-900/90 px-3 py-1.5 rounded-xl border border-gray-800 text-xs">
            <span class="text-gray-400">Origen:</span>
            <select id="sourceSelector" onchange="changeCameraSource(this.value)" class="bg-gray-800 text-gray-200 border border-gray-700 rounded-lg px-2 py-1 text-xs focus:outline-none">
              <option value="pc">💻 Cámara PC</option>
              <option value="phone">📱 Celular HD</option>
            </select>
          </div>

        </div>

      </div>

      <!-- HISTORIAL DE RESIDUOS -->
      <div class="glass-card rounded-2xl p-4 border border-gray-800/80">
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center space-x-2">
            <i data-lucide="history" class="w-4 h-4 text-cyan-400"></i>
            <h4 class="text-xs font-bold text-gray-300 uppercase tracking-wider">Historial de Residuos Clasificados</h4>
          </div>
          <span id="historyCount" class="text-[11px] text-gray-500 font-mono">0 registros</span>
        </div>
        <div id="historyList" class="max-h-36 overflow-y-auto space-y-2 pr-1 text-xs">
          <p class="text-gray-600 text-center py-4">Los objetos analizados aparecerán aquí.</p>
        </div>
      </div>

    </div>

    <!-- COLUMNA DERECHA: CANECA ASIGNADA & LECTURA MEJORADA (5 columnas) -->
    <div class="lg:col-span-5 flex flex-col space-y-4">
      
      <!-- TARJETA PRINCIPAL: CANECA -->
      <div class="glass-card rounded-2xl p-5 border border-gray-800/80 shadow-2xl relative overflow-hidden">
        
        <div class="flex items-center justify-between border-b border-gray-800 pb-3 mb-4">
          <div class="flex items-center space-x-2">
            <i data-lucide="recycle" class="w-5 h-5 text-emerald-400"></i>
            <span class="text-xs font-bold uppercase tracking-wider text-gray-300">Caneca Asignada</span>
          </div>
          <span class="text-[10px] text-gray-400 font-semibold px-2 py-0.5 rounded bg-gray-800 border border-gray-700" id="engineBadge">Gemini Vision AI</span>
        </div>

        <!-- Banner de Caneca Dinámica -->
        <div id="containerCard" class="rounded-2xl p-5 border-2 transition-all duration-300 bg-gray-900/90 border-gray-700 flex flex-col space-y-3">
          
          <div class="flex items-start justify-between">
            <div>
              <span id="targetBinTag" class="text-[11px] uppercase tracking-wider font-black px-3 py-1 rounded-full bg-gray-800 text-gray-300 inline-block mb-1 border border-gray-700">Contenedor</span>
              <h2 id="targetBinName" class="text-2xl font-black text-white tracking-tight">Esperando residuo...</h2>
              <p id="targetBinType" class="text-xs text-gray-400 font-medium">Apunta el residuo con la cámara</p>
            </div>
            <div id="targetBinIconBox" class="w-14 h-14 rounded-2xl bg-gray-800 border border-gray-700 flex items-center justify-center text-3xl shadow-inner">
              ♻️
            </div>
          </div>

          <!-- Material y Confianza -->
          <div class="grid grid-cols-2 gap-2 pt-1 border-t border-white/5">
            <div>
              <p class="text-[10px] text-gray-400">Material Detectado:</p>
              <p class="text-xs font-semibold text-gray-200" id="targetMaterial">Pendiente</p>
            </div>
            <div>
              <p class="text-[10px] text-gray-400">Certeza IA:</p>
              <p class="text-xs font-semibold text-emerald-400 font-mono" id="targetConf">0.0%</p>
            </div>
          </div>

          <!-- Instrucción de Preparación -->
          <div class="bg-black/40 rounded-xl p-3 border border-white/5 flex items-start space-x-2.5">
            <i data-lucide="check-circle" class="w-4 h-4 text-emerald-400 mt-0.5 shrink-0"></i>
            <p id="targetInstruction" class="text-xs text-gray-300 leading-relaxed">
              Muestra una botella, lata, papel, caja o resto de comida frente al lente.
            </p>
          </div>

        </div>

        <!-- LAS 3 CANECAS DE LA RESOLUCIÓN 2184 -->
        <div class="grid grid-cols-3 gap-3 mt-4">
          
          <!-- BLANCA -->
          <div id="binBlanco" class="p-3 rounded-xl border-2 border-gray-800 bg-gray-900/60 flex flex-col items-center text-center transition-all duration-300">
            <div class="w-8 h-8 rounded-lg bg-gray-100 text-gray-900 font-black text-xs flex items-center justify-center shadow-md">
              <i data-lucide="recycle" class="w-5 h-5 text-gray-900"></i>
            </div>
            <span class="text-xs font-bold text-white mt-2">BLANCA</span>
            <span class="text-[10px] text-gray-400 mt-0.5 leading-tight">Aprovechables (Plástico, Latas, Papel, Vidrio)</span>
          </div>

          <!-- VERDE -->
          <div id="binVerde" class="p-3 rounded-xl border-2 border-gray-800 bg-gray-900/60 flex flex-col items-center text-center transition-all duration-300">
            <div class="w-8 h-8 rounded-lg bg-emerald-500 text-white font-black text-xs flex items-center justify-center shadow-md">
              <i data-lucide="apple" class="w-5 h-5 text-white"></i>
            </div>
            <span class="text-xs font-bold text-white mt-2">VERDE</span>
            <span class="text-[10px] text-gray-400 mt-0.5 leading-tight">Orgánicos (Cáscaras, Restos de comida)</span>
          </div>

          <!-- NEGRA -->
          <div id="binNegro" class="p-3 rounded-xl border-2 border-gray-800 bg-gray-900/60 flex flex-col items-center text-center transition-all duration-300">
            <div class="w-8 h-8 rounded-lg bg-gray-950 border border-gray-700 text-white font-black text-xs flex items-center justify-center shadow-md">
              <i data-lucide="trash-2" class="w-5 h-5 text-gray-400"></i>
            </div>
            <span class="text-xs font-bold text-white mt-2">NEGRA</span>
            <span class="text-[10px] text-gray-400 mt-0.5 leading-tight">No aprovechables (Servilletas, sucios)</span>
          </div>

        </div>

      </div>

      <!-- TARJETA: ¿POR QUÉ VA EN ESTA CANECA? (LECTURA MEJORADA) -->
      <div class="glass-card rounded-2xl p-5 border border-gray-800/80 shadow-2xl flex flex-col space-y-3">
        <div class="flex items-center space-x-2">
          <i data-lucide="file-check" class="w-4 h-4 text-cyan-400"></i>
          <h4 class="text-xs font-bold text-gray-300 uppercase tracking-wider">Dictamen Normativo (Res. 2184)</h4>
        </div>
        <p id="targetWhy" class="text-xs text-gray-300 leading-relaxed bg-gray-900/80 p-3.5 rounded-xl border border-gray-800">
          La inteligencia artificial evalúa el residuo según su material, estado de limpieza y si es apto para reciclaje o compostaje en Colombia.
        </p>
      </div>

    </div>

  </main>

  <!-- MODAL: ENLAZAR CÁMARA DEL CELULAR -->
  <div id="phoneModal" class="fixed inset-0 bg-black/80 backdrop-blur-md hidden items-center justify-center z-50 p-4">
    <div class="glass-card rounded-2xl max-w-md w-full p-6 border border-gray-700 shadow-2xl flex flex-col space-y-4">
      
      <div class="flex items-center justify-between border-b border-gray-800 pb-3">
        <div class="flex items-center space-x-2">
          <i data-lucide="smartphone" class="w-5 h-5 text-cyan-400"></i>
          <h3 class="text-base font-bold text-white">Vincular Celular como Cámara HD</h3>
        </div>
        <button onclick="closePhoneModal()" class="text-gray-400 hover:text-white">
          <i data-lucide="x" class="w-5 h-5"></i>
        </button>
      </div>

      <p class="text-xs text-gray-300 leading-relaxed">
        Usa la cámara trasera de tu smartphone para obtener <b>máxima nitidez y enfoque perfecto</b> sobre los residuos.
      </p>

      <!-- Código QR -->
      <div class="flex flex-col items-center justify-center p-4 bg-white rounded-2xl shadow-lg">
        <img id="qrImage" src="https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=http://{LOCAL_IP}:8000/remote" alt="QR Celular" class="w-48 h-48 rounded-lg">
        <p class="text-[11px] text-gray-600 font-mono mt-2 font-bold">http://{LOCAL_IP}:8000/remote</p>
      </div>

      <div class="bg-gray-900/80 p-3 rounded-xl border border-gray-800 text-xs text-gray-300 space-y-1">
        <p class="font-bold text-cyan-300">📱 Pasos para conectar:</p>
        <p>1. Asegúrate de que el celular esté en la misma red Wi-Fi.</p>
        <p>2. Abre la cámara del celular y escanea el código QR de arriba.</p>
        <p>3. ¡Listo! La imagen de tu celular se transmitirá directamente a esta pantalla.</p>
      </div>

      <div class="flex items-center justify-end pt-2">
        <button onclick="closePhoneModal()" class="px-5 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold">Entendido</button>
      </div>

    </div>
  </div>

  <!-- MODAL: CONFIGURAR API KEY GEMINI -->
  <div id="keyModal" class="fixed inset-0 bg-black/80 backdrop-blur-md hidden items-center justify-center z-50 p-4">
    <div class="glass-card rounded-2xl max-w-md w-full p-6 border border-gray-700 shadow-2xl flex flex-col space-y-4">
      
      <div class="flex items-center justify-between border-b border-gray-800 pb-3">
        <div class="flex items-center space-x-2">
          <i data-lucide="key" class="w-5 h-5 text-amber-400"></i>
          <h3 class="text-base font-bold text-white">Configurar Gemini API Key</h3>
        </div>
        <button onclick="closeKeyModal()" class="text-gray-400 hover:text-white">
          <i data-lucide="x" class="w-5 h-5"></i>
        </button>
      </div>

      <p class="text-xs text-gray-300 leading-relaxed">
        Ingresa tu clave de Gemini para activar la visión multimodal ultrarrápida (detecta marcas, materiales exactos y estado del residuo).
      </p>

      <div>
        <label class="block text-xs font-semibold text-gray-300 mb-1">Clave de API de Gemini:</label>
        <input type="password" id="inputApiKey" placeholder="AIzaSy..." class="w-full bg-gray-900 border border-gray-700 rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-emerald-500 font-mono">
        <a href="https://aistudio.google.com/app/apikey" target="_blank" class="inline-block text-[11px] text-cyan-400 hover:underline mt-1.5">
          👉 Obtener API Key gratis en Google AI Studio
        </a>
      </div>

      <div class="flex items-center justify-end space-x-2 pt-2 border-t border-gray-800">
        <button onclick="closeKeyModal()" class="px-4 py-2 rounded-xl bg-gray-800 hover:bg-gray-700 text-xs font-semibold text-gray-300">Cancelar</button>
        <button onclick="saveApiKey()" class="px-5 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold">Guardar y Conectar</button>
      </div>

    </div>
  </div>

  <!-- SCRIPT LOGIC -->
  <script>
    lucide.createIcons();

    const video = document.getElementById("videoElement");
    const phoneImg = document.getElementById("phoneImgElement");
    const canvas = document.getElementById("hiddenCanvas");
    const ctx = canvas.getContext("2d");
    
    let stream = null;
    let isCameraRunning = false;
    let isMirror = true; // Por defecto modo espejo activado
    let currentSource = "pc"; // "pc" | "phone"
    let currentEngine = "gemini";
    let storedApiKey = localStorage.getItem("GEMINI_API_KEY") || "";
    let voiceEnabled = true;
    let isPredicting = false;
    let phoneWebSocket = null;
    let isPhoneStreaming = false;
    let historyItems = [];
    let autoScanTimer = null;
    let lastSpokenClass = "";
    let lastSpokenTime = 0;

    // Configuración inicial
    async function initApp() {{
      try {{
        const resp = await fetch("/api/config");
        const cfg = await resp.json();
        if (cfg.has_gemini_key) {{
          document.getElementById("keyStatusText").innerText = "Gemini: OK";
          document.getElementById("keyStatusText").className = "text-emerald-400 font-semibold";
          setEngine("gemini");
        }} else if (storedApiKey) {{
          saveKeyToServer(storedApiKey);
        }} else {{
          document.getElementById("keyStatusText").innerText = "Sin Key";
          setEngine("local");
        }}
      }} catch (err) {{
        console.error(err);
      }}

      // Conectar WebSocket para escuchar celular
      initDesktopWebSocket();
      // Iniciar webcam del PC por defecto
      startCamera();
      // Iniciar ciclo de auto-escaneo rápido
      startAutoScanLoop();
    }}

    function initDesktopWebSocket() {{
      const loc = window.location;
      const wsUrl = (loc.protocol === "https:" ? "wss://" : "ws://") + loc.host + "/ws/phone_cam?role=desktop";
      phoneWebSocket = new WebSocket(wsUrl);

      phoneWebSocket.onmessage = (event) => {{
        try {{
          const msg = JSON.parse(event.data);
          if (msg.type === "phone_status") {{
            updatePhoneStatus(msg.connected);
          }} else if (msg.type === "frame") {{
            if (currentSource === "phone") {{
              phoneImg.src = msg.image;
              isPhoneStreaming = true;
              triggerScan(false);
            }}
          }}
        }} catch (e) {{
          console.error(e);
        }}
      }};

      phoneWebSocket.onclose = () => {{
        setTimeout(initDesktopWebSocket, 3000);
      }};
    }}

    function updatePhoneStatus(connected) {{
      const dot = document.getElementById("phoneDot");
      const btnText = document.getElementById("phoneBtnText");
      if (connected) {{
        dot.className = "w-2 h-2 rounded-full bg-emerald-400 animate-pulse";
        btnText.innerText = "Celular Vinculado";
        // Si el usuario no ha cambiado origen manualmente, ofrecer cambio a celular
        if (currentSource === "pc") {{
          changeCameraSource("phone");
        }}
      }} else {{
        dot.className = "w-2 h-2 rounded-full bg-gray-500";
        btnText.innerText = "Enlazar Celular";
        if (currentSource === "phone") {{
          changeCameraSource("pc");
        }}
      }}
    }}

    function changeCameraSource(src) {{
      currentSource = src;
      document.getElementById("sourceSelector").value = src;
      const label = document.getElementById("sourceLabel");

      if (src === "phone") {{
        video.classList.add("hidden");
        phoneImg.classList.remove("hidden");
        label.innerText = "Cámara Celular HD";
        document.getElementById("camInactiveOverlay").classList.add("hidden");
        document.getElementById("reticle").classList.remove("hidden");
      }} else {{
        phoneImg.classList.add("hidden");
        video.classList.remove("hidden");
        label.innerText = isMirror ? "Cámara PC (Modo Espejo)" : "Cámara PC (Normal)";
        if (!isCameraRunning) startCamera();
      }}
    }}

    // Control de Efecto Espejo / Volteo Horizontal
    function toggleMirror() {{
      isMirror = !isMirror;
      if (isMirror) {{
        video.classList.add("video-mirror");
        document.getElementById("mirrorStatusText").innerText = "Espejo: ON";
        document.getElementById("sourceLabel").innerText = "Cámara PC (Modo Espejo)";
      }} else {{
        video.classList.remove("video-mirror");
        document.getElementById("mirrorStatusText").innerText = "Espejo: OFF";
        document.getElementById("sourceLabel").innerText = "Cámara PC (Normal)";
      }}
    }}

    async function startCamera() {{
      try {{
        stream = await navigator.mediaDevices.getUserMedia({{
          video: {{ width: {{ ideal: 640 }}, height: {{ ideal: 480 }}, facingMode: "user" }}
        }});
        video.srcObject = stream;
        isCameraRunning = true;
        document.getElementById("camInactiveOverlay").classList.add("hidden");
        document.getElementById("reticle").classList.remove("hidden");
        document.getElementById("camToggleText").innerText = "Pausar";
      }} catch (err) {{
        console.warn("Webcam local ocupada o denegada: " + err.message);
      }}
    }}

    function stopCamera() {{
      isCameraRunning = false;
      if (stream) {{
        stream.getTracks().forEach(t => t.stop());
      }}
      document.getElementById("camInactiveOverlay").classList.remove("hidden");
      document.getElementById("reticle").classList.add("hidden");
      document.getElementById("camToggleText").innerText = "Cámara";
    }}

    function toggleCamera() {{
      if (isCameraRunning) stopCamera();
      else startCamera();
    }}

    function setEngine(engine) {{
      currentEngine = engine;
      const btnG = document.getElementById("btnEngineGemini");
      const btnL = document.getElementById("btnEngineLocal");
      const badge = document.getElementById("engineBadge");

      if (engine === "gemini") {{
        btnG.className = "px-3 py-1.5 rounded-lg font-semibold flex items-center space-x-1.5 transition bg-emerald-500/20 text-emerald-300 border border-emerald-500/40";
        btnL.className = "px-3 py-1.5 rounded-lg font-semibold flex items-center space-x-1.5 text-gray-400 hover:text-white transition";
        badge.innerText = "Gemini Vision AI";
        badge.className = "text-[10px] text-cyan-300 font-semibold px-2 py-0.5 rounded bg-cyan-950/70 border border-cyan-800";
      }} else {{
        btnL.className = "px-3 py-1.5 rounded-lg font-semibold flex items-center space-x-1.5 transition bg-blue-500/20 text-blue-300 border border-blue-500/40";
        btnG.className = "px-3 py-1.5 rounded-lg font-semibold flex items-center space-x-1.5 text-gray-400 hover:text-white transition";
        badge.innerText = "MobileNetV2 Local";
        badge.className = "text-[10px] text-blue-300 font-semibold px-2 py-0.5 rounded bg-blue-950/70 border border-blue-800";
      }}
    }}

    // Inferencia Ultrarrápida
    async function triggerScan(isManual = true) {{
      if (isPredicting) return;

      let b64 = "";
      if (currentSource === "phone") {{
        if (!phoneImg.src || phoneImg.src.length < 100) return;
        b64 = phoneImg.src;
      }} else {{
        if (!isCameraRunning || video.videoWidth === 0) return;
        canvas.width = 400;
        canvas.height = 300;
        ctx.drawImage(video, 0, 0, 400, 300);
        b64 = canvas.toDataURL("image/jpeg", 0.65);
      }}

      isPredicting = true;
      if (isManual) {{
        document.getElementById("liveLabel").innerText = "Analizando con IA...";
      }}

      const t0 = performance.now();
      try {{
        const resp = await fetch("/api/predict", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            image: b64,
            engine: currentEngine,
            api_key: storedApiKey || undefined,
            umbral: 0.50
          }})
        }});
        const data = await resp.json();
        const latency = Math.round(performance.now() - t0);
        document.getElementById("latencyBadge").innerText = latency + " ms";
        renderResult(data);
      }} catch (err) {{
        console.error("Error en detección:", err);
      }} finally {{
        isPredicting = false;
      }}
    }}

    function startAutoScanLoop() {{
      if (autoScanTimer) clearInterval(autoScanTimer);
      // Intervalo optimizado a 900ms para lecturas casi instantáneas
      autoScanTimer = setInterval(() => {{
        triggerScan(false);
      }}, 900);
    }}

    function renderResult(data) {{
      if (!data) return;

      const conf = data.confianza || 0.90;
      const confPct = Math.round(conf * 100) + "%";
      const nombre = data.nombre_comun || data.clase_final || "Objeto detectado";
      const contenedor = data.contenedor || "Negro";
      const material = data.material || "Residuo";
      const instruccion = data.instruccion || "Depositar en la caneca correspondiente.";
      const porQue = data.por_que || "Según la Resolución 2184 de 2019 de Colombia.";
      const icono = data.icono || "♻️";

      // Overlay cámara
      document.getElementById("liveIcon").innerText = icono;
      document.getElementById("liveLabel").innerText = "Caneca " + contenedor;
      document.getElementById("liveObject").innerText = nombre;

      // Tarjeta principal
      const card = document.getElementById("containerCard");
      const binName = document.getElementById("targetBinName");
      const binType = document.getElementById("targetBinType");
      const binTag = document.getElementById("targetBinTag");
      const binIcon = document.getElementById("targetBinIconBox");
      const matEl = document.getElementById("targetMaterial");
      const confEl = document.getElementById("targetConf");
      const instEl = document.getElementById("targetInstruction");
      const whyEl = document.getElementById("targetWhy");

      binName.innerText = "Caneca " + contenedor;
      binType.innerText = data.tipo_contenedor || "";
      binIcon.innerText = icono;
      matEl.innerText = material;
      confEl.innerText = confPct;
      instEl.innerText = instruccion;
      whyEl.innerText = porQue;

      // Resaltado de canecas 3D
      const bBlanco = document.getElementById("binBlanco");
      const bVerde = document.getElementById("binVerde");
      const bNegro = document.getElementById("binNegro");

      bBlanco.className = "p-3 rounded-xl border-2 border-gray-800 bg-gray-900/60 flex flex-col items-center text-center transition-all duration-300";
      bVerde.className = "p-3 rounded-xl border-2 border-gray-800 bg-gray-900/60 flex flex-col items-center text-center transition-all duration-300";
      bNegro.className = "p-3 rounded-xl border-2 border-gray-800 bg-gray-900/60 flex flex-col items-center text-center transition-all duration-300";

      if (contenedor === "Blanco") {{
        card.style.borderColor = "#60a5fa";
        binTag.className = "text-[11px] uppercase tracking-wider font-black px-3 py-1 rounded-full bg-blue-500/20 text-blue-300 inline-block mb-1 border border-blue-500/40";
        bBlanco.classList.add("bin-active-blanco");
      }} else if (contenedor === "Verde") {{
        card.style.borderColor = "#34d399";
        binTag.className = "text-[11px] uppercase tracking-wider font-black px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 inline-block mb-1 border border-emerald-500/40";
        bVerde.classList.add("bin-active-verde");
      }} else {{
        card.style.borderColor = "#6b7280";
        binTag.className = "text-[11px] uppercase tracking-wider font-black px-3 py-1 rounded-full bg-gray-700/50 text-gray-300 inline-block mb-1 border border-gray-600";
        bNegro.classList.add("bin-active-negro");
      }}

      // Agregar a historial si es objeto válido
      if (nombre !== "Esperando residuo..." && !nombre.includes("No Identificado") && !nombre.includes("Listo para")) {{
        addHistoryItem(nombre, contenedor, confPct);
      }}

      // Asistente de voz en español
      if (voiceEnabled && 'speechSynthesis' in window) {{
        const now = Date.now();
        if (data.clase_final !== "no_identificado" && data.clase_final !== lastSpokenClass && (now - lastSpokenTime > 3500)) {{
          lastSpokenClass = data.clase_final;
          lastSpokenTime = now;
          const text = `${{nombre}}. Caneca ${{contenedor}}.`;
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.lang = "es-CO";
          utterance.rate = 1.05;
          window.speechSynthesis.cancel();
          window.speechSynthesis.speak(utterance);
        }}
      }}
    }}

    function addHistoryItem(name, bin, conf) {{
      if (historyItems.length > 0 && historyItems[0].name === name) return;
      const timeStr = new Date().toLocaleTimeString([], {{ hour: '2-digit', minute: '2-digit', second: '2-digit' }});
      historyItems.unshift({{ name, bin, conf, timeStr }});
      if (historyItems.length > 10) historyItems.pop();

      document.getElementById("historyCount").innerText = historyItems.length + " registros";
      document.getElementById("historyList").innerHTML = historyItems.map(it => `
        <div class="flex items-center justify-between p-2 rounded-xl bg-gray-900/80 border border-gray-800">
          <div class="flex items-center space-x-2">
            <span class="w-1.5 h-1.5 rounded-full ${{it.bin === 'Blanco' ? 'bg-blue-400' : (it.bin === 'Verde' ? 'bg-emerald-400' : 'bg-gray-400')}}"></span>
            <span class="font-medium text-gray-200">${{it.name}}</span>
          </div>
          <div class="flex items-center space-x-3 text-[11px]">
            <span class="${{it.bin === 'Blanco' ? 'text-blue-400 font-bold' : (it.bin === 'Verde' ? 'text-emerald-400 font-bold' : 'text-gray-400 font-bold')}}">Caneca ${{it.bin}}</span>
            <span class="text-gray-500 font-mono">${{it.conf}}</span>
            <span class="text-gray-600">${{it.timeStr}}</span>
          </div>
        </div>
      `).join("");
    }}

    // Modales
    function openPhoneModal() {{
      const m = document.getElementById("phoneModal");
      m.classList.remove("hidden");
      m.classList.add("flex");
    }}
    function closePhoneModal() {{
      const m = document.getElementById("phoneModal");
      m.classList.add("hidden");
      m.classList.remove("flex");
    }}

    function openKeyModal() {{
      const m = document.getElementById("keyModal");
      m.classList.remove("hidden");
      m.classList.add("flex");
      document.getElementById("inputApiKey").value = storedApiKey;
    }}
    function closeKeyModal() {{
      const m = document.getElementById("keyModal");
      m.classList.add("hidden");
      m.classList.remove("flex");
    }}

    async function saveApiKey() {{
      const key = document.getElementById("inputApiKey").value.trim();
      if (!key) return;
      storedApiKey = key;
      localStorage.setItem("GEMINI_API_KEY", key);
      await saveKeyToServer(key);
      closeKeyModal();
      document.getElementById("keyStatusText").innerText = "Gemini: OK";
      document.getElementById("keyStatusText").className = "text-emerald-400 font-semibold";
      setEngine("gemini");
      alert("✅ API Key conectada exitosamente.");
    }}

    async function saveKeyToServer(key) {{
      try {{
        await fetch("/api/save_key", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ api_key: key }})
        }});
      }} catch (e) {{}}
    }}

    function toggleAudio() {{
      voiceEnabled = !voiceEnabled;
      document.getElementById("audioStatusText").innerText = voiceEnabled ? "Voz: ON" : "Voz: OFF";
      document.getElementById("audioIcon").className = voiceEnabled ? "w-4 h-4 text-emerald-400" : "w-4 h-4 text-gray-500";
    }}

    // Atajo barra espaciadora
    window.addEventListener("keydown", (e) => {{
      if (e.code === "Space" && e.target.tagName !== "INPUT") {{
        e.preventDefault();
        triggerScan(true);
      }} else if (e.key === "m" || e.key === "M") {{
        // Tecla 'M' para alternar espejo rápidamente
        toggleMirror();
      }}
    }});

    window.addEventListener("DOMContentLoaded", initApp);
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return HTMLResponse(content=DESKTOP_HTML)

@app.get("/remote", response_class=HTMLResponse)
async def serve_remote():
    return HTMLResponse(content=REMOTE_HTML)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n=======================================================")
    print(f"  🌱 EcoVision Multimodal AI 2.0")
    print(f"  💻 Interfaz PC:     http://localhost:{port}")
    print(f"  📱 Enlace Celular:  http://{LOCAL_IP}:{port}/remote")
    print(f"=======================================================\n")
    uvicorn.run("web_app:app", host="0.0.0.0", port=port, reload=False)
