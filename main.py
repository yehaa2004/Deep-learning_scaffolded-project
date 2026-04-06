# ============================================================
# main.py  —  FastAPI Backend for Marine Species Detection
# Run: python -m uvicorn main:app --reload --port 8000
# ============================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import librosa
import io
import os
import pickle
import base64
from PIL import Image
import tensorflow as tf

app = FastAPI(title="Marine Species Detection API", version="1.0.0")

# ── CORS ── Allow all origins (works with file://, localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve the HTML frontend directly from FastAPI ──
# Place your index_with_demo.html (or index.html) in the same folder as main.py
# Then open http://127.0.0.1:8000/demo  instead of opening the .html file directly.
# This avoids browser CORS blocks that occur when opening html via file:// protocol.
from fastapi.responses import HTMLResponse

@app.get("/demo", response_class=HTMLResponse)
def serve_demo():
    # Search in multiple locations robustly
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cwd       = os.getcwd()
    html_candidates = [
        os.path.join(base_dir, "index_with_demo.html"),
        os.path.join(base_dir, "index.html"),
        os.path.join(cwd,      "index_with_demo.html"),
        os.path.join(cwd,      "index.html"),
        "index_with_demo.html",
        "index.html",
    ]
    for p in html_candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    # Show helpful debug info
    files_here = os.listdir(cwd)
    return f"""<html><body style='font-family:monospace;background:#111;color:#eee;padding:30px;'>
    <h2 style='color:#f88;'>HTML file not found!</h2>
    <p>Place <b>index_with_demo.html</b> in the same folder as <b>main.py</b></p>
    <p>main.py location: <code>{base_dir}</code></p>
    <p>Current working dir: <code>{cwd}</code></p>
    <p>Files found here: <code>{files_here}</code></p>
    </body></html>"""


# ── CLASS NAMES (32 marine species) ──
CLASS_NAMES = [
    "Atlantic_Spotted_Dolphin", "Bearded_Seal", "Beluga", "Blue_Whale",
    "Bowhead_Whale", "Common_Dolphin", "Dugong", "Fin_Whale",
    "Gray_Seal", "Gray_Whale", "Harbor_Porpoise", "Harbor_Seal",
    "Harp_Seal", "Hooded_Seal", "Humpback_Whale", "Killer_Whale",
    "Leopard_Seal", "Minke_Whale", "Narwhal", "North_Atlantic_Right_Whale",
    "Northern_Elephant_Seal", "Pacific_White_Sided_Dolphin",
    "Pantropical_Spotted_Dolphin", "Pilot_Whale", "Ribbon_Seal",
    "Ringed_Seal", "Ross_Seal", "Southern_Elephant_Seal", "Sperm_Whale",
    "Spinner_Dolphin", "Spotted_Seal", "Weddell_Seal",
]

# ── MODEL PATHS ──
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

MODEL_PATHS = {
    "cnn":           os.path.join(MODEL_DIR, "cnn_model.keras"),
    "mlp":           os.path.join(MODEL_DIR, "mlp_model.keras"),
    "pretrained_cnn":os.path.join(MODEL_DIR, "pretrained_cnn_model.keras"),
    "rnn":           os.path.join(MODEL_DIR, "rnn_model.keras"),
    "lstm":          os.path.join(MODEL_DIR, "lstm_model.keras"),
    "gru":           os.path.join(MODEL_DIR, "gru_model.keras"),
    "autoencoder":   os.path.join(MODEL_DIR, "autoencoder_model.keras"),
    "gan_generator": os.path.join(MODEL_DIR, "gan_generator.keras"),
}

SCALER_PATH = os.path.join(MODEL_DIR, "audio_scaler.pkl")

# ── Lazy-load models ──
_models = {}

def get_model(name: str):
    if name not in _models:
        path = MODEL_PATHS.get(name)
        if not path or not os.path.exists(path):
            raise HTTPException(
                status_code=404,
                detail=f"Model '{name}' not found. Make sure {path} exists."
            )
        _models[name] = tf.keras.models.load_model(path)
    return _models[name]

def get_scaler():
    if not os.path.exists(SCALER_PATH):
        return None
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

# ── Helpers ──

def image_to_base64(arr: np.ndarray) -> str:
    """Convert numpy image array (H,W,3) float [0,1] or uint8 to base64 PNG string."""
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def preprocess_image(file_bytes: bytes, flatten: bool = False, target_size: int = 64) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size))
    img = img.astype(np.float32) / 255.0
    if flatten:
        img = img.flatten()
    return np.expand_dims(img, 0)

def preprocess_audio(file_bytes: bytes, n_mfcc: int = 40, max_frames: int = 128):
    audio, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, mono=True)

    # MFCC extraction (FIXED)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T  # (time_steps, n_mfcc)

    # Debug (optional)
    print("MFCC shape:", mfcc.shape)

    # Pad / truncate
    if mfcc.shape[0] < max_frames:
        pad = np.zeros((max_frames - mfcc.shape[0], n_mfcc), dtype=np.float32)
        mfcc = np.vstack([mfcc, pad])
    else:
        mfcc = mfcc[:max_frames, :]

    scaler = get_scaler()
    if scaler:
        mfcc = scaler.transform(mfcc)

    return np.expand_dims(mfcc.astype(np.float32), 0)
def build_top5(probs: np.ndarray):
    top_idx = np.argsort(probs)[::-1][:5]
    return [
        {"species": CLASS_NAMES[i].replace("_", " "), "confidence": round(float(probs[i]) * 100, 2)}
        for i in top_idx
    ]

# ═══════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════

@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/demo")

@app.get("/models")
def list_models():
    available = {k: os.path.exists(v) for k, v in MODEL_PATHS.items()}
    return {"models": available}

# ── IMAGE MODELS ──

@app.post("/predict/cnn")
async def predict_cnn(file: UploadFile = File(...)):
    data  = await file.read()
    img   = preprocess_image(data)
    model = get_model("cnn")
    probs = model.predict(img, verbose=0)[0]
    return {"model": "CNN", "modality": "image",
            "top1": CLASS_NAMES[np.argmax(probs)].replace("_"," "),
            "confidence": round(float(np.max(probs)) * 100, 2),
            "top5": build_top5(probs)}

@app.post("/predict/mlp")
async def predict_mlp(file: UploadFile = File(...)):
    data  = await file.read()
    img   = preprocess_image(data, flatten=True)
    model = get_model("mlp")
    probs = model.predict(img, verbose=0)[0]
    return {"model": "MLP", "modality": "image",
            "top1": CLASS_NAMES[np.argmax(probs)].replace("_"," "),
            "confidence": round(float(np.max(probs)) * 100, 2),
            "top5": build_top5(probs)}

@app.post("/predict/pretrained_cnn")
async def predict_pretrained(file: UploadFile = File(...)):
    data  = await file.read()
    img   = preprocess_image(data, target_size=96)
    model = get_model("pretrained_cnn")
    probs = model.predict(img, verbose=0)[0]
    return {"model": "Pretrained CNN", "modality": "image",
            "top1": CLASS_NAMES[np.argmax(probs)].replace("_"," "),
            "confidence": round(float(np.max(probs)) * 100, 2),
            "top5": build_top5(probs)}

# ── AUDIO MODELS ──

@app.post("/predict/rnn")
async def predict_rnn(file: UploadFile = File(...)):
    data  = await file.read()
    feat  = preprocess_audio(data)
    model = get_model("rnn")
    probs = model.predict(feat, verbose=0)[0]
    return {"model": "RNN", "modality": "audio",
            "top1": CLASS_NAMES[np.argmax(probs)].replace("_"," "),
            "confidence": round(float(np.max(probs)) * 100, 2),
            "top5": build_top5(probs)}

@app.post("/predict/lstm")
async def predict_lstm(file: UploadFile = File(...)):
    data  = await file.read()
    feat  = preprocess_audio(data)
    model = get_model("lstm")
    probs = model.predict(feat, verbose=0)[0]
    return {"model": "LSTM", "modality": "audio",
            "top1": CLASS_NAMES[np.argmax(probs)].replace("_"," "),
            "confidence": round(float(np.max(probs)) * 100, 2),
            "top5": build_top5(probs)}

@app.post("/predict/gru")
async def predict_gru(file: UploadFile = File(...)):
    data  = await file.read()
    feat  = preprocess_audio(data)
    model = get_model("gru")
    probs = model.predict(feat, verbose=0)[0]
    return {"model": "GRU", "modality": "audio",
            "top1": CLASS_NAMES[np.argmax(probs)].replace("_"," "),
            "confidence": round(float(np.max(probs)) * 100, 2),
            "top5": build_top5(probs)}

# ── GENERATIVE MODELS ──

@app.post("/predict/autoencoder")
async def predict_autoencoder(file: UploadFile = File(...)):
    from skimage.metrics import structural_similarity as ssim
    data     = await file.read()
    img      = preprocess_image(data)           # (1,64,64,3) float [0,1]
    model    = get_model("autoencoder")
    recon    = model.predict(img, verbose=0)    # (1,64,64,3)

    mse      = float(np.mean((img - recon) ** 2))
    psnr     = float(10 * np.log10(1.0 / mse)) if mse > 0 else 99.0
    ssim_val = float(ssim(img[0], recon[0], data_range=1.0, channel_axis=-1))

    # Convert both original and reconstructed to base64 for display
    original_b64 = image_to_base64(img[0])
    recon_b64    = image_to_base64(recon[0])

    return {
        "model": "Autoencoder", "modality": "image",
        "mse": round(mse, 6),
        "psnr_db": round(psnr, 2),
        "ssim": round(ssim_val, 4),
        "quality": "Excellent" if ssim_val > 0.95 else "Good",
        "original_image": original_b64,
        "reconstructed_image": recon_b64,
    }

@app.post("/predict/gan")
async def predict_gan(file: UploadFile = File(default=None), seed: int = 42):
    model = get_model("gan_generator")
    original_b64 = None
    z_source_label = f"Random z (seed={seed})"

    try:
        has_file = file is not None and file.filename not in (None, "", "undefined")
        if has_file:
            data = await file.read()
            if data and len(data) > 0:
                img_arr = preprocess_image(data)
                flat    = img_arr.reshape(1, -1)
                rng     = np.random.default_rng(seed)
                proj    = rng.standard_normal((12288, 128)).astype(np.float32)
                z       = flat @ proj
                z       = (z - z.mean()) / (z.std() + 1e-8)
                original_b64  = image_to_base64(img_arr[0])
                z_source_label = "Image-conditioned z"
            else:
                has_file = False
    except Exception:
        has_file = False

    if not has_file:
        np.random.seed(seed)
        z = np.random.normal(0, 1, (1, 128)).astype(np.float32)
        # Visualise the latent vector as a 64x64 heatmap so we always have two panels
        z_vis = z[0].reshape(1, -1)
        z_tile = np.tile(z_vis, (64, 1))            # (64, 128)
        # Normalise to [0,255] and repeat to make 64x64x3
        z_tile = ((z_tile - z_tile.min()) / (z_tile.max() - z_tile.min() + 1e-8) * 255).astype(np.uint8)
        z_tile = z_tile[:, :64]                     # crop to 64 cols
        z_rgb  = np.stack([z_tile] * 3, axis=-1)    # (64,64,3)
        original_b64 = image_to_base64(z_rgb)

    gen_img = model.predict(z, verbose=0)[0]        # (H,W,3) Tanh [-1,1]
    # Robust rescale: handle models that output [0,1] or [-1,1]
    g_min, g_max = gen_img.min(), gen_img.max()
    if g_min < -0.01:                               # Tanh output
        gen_img = ((gen_img + 1) / 2.0 * 255)
    else:                                           # Sigmoid output
        gen_img = (gen_img * 255)
    gen_img = np.clip(gen_img, 0, 255).astype(np.uint8)
    gen_b64 = image_to_base64(gen_img)

    return {
        "model": "GAN", "modality": "generated",
        "seed": seed,
        "z_source": z_source_label,
        "input_image":     original_b64,
        "generated_image": gen_b64,
        "shape": f"{gen_img.shape[1]}x{gen_img.shape[0]}x3"
    }
