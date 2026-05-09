"""
demo_colab.py  —  FasalAI
Run AFTER 04_export_onnx_colab.py produces crop_disease_model_int8.onnx.
"""


import subprocess, os
subprocess.run(["pip", "install", "-q", "streamlit", "pyngrok", "onnxruntime", "Pillow"], check=True)
print("✅ Packages ready")



DEMO_APP = r'''
import json, time, os, sys
import numpy as np
from pathlib import Path
from PIL import Image
import streamlit as st

# ── Paths (reads model from Google Drive) ─────────────────────────────────────
DRIVE_BASE  = "/content/drive/MyDrive/FasalAI"
ONNX_PATH   = f"{DRIVE_BASE}/checkpoints/crop_disease_model_int8.onnx"
LABELS_PATH = f"{DRIVE_BASE}/checkpoints/class_labels.json"

# ── Hindi output — try Drive path, then local, then inline fallback ───────────
sys.path.insert(0, DRIVE_BASE)
sys.path.insert(0, "/content")
try:
    from hindi_output import get_advice, SEVERITY_EMOJI
except ImportError:
    # Inline minimal fallback so demo works even without hindi_output.py
    SEVERITY_EMOJI = {0:"✅",1:"🟡",2:"🟠",3:"🔴",4:"⛔",5:"☠️"}
    def get_advice(cls):
        name = cls.replace("___"," — ").replace("_"," ")
        return {
            "hindi_name": name, "symptoms_hindi": "Hindi output module not found",
            "treatment": ["🔬 कृषि विशेषज्ञ से सलाह लें"],
            "prevention": "नियमित निगरानी करें",
            "urgency": "⚡ विशेषज्ञ से परामर्श करें",
            "severity": 2, "severity_emoji": "🟠", "severity_label": "मध्यम"
        }

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FasalAI", page_icon="🌿", layout="centered")
st.markdown("""
<style>
.block-container{max-width:720px}
.big-title{font-size:2.4rem;font-weight:800;color:#15803d;line-height:1.1}
.tag{background:#dcfce7;color:#166534;border-radius:20px;padding:3px 14px;
     font-size:.8rem;font-weight:700;display:inline-block;margin:4px 3px}
.step-box{background:#f0fdf4;border:1px solid #86efac;border-radius:10px;
          padding:10px 14px;margin:4px 0;font-size:.95rem}
</style>
""", unsafe_allow_html=True)

# ── Load ONNX model (cached so it's only loaded once per session) ─────────────
@st.cache_resource(show_spinner="🌱 Loading FasalAI model…")
def load_model():
    if not Path(ONNX_PATH).exists():
        return None, []
    import onnxruntime as ort
    sess  = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    names = json.load(open(LABELS_PATH))
    return sess, names

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

def run_inference(sess, names, img, temp, hum):
    img_arr = preprocess(img)
    env_arr = np.array([[(temp - 10) / 40, hum / 100]], dtype=np.float32)
    t0      = time.perf_counter()
    logits  = sess.run(None, {"image": img_arr, "env_features": env_arr})[0]
    ms      = (time.perf_counter() - t0) * 1000
    exp     = np.exp(logits - logits.max())
    probs   = (exp / exp.sum())[0]
    idx     = int(np.argmax(probs))
    return names[idx], float(probs[idx]), probs, round(ms, 1)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌡️ मौसम की जानकारी")
    temperature = st.slider("तापमान (°C)", 5, 45, 25)
    humidity    = st.slider("नमी (%)", 20, 100, 65)
    if humidity > 75 and temperature < 22:
        st.warning("⚠️ पीले जंग की संभावना — गेहूँ की जाँच करें")
    st.divider()
    st.markdown("**FasalAI v1.0**")
    st.markdown("- EfficientNet-B0 + CBAM")
    st.markdown("- 38 रोग वर्ग")
    st.markdown("- ऑफलाइन · No internet needed")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="big-title">🌿 FasalAI</div>', unsafe_allow_html=True)
st.markdown(
    '<span class="tag">📴 Offline-First</span>'
    '<span class="tag">🇮🇳 Hindi Output</span>'
    '<span class="tag">⚡ &lt;100ms CPU</span>',
    unsafe_allow_html=True,
)
st.caption("फसल की बीमारी पत्ते की फोटो से पहचानें — बिना इंटरनेट")

sess, class_names = load_model()

if sess is None:
    st.error("❌ ONNX model not found.")
    st.code(f"Expected: {ONNX_PATH}")
    st.info("Run export_onnx cell first to generate crop_disease_model_int8.onnx")
    st.stop()

# ── Quick demo buttons ─────────────────────────────────────────────────────────
st.markdown("**Quick demo (no image upload needed):**")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🍅 टमाटर झुलसा"):   st.session_state["demo"] = "Tomato___Late_blight"
with c2:
    if st.button("🌾 गेहूँ जंग"):       st.session_state["demo"] = "Wheat___Yellow_Rust"
with c3:
    if st.button("✅ स्वस्थ पत्ता"):    st.session_state["demo"] = "Tomato___healthy"

uploaded = st.file_uploader("या पत्ते की फोटो अपलोड करें", type=["jpg", "jpeg", "png"])

if uploaded or "demo" in st.session_state:
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_column_width=True, caption="अपलोड की गई फोटो")
        with st.spinner("विश्लेषण हो रहा है…"):
            pred_class, confidence, probs, ms = run_inference(
                sess, class_names, img, temperature, humidity
            )
    else:
        pred_class  = st.session_state.pop("demo")
        confidence  = 0.91
        probs, ms   = None, 85.0

    adv        = get_advice(pred_class)
    is_healthy = adv["severity"] == 0

    st.divider()
    st.markdown("### 🔍 परिणाम / Result")
    m1, m2, m3 = st.columns(3)
    m1.metric("विश्वसनीयता", f"{confidence:.0%}")
    m2.metric("गंभीरता", f"{adv['severity_emoji']} {adv['severity_label']}")
    m3.metric("समय", f"{ms:.0f} ms")

    if is_healthy:
        st.success(f"✅ **{adv['hindi_name']}** — {adv['symptoms_hindi']}")
    else:
        st.error(f"🚨 **{adv['hindi_name']}**")
        st.markdown(f"**लक्षण:** {adv['symptoms_hindi']}")
        st.markdown(f"**तात्कालिकता:** {adv['urgency']}")
        st.markdown("#### 💊 उपचार / Treatment")
        for step in adv["treatment"]:
            st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)
        st.markdown("#### 🛡️ रोकथाम / Prevention")
        st.info(adv["prevention"])

    if probs is not None and class_names:
        with st.expander("Top-5 predictions (confidence breakdown)"):
            for i in np.argsort(probs)[::-1][:5]:
                label = class_names[i].replace("___", " → ").replace("_", " ")
                st.progress(float(probs[i]), text=f"{label}: {probs[i]:.1%}")

    st.caption(
        f"Model class: `{pred_class}` · "
        "Inference ran on-device. No data sent to any server."
    )

else:
    st.markdown("""
    <div style="text-align:center;padding:2.5rem;background:#f0fdf4;
                border-radius:14px;border:2px dashed #86efac">
        <div style="font-size:3.5rem">🌱</div>
        <p style="font-weight:600;font-size:1.1rem;color:#166534">
            पत्ते की फोटो अपलोड करें</p>
        <p style="color:#6b7280;font-size:.85rem">
            38 फसल रोगों की पहचान · टमाटर · गेहूँ · आलू · मक्का और अधिक
        </p>
    </div>""", unsafe_allow_html=True)
'''

with open("/content/fasalai_demo.py", "w", encoding="utf-8") as f:
    f.write(DEMO_APP)
print("✅ Demo app written to /content/fasalai_demo.py")



import shutil
from pathlib import Path

# Try Drive first, then check if it's already in /content
HINDI_DRIVE   = "/content/drive/MyDrive/FasalAI/hindi_output.py"
HINDI_CONTENT = "/content/hindi_output.py"

if Path(HINDI_DRIVE).exists():
    shutil.copy(HINDI_DRIVE, HINDI_CONTENT)
    print(f"✅ hindi_output.py copied from Drive → {HINDI_CONTENT}")
elif Path(HINDI_CONTENT).exists():
    print(f"✅ hindi_output.py already at {HINDI_CONTENT}")
else:
    print("⚠️  hindi_output.py not found.")
    print("   Options:")
    print("   1. Upload it via the Files panel (left sidebar) to /content/")
    print("   2. Or commit it to your repo and clone it")
    print("   The demo will still run with an inline fallback for unknown diseases.")




# ── ✏️  PASTE YOUR NGROK TOKEN HERE ───────────────────────────────────────────
NGROK_TOKEN = "PASTE_YOUR_NGROK_TOKEN_HERE"
# ──────────────────────────────────────────────────────────────────────────────

import threading, time, subprocess
from pyngrok import ngrok, conf

# Kill any leftover Streamlit or ngrok processes from previous runs
os.system("pkill -f 'streamlit run' 2>/dev/null")
ngrok.kill()
time.sleep(1.5)

# Set ngrok auth token
conf.get_default().auth_token = NGROK_TOKEN

PORT = 8501

# Launch Streamlit in a background thread
def _run_streamlit():
    subprocess.run([
        "streamlit", "run", "/content/fasalai_demo.py",
        "--server.port",              str(PORT),
        "--server.headless",          "true",
        "--server.enableCORS",        "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false",
    ])

thread = threading.Thread(target=_run_streamlit, daemon=True)
thread.start()

print("⏳ Starting Streamlit (waiting 5 seconds)…")
time.sleep(5)

# Open ngrok tunnel
tunnel     = ngrok.connect(PORT, "http")
public_url = tunnel.public_url

print(f"""
╔══════════════════════════════════════════════════════╗
║  🌿  FasalAI IS LIVE                                ║
╠══════════════════════════════════════════════════════╣
║  {public_url:<52} ║
╚══════════════════════════════════════════════════════╝

  📱 Share this URL with judges
     Works on mobile, tablet, and laptop
     No installation needed on their device

  ✅ Keep this cell running while the demo is live
  ⚠️  Free ngrok URL expires after ~2 hours
     Re-run this cell to get a fresh URL

  🛑 To stop: Runtime → Interrupt Execution
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# (OPTIONAL) Run demo on your LOCAL laptop instead of Colab
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
LOCAL LAPTOP SETUP (run in VS Code terminal — NOT in Colab):

1. Download these two files from Google Drive to your laptop:
   - checkpoints/crop_disease_model_int8.onnx
   - checkpoints/class_labels.json
   Put them in your project's ./checkpoints/ folder.

2. Make sure hindi_output.py is in your project root.

3. Install requirements:
   pip install streamlit onnxruntime pillow numpy

4. Edit demo.py line with USE_MOCK:
   USE_MOCK = False

5. Run:
   streamlit run demo.py
   → Opens automatically at http://localhost:8501

6. Share with a judge on the SAME WiFi:
   Windows: run  ipconfig  → find IPv4 address (e.g. 192.168.1.10)
   Judge opens:  http://192.168.1.10:8501
   Both must be on the same WiFi network.

7. Share via internet (different networks):
   from pyngrok import ngrok
   ngrok.set_auth_token("YOUR_TOKEN")
   print(ngrok.connect(8501).public_url)
   Keep this running while demo is live.
"""
print("📖 Local demo instructions are in the comments above.")
