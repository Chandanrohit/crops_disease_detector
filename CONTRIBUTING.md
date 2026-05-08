# Contributing to FasalAI 🌿

Thank you for contributing. This guide explains exactly how to add new disease classes, improve Hindi translations, retrain the model, and update the inference pipeline.

---

## Table of Contents

1. [Repo Structure](#repo-structure)
2. [Adding a New Disease Class](#1-adding-a-new-disease-class)
3. [Updating Hindi Output](#2-updating-hindi-output)
4. [Changing Training Hyperparameters](#3-changing-training-hyperparameters)
5. [Retraining the Model](#4-retraining-the-model)
6. [ONNX Export](#5-onnx-export)
7. [INT8 Quantization](#6-int8-quantization)
8. [Running the Demo](#7-running-the-demo)
9. [Running Inference CLI](#8-running-inference-cli)
10. [Evaluation & Metrics](#9-evaluation--metrics)
11. [Git Workflow](#10-git-workflow)

---

## Repo Structure

```
fasalai/
├── train.py                  ← Model definition + training loop
├── inference.py              ← Standalone inference (no Streamlit dependency)
├── export_onnx.py            ← Export trained .pt → .onnx
├── quantize.py               ← FP32 ONNX → INT8 ONNX
├── demo.py                   ← Streamlit web demo
├── utils/
│   └── hindi_output.py       ← Hindi disease advice database
├── data/
│   └── PlantVillage/         ← Dataset root (not committed — see .gitignore)
├── checkpoints/
│   ├── efficientnet_crop.pt          ← Best PyTorch checkpoint (not committed)
│   ├── crop_disease_model.onnx       ← FP32 ONNX (committed if <50MB)
│   ├── crop_disease_model_int8.onnx  ← INT8 ONNX (committed)
│   └── class_labels.json             ← Ordered class names (committed)
├── notebooks/
│   └── FasalAI_Colab.ipynb   ← Full guided training notebook
├── assets/                   ← Training graphs, confusion matrix
└── docs/                     ← Presentation and PDF
```

---

## 1. Adding a New Disease Class

### Step 1 — Add images to the dataset

Create a new folder under `data/PlantVillage/` with the exact class name format:

```
data/PlantVillage/
    Crop___Disease_Name/
        img001.jpg
        img002.jpg
        ...   (minimum 200 images recommended)
```

Use the naming convention: `CropName___Disease_Name` (three underscores between crop and disease).

### Step 2 — Add Hindi advice to `utils/hindi_output.py`

Open `utils/hindi_output.py` and add a new entry to the `DISEASE_DATA` dictionary:

```python
"Crop___Disease_Name": {
    "hindi_name":     "हिंदी में रोग का नाम",
    "symptoms_hindi": "लक्षण: पत्तियों पर ... (visible symptoms in Hindi)",
    "treatment": [
        "🔬 Step 1 treatment instruction in Hindi",
        "✂️ Step 2 treatment instruction",
        "🔄 Step 3: follow-up action",
    ],
    "prevention": "Prevention advice in Hindi — one to two sentences.",
    "urgency":    "⚡ जल्दी उपचार करें — 7 दिनों के अंदर",
    # urgency options: "🚨 तुरंत", "⚡ जल्दी", "⏰ मध्यम", "कोई तात्कालिकता नहीं"
    "severity":   2,
    # severity scale: 0=healthy, 1=mild, 2=moderate, 3=serious, 4=critical, 5=devastating
},
```

### Step 3 — Retrain the model

After adding images and Hindi advice, retrain from scratch or fine-tune:

```bash
# From scratch
python train.py

# Fine-tune from existing checkpoint (faster)
# Edit train.py: set CKPT_PATH to existing checkpoint, reduce LR to 1e-4, reduce EPOCHS to 10
python train.py
```

### Step 4 — Re-export and re-quantize

```bash
python export_onnx.py   # → checkpoints/crop_disease_model.onnx
python quantize.py      # → checkpoints/crop_disease_model_int8.onnx
```

### Step 5 — Verify

```bash
python inference.py --image test_leaf.jpg
# Confirm new class appears in top-5 predictions
```

---

## 2. Updating Hindi Output

All Hindi content lives in `utils/hindi_output.py` in the `DISEASE_DATA` dict.

### Severity scale reference

| Value | Label | Emoji | Meaning |
|---|---|---|---|
| 0 | स्वस्थ | ✅ | Healthy — no action |
| 1 | सावधान | 🟡 | Mild — monitor closely |
| 2 | मध्यम | 🟠 | Moderate — treat within 7 days |
| 3 | गंभीर | 🔴 | Serious — treat within 3 days |
| 4 | अत्यंत गंभीर | ⛔ | Critical — treat within 24–48 hours |
| 5 | विनाशकारी | ☠️ | Devastating — remove and burn plants |

### Urgency phrasing guide

```python
"urgency": "🚨 तुरंत उपचार करें — 24 घंटे के अंदर"   # Critical
"urgency": "⚡ जल्दी उपचार करें — 3 दिनों के अंदर"   # Serious
"urgency": "⚡ जल्दी उपचार करें — 7 दिनों के अंदर"   # Moderate
"urgency": "⏰ मध्यम — 10 दिनों के अंदर उपचार करें"  # Mild
"urgency": "कोई तात्कालिकता नहीं"                     # Healthy
```

### Test after editing

```bash
python -c "
from utils.hindi_output import get_advice
result = get_advice('Tomato___Late_blight')
print(result['hindi_name'])
print(result['treatment'])
"
```

---

## 3. Changing Training Hyperparameters

All hyperparameters are defined at the top of `train.py` under the `Configuration` section:

```python
# ── Core Hyperparameters — change only these ───────────────────────────────────
EPOCHS      = 20       # Increase to 50 if val_acc still rising at epoch 20
BATCH_SIZE  = 32       # Reduce to 16 if GPU OOM error; increase to 64 for speed
LR          = 1e-3     # Reduce to 1e-4 if loss oscillates; raise if learning too slow
IMG_SIZE    = 224      # Do not change — EfficientNet-B0 is optimized for 224×224
VAL_SPLIT   = 0.20     # Fraction of data used for validation (0.15–0.25 recommended)
WEIGHT_DECAY = 1e-4    # L2 regularization — increase to 1e-3 if overfitting
LABEL_SMOOTH = 0.1     # Label smoothing — 0.0 to disable, 0.1–0.2 typical range

# ── Fine-tune depth ────────────────────────────────────────────────────────────
# EfficientNet-B0 has blocks features.0 through features.8
# Adding more blocks = higher accuracy but slower training
TRAINABLE_LAYERS = ["features.5", "features.6", "features.7", "features.8", "classifier"]
# To train more of the network: add "features.4"
# To train less (faster, less overfit risk): remove "features.5"

# ── Dropout ───────────────────────────────────────────────────────────────────
DROPOUT_HEAD = 0.4     # In classifier head. Increase to 0.5 if overfitting.
```

### Diagnosing training problems from loss curves

| Curve pattern | Problem | Fix |
|---|---|---|
| Train↓ Val↓ together | ✅ Healthy | Continue |
| Train↓ Val plateaus | Underfitting | Increase epochs, LR, or trainable layers |
| Train↓ Val↑ (diverge) | Overfitting | Increase dropout, reduce epochs, add augmentation |
| Both oscillate/spike | LR too high | Multiply LR by 0.1 |
| Both barely move | LR too low | Multiply LR by 10 |
| Loss → NaN | Exploding gradients | Already handled by `clip_grad_norm_(max_norm=1.0)` |

---

## 4. Retraining the Model

### Local (GPU required)

```bash
# Ensure dataset is at ./data/PlantVillage/
python train.py
# Output: checkpoints/efficientnet_crop.pt + checkpoints/class_labels.json
```

### Google Colab (recommended)

Open `notebooks/FasalAI_Colab.ipynb` in Colab:
- Runtime → Change runtime type → GPU (T4)
- Run cells in order: Setup → Dataset → Train → Evaluate → Export
- All checkpoints auto-save to Google Drive

### Resuming after a crash

The training loop saves `checkpoints/efficientnet_crop_last.pt` after every epoch. If training is interrupted, re-run `train.py` — it auto-detects and resumes from the last checkpoint.

---

## 5. ONNX Export

Run after training completes:

```bash
python export_onnx.py
```

This reads `checkpoints/efficientnet_crop.pt` and writes `checkpoints/crop_disease_model.onnx`.

**To change the input names or opset version**, edit `export_onnx.py`:

```python
# Line ~45: change opset if needed (17 is current default, minimum 11)
opset_version = 17

# Line ~38–42: change input names if your inference.py uses different names
input_names  = ["image", "env_features"]
output_names = ["logits"]
```

---

## 6. INT8 Quantization

Run after ONNX export:

```bash
python quantize.py
```

This reads `checkpoints/crop_disease_model.onnx` and writes `checkpoints/crop_disease_model_int8.onnx`.

**To switch to static quantization** (higher accuracy, requires calibration data), edit `quantize.py`:

```python
# Current: dynamic quantization (no calibration data needed)
quantize_dynamic(model_input, model_output, weight_type=QuantType.QInt8)

# Alternative: static quantization (better accuracy, needs calibration)
# See onnxruntime quantization docs for static calibration setup
```

---

## 7. Running the Demo

### Mock mode (no model needed — for UI development)

```bash
# demo.py line 28: USE_MOCK = True   ← default
streamlit run demo.py
```

### Real model mode

```bash
# 1. Ensure checkpoints/crop_disease_model_int8.onnx exists
# 2. Edit demo.py line 28: USE_MOCK = False
streamlit run demo.py
# Opens http://localhost:8501
```

### Share with judges via ngrok

```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")   # free at ngrok.com
tunnel = ngrok.connect(8501)
print(tunnel.public_url)             # share this URL
```

---

## 8. Running Inference CLI

```bash
# Single image
python inference.py --image path/to/leaf.jpg

# With environment data
python inference.py --image leaf.jpg --temperature 22 --humidity 78

# Show top-5 predictions
python inference.py --image leaf.jpg --top-k 5

# Use INT8 model (faster)
python inference.py --image leaf.jpg
# To switch model path, edit ONNX_PATH at top of inference.py:
# ONNX_PATH = "./checkpoints/crop_disease_model_int8.onnx"
```

---

## 9. Evaluation & Metrics

After training, run full evaluation:

```bash
python evaluate.py
# Generates: confusion matrix, per-class F1, classification report
```

Key metrics to track:

| Metric | Target | Why |
|---|---|---|
| Top-1 Accuracy | >90% | Overall correctness |
| Macro F1 | >0.85 | Balanced across all 38 classes |
| Recall | >0.85 | Missing a disease is worse than a false alarm |
| Inference ms | <100ms | Real-time usability |
| INT8 model size | <10MB | Fits on farmer's phone |

---

## 10. Git Workflow

```bash
# Feature branch
git checkout -b feat/your-feature-name

# After changes
git add -A
git commit -m "feat: description of change"
git push origin feat/your-feature-name

# Open PR → main branch
# PR description must include: what changed, test results, metrics impact
```

### Commit message format

```
feat: add Rice___Brown_Spot disease class with Hindi output
fix: correct Wheat___Yellow_Rust urgency level (3→4)
refactor: extract preprocess() into utils/transforms.py
docs: update CONTRIBUTING.md with sensor integration steps
model: retrain on 40-class dataset — val_acc 92.3% → 93.1%
```

---

## Questions?

Open an issue or reach out to the team. Every contribution — however small — directly helps Indian farmers.
