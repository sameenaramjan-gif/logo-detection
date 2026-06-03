# 🔍 Logo Detection System

A production-grade, end-to-end logo detection pipeline built with **Python 3.12** and **TensorFlow/Keras**. Capture your own training data with a webcam, train a custom CNN, and run real-time inference — all from a single, elegant interactive CLI.

---

## ✨ Features

| Module | Description |
|--------|-------------|
| **Module 1 — Data Collector** | Guided webcam capture with auto-CLAHE lighting normalisation, automatic train/val split routing, and a live HUD overlay |
| **Module 2 — CNN Trainer** | Custom 4-block CNN with BatchNorm, L2 regularisation, data augmentation, EarlyStopping, ReduceLROnPlateau, and model checkpointing |
| **Module 3 — Inference Engine** | Real-time webcam inference with a colour-coded confidence HUD, probability bar chart, FPS counter, freeze-frame, and screenshot capture |

---

## 🗂 Project Structure

```
logo_detection/
├── main.py               ← Entry point — interactive CLI menu
├── config.py             ← All global constants, paths & hyperparameters
├── dataset_builder.py    ← Module 1: webcam data collection
├── model_trainer.py      ← Module 2: CNN architecture & training loop
├── webcam_inference.py   ← Module 3: real-time live inference
├── requirements.txt      ← Python dependencies
│
├── dataset/              ← Created automatically on first run
│   ├── train/
│   │   ├── brand_apple/
│   │   ├── brand_nike/
│   │   └── brand_nike_air/
│   └── val/
│       ├── brand_apple/
│       ├── brand_nike/
│       └── brand_nike_air/
│
├── models/               ← Created automatically after training
│   ├── logo_detector.keras
│   ├── best_checkpoint.keras
│   └── training_curves.png
│
├── logs/                 ← TensorBoard logs (optional)
└── screenshots/          ← Saved via [S] key during inference
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/logo-detection-system.git
cd logo-detection-system
```

### 2. Create a virtual environment (recommended)

```bash
# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** replace `tensorflow` with `tensorflow[and-cuda]` (Linux) or install the appropriate GPU build for your platform. See the [TF install guide](https://www.tensorflow.org/install/gpu).

### 4. Launch the application

```bash
python main.py
```

You will see the interactive menu:

```
╔═══════════════════════════════════════════════════════════════╗
║                         MAIN MENU                             ║
╚═══════════════════════════════════════════════════════════════╝

  [1]  Module 1 — Interactive Data Collector
       Capture images from your webcam

  [2]  Module 2 — Deep Learning CNN Trainer
       Train the CNN on your dataset

  [3]  Module 3 — Real-Time Inference Engine
       Run live logo detection

  [4]  Dataset Status
  [5]  System Information
  [Q]  Quit
```

---

## 📸 Workflow

### Step 1 — Collect Training Data (Module 1)

1. Select **[1]** from the main menu.
2. Choose a class (e.g., `brand_nike`) or capture all classes.
3. A webcam preview window opens with a live HUD.
4. Use these keys:

| Key | Action |
|-----|--------|
| `SPACE` | Capture a single frame |
| `A` | Toggle auto-capture mode |
| `P` | Pause / resume auto-capture |
| `Q` | Finish and close the window |

5. Aim for **≥ 200 images per class** for reasonable accuracy.

> **Tips:**
> - Vary the angle, distance, and background.
> - Try different lighting conditions (natural light, indoor lamp, shadows).
> - The system applies CLAHE normalisation automatically, but diversity still helps.

---

### Step 2 — Train the Model (Module 2)

1. Select **[2]** from the main menu.
2. The system validates your dataset, builds tf.data pipelines, and shows the model summary.
3. Press **Y** to start training.
4. Training runs for up to 30 epochs with EarlyStopping.
5. On completion the model is saved to `models/logo_detector.keras`.

**What you'll see during training:**

```
Epoch 1/30
  16/16 ━━━━━━━━━━━━━━━━━━━━ 12s  loss: 1.0821  accuracy: 0.3812
         val_loss: 0.9734  val_accuracy: 0.4400
Epoch 2/30 ...
```

A training-curve PNG is saved to `models/training_curves.png` after training.

---

### Step 3 — Live Inference (Module 3)

1. Select **[3]** from the main menu.
2. A webcam window opens with the full detection HUD.
3. Hold a logo inside the **bracket box** in the centre of the frame.

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit inference |
| `F` | Freeze / unfreeze frame |
| `S` | Save a screenshot to `screenshots/` |

**HUD Elements:**

- **Bracket box** — green (confident ≥ 80%), amber (50–80%), red (< 50%)
- **Label above box** — predicted class + confidence %
- **Probability bars** — bottom-left panel, all classes at once
- **FPS counter** — top-right corner

---

## ⚙️ Configuration

All tunable parameters live in **`config.py`**. Key ones:

```python
# Classes to detect — add your own!
CLASS_NAMES = ["brand_apple", "brand_nike", "brand_nike_air"]

# Capture
IMAGES_PER_CLASS = 200    # target images per class
VAL_SPLIT        = 0.20   # 20% goes to validation

# Training
BATCH_SIZE     = 32
EPOCHS         = 30
LEARNING_RATE  = 1e-3

# Inference thresholds
HIGH_CONF_THRESHOLD = 0.80   # green box
MED_CONF_THRESHOLD  = 0.50   # amber box
```

---

## 🏗 CNN Architecture

```
Input (224×224×3)
    ↓
Augmentation Block (flip, rotate, zoom, translate, brightness)
    ↓
Conv Block 1: Conv(32)×2 → BN → ReLU → MaxPool
Conv Block 2: Conv(64)×2 → BN → ReLU → MaxPool
Conv Block 3: Conv(128)×2 → BN → ReLU → MaxPool
Conv Block 4: Conv(256)×2 → BN → ReLU → GlobalAvgPool
    ↓
Dense(512) → BN → ReLU → Dropout(0.5)
    ↓
Dense(3, softmax) → Predicted class probabilities
```

---

## 📊 TensorBoard (Optional)

Training metrics are automatically logged to `logs/`. View them with:

```bash
tensorboard --logdir logs/
# Then open http://localhost:6006 in your browser
```

---

## 🛠 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cannot open webcam at index 0` | Try changing `WEBCAM_INDEX = 1` in `config.py`, or check that no other app is using the camera |
| TensorFlow not found | Run `pip install tensorflow` |
| Training is very slow | No GPU detected — this is normal for CPU. Reduce `EPOCHS` or `BATCH_SIZE` |
| Low accuracy | Collect more varied images (Module 1). Aim for 200+ per class |
| `Model not found` error in Module 3 | Run Module 2 first to train and save the model |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | ≥ 2.13 | CNN training & inference |
| `opencv-python` | ≥ 4.8 | Webcam capture & rendering |
| `numpy` | ≥ 1.24 | Numerical operations |
| `matplotlib` | ≥ 3.7 | Training curve plots (optional) |

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push and open a Pull Request.

---

## 📄 Licence

MIT — free to use, modify, and distribute.

---

*Built as a learning project — from raw webcam frames to a live inference HUD in five clean Python files.*
