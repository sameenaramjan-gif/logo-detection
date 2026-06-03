# =============================================================================
# config.py — Global Configuration, Paths & Hyperparameters
# =============================================================================
# This file is the single source of truth for every constant used across
# the Logo Detection System. Changing a value here automatically propagates
# to all other modules — no hunting through multiple files.
#
# Usage:
#   from config import *          # pull in everything
#   from config import IMG_SIZE   # pull in one item
# =============================================================================

import os

# ---------------------------------------------------------------------------
# ─── TERMINAL COLOUR PALETTE (ANSI escape codes) ───────────────────────────
# Used by every module to produce consistent, colour-coded console output.
# ---------------------------------------------------------------------------
class Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    # Foreground colours
    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

    # Bright foreground colours
    BR_RED     = "\033[91m"
    BR_GREEN   = "\033[92m"
    BR_YELLOW  = "\033[93m"
    BR_BLUE    = "\033[94m"
    BR_MAGENTA = "\033[95m"
    BR_CYAN    = "\033[96m"
    BR_WHITE   = "\033[97m"

    # Background colours (used for overlay boxes in webcam display)
    BG_BLACK   = "\033[40m"
    BG_RED     = "\033[41m"
    BG_GREEN   = "\033[42m"
    BG_YELLOW  = "\033[43m"
    BG_BLUE    = "\033[44m"


# ---------------------------------------------------------------------------
# ─── DIRECTORY STRUCTURE ───────────────────────────────────────────────────
# All paths are defined relative to THIS file's location so the project
# works regardless of where the user clones it.
# ---------------------------------------------------------------------------

# Root of the entire project (same folder as config.py)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Raw dataset root — sub-divided into train/ and val/ automatically
DATASET_DIR  = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_DIR    = os.path.join(DATASET_DIR,  "train")
VAL_DIR      = os.path.join(DATASET_DIR,  "val")

# Where the trained Keras model (.keras) will be saved
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH   = os.path.join(MODELS_DIR,   "logo_detector.keras")

# Optional: TensorBoard logs (nice to have for students exploring training)
LOGS_DIR     = os.path.join(PROJECT_ROOT, "logs")


# ---------------------------------------------------------------------------
# ─── TARGET CLASS DEFINITIONS ──────────────────────────────────────────────
# Add or remove class names here — everything else adapts automatically.
# Order matters: index 0 → CLASS_NAMES[0], etc.
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "brand_apple",      # Apple logo (bitten-apple silhouette)
    "brand_nike",       # Nike Swoosh logo
    "brand_nike_air",   # Nike Air text / logo variant
]

NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# ─── IMAGE DIMENSIONS ──────────────────────────────────────────────────────
# Standard MobileNet / VGG-compatible 224×224 RGB input.
# Changing these two constants resizes everything throughout the pipeline.
# ---------------------------------------------------------------------------
IMG_HEIGHT = 224
IMG_WIDTH  = 224
IMG_SIZE   = (IMG_HEIGHT, IMG_WIDTH)          # (H, W) tuple — for tf.keras
IMG_SHAPE  = (IMG_HEIGHT, IMG_WIDTH, 3)       # (H, W, C) — full tensor shape


# ---------------------------------------------------------------------------
# ─── DATA COLLECTION PARAMETERS ────────────────────────────────────────────
# ---------------------------------------------------------------------------
# How many images to capture per class during the collection session.
IMAGES_PER_CLASS = 200

# Fraction of captured images routed to validation (the rest go to train).
# E.g. 0.2 → 80 % train, 20 % val.
VAL_SPLIT = 0.2

# Webcam device index (0 = built-in webcam, 1 = first external, etc.)
WEBCAM_INDEX = 0

# Countdown in seconds shown on screen before capture begins.
CAPTURE_COUNTDOWN = 3

# Delay (seconds) between automatic frame captures to avoid blurry duplicates.
CAPTURE_DELAY = 0.15   # ~6–7 fps capture rate

# CLAHE (Contrast Limited Adaptive Histogram Equalization) clip-limit used
# when applying real-world lighting normalisation during capture.
CLAHE_CLIP_LIMIT    = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


# ---------------------------------------------------------------------------
# ─── TRAINING HYPERPARAMETERS ──────────────────────────────────────────────
# ---------------------------------------------------------------------------
BATCH_SIZE      = 32
EPOCHS          = 30           # Maximum epochs; EarlyStopping may stop earlier
LEARNING_RATE   = 1e-3         # Adam initial LR
LR_DECAY_FACTOR = 0.5          # ReduceLROnPlateau: multiply LR by this on plateau
LR_PATIENCE     = 4            # Epochs with no val_loss improvement before LR drop
EARLY_STOP_PAT  = 8            # Epochs before early stopping kicks in
MIN_LR          = 1e-6         # Floor for ReduceLROnPlateau

# Random seed — keeps dataset splits and weight initialisation reproducible.
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# ─── DATA AUGMENTATION PARAMETERS (applied only to training set) ───────────
# ---------------------------------------------------------------------------
AUG_ROTATION_RANGE      = 20      # degrees  ±
AUG_WIDTH_SHIFT_RANGE   = 0.15    # fraction of image width
AUG_HEIGHT_SHIFT_RANGE  = 0.15    # fraction of image height
AUG_SHEAR_RANGE         = 0.10    # shear angle in radians
AUG_ZOOM_RANGE          = 0.20    # [1-zoom, 1+zoom] zoom range
AUG_BRIGHTNESS_RANGE    = (0.70, 1.30)   # simulates different lighting
AUG_HORIZONTAL_FLIP     = True    # logos can appear mirrored in real life
AUG_FILL_MODE           = "nearest"


# ---------------------------------------------------------------------------
# ─── CNN ARCHITECTURE PARAMETERS ───────────────────────────────────────────
# ---------------------------------------------------------------------------
DROPOUT_RATE    = 0.50     # Dropout before the final dense layer
L2_REG          = 1e-4     # L2 kernel regularisation weight


# ---------------------------------------------------------------------------
# ─── INFERENCE / LIVE DISPLAY PARAMETERS ───────────────────────────────────
# ---------------------------------------------------------------------------
# Confidence thresholds that control the colour of the on-screen label box.
#   ≥ HIGH_CONF  → bright green  (very confident)
#   ≥ MED_CONF   → yellow        (moderate confidence)
#   <  MED_CONF  → red           (uncertain / unknown)
HIGH_CONF_THRESHOLD = 0.80
MED_CONF_THRESHOLD  = 0.50

# BGR colour tuples (OpenCV uses BGR, not RGB!)
COLOR_HIGH = (0,   230,  80)    # bright green
COLOR_MED  = (0,   210, 255)    # amber / yellow
COLOR_LOW  = (50,   50, 230)    # red-ish

# Region-of-interest (ROI) box drawn on the live webcam frame.
# The model crops & resizes this region for inference.
ROI_TOP_LEFT_RATIO     = (0.25, 0.15)   # (x_ratio, y_ratio) from frame top-left
ROI_BOTTOM_RIGHT_RATIO = (0.75, 0.85)   # (x_ratio, y_ratio)

# Font scale & thickness for OpenCV text overlays
OSD_FONT_SCALE      = 0.75
OSD_FONT_THICKNESS  = 2
OSD_PADDING         = 8     # pixels of padding inside label boxes

# Display window title
WINDOW_TITLE = "Logo Detection System — Live Inference  |  Press Q to quit"


# ---------------------------------------------------------------------------
# ─── PRETTY-PRINT HELPERS ──────────────────────────────────────────────────
# Utility strings used by every module for consistent console formatting.
# ---------------------------------------------------------------------------
DIVIDER_THICK = f"{Colors.CYAN}{'═' * 65}{Colors.RESET}"
DIVIDER_THIN  = f"{Colors.DIM}{'─' * 65}{Colors.RESET}"

def section_header(title: str) -> str:
    """Return a formatted section header string ready for print()."""
    pad = (63 - len(title)) // 2
    return (
        f"\n{DIVIDER_THICK}\n"
        f"{Colors.CYAN}║{Colors.RESET}"
        f"{' ' * pad}"
        f"{Colors.BOLD}{Colors.BR_WHITE}{title}{Colors.RESET}"
        f"{' ' * (63 - pad - len(title))}"
        f"{Colors.CYAN}║{Colors.RESET}\n"
        f"{DIVIDER_THICK}"
    )

def info(msg: str)    -> None: print(f"  {Colors.BR_CYAN}ℹ  {Colors.RESET}{msg}")
def success(msg: str) -> None: print(f"  {Colors.BR_GREEN}✔  {Colors.RESET}{msg}")
def warning(msg: str) -> None: print(f"  {Colors.BR_YELLOW}⚠  {Colors.RESET}{msg}")
def error(msg: str)   -> None: print(f"  {Colors.BR_RED}✖  {Colors.RESET}{msg}")
def step(msg: str)    -> None: print(f"  {Colors.MAGENTA}→  {Colors.RESET}{msg}")
