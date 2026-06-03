#!/usr/bin/env python3
# =============================================================================
# main.py — Central Execution Engine & Interactive CLI Menu
# =============================================================================
# Logo Detection System
# ─────────────────────
# A production-grade, end-to-end logo detection pipeline built with
# Python 3.12 and TensorFlow/Keras.
#
# Modules
# ───────
#   1  Interactive Data Collector   — Capture labelled images via webcam
#   2  Deep Learning CNN Trainer    — Train a custom CNN on your dataset
#   3  Real-Time Inference Engine   — Live detection with a polished HUD
#   4  Dataset Status               — Quick inventory of collected images
#   5  System Information           — Show environment & config details
#   Q  Quit
#
# Usage
# ─────
#   python main.py
#
# Requirements
# ────────────
#   pip install tensorflow opencv-python numpy matplotlib
#   Python 3.12+  |  TensorFlow 2.x
# =============================================================================

import os
import sys
import platform
import textwrap
import time

# ---------------------------------------------------------------------------
# Ensure we can import our own modules even if the user runs main.py from a
# different working directory (e.g. python logo_detection/main.py).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import (
    CLASS_NAMES, MODEL_PATH, TRAIN_DIR, VAL_DIR,
    IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    WEBCAM_INDEX, IMAGES_PER_CLASS, VAL_SPLIT,
    PROJECT_ROOT,
    Colors, section_header, info, success, warning, error, step,
    DIVIDER_THICK, DIVIDER_THIN,
)

# Lazy imports — pulled in only when the user selects a module, so startup
# is near-instant even if TensorFlow takes a moment to import.
_dataset_builder   = None
_model_trainer     = None
_webcam_inference  = None


def _lazy_import_dataset_builder():
    global _dataset_builder
    if _dataset_builder is None:
        import dataset_builder as db
        _dataset_builder = db
    return _dataset_builder


def _lazy_import_model_trainer():
    global _model_trainer
    if _model_trainer is None:
        import model_trainer as mt
        _model_trainer = mt
    return _model_trainer


def _lazy_import_webcam_inference():
    global _webcam_inference
    if _webcam_inference is None:
        import webcam_inference as wi
        _webcam_inference = wi
    return _webcam_inference


# =============================================================================
# ─── SPLASH SCREEN ────────────────────────────────────────────────────────────
# =============================================================================

LOGO_ART = r"""
  ██╗      ██████╗  ██████╗  ██████╗     ██████╗ ███████╗████████╗
  ██║     ██╔═══██╗██╔════╝ ██╔═══██╗    ██╔══██╗██╔════╝╚══██╔══╝
  ██║     ██║   ██║██║  ███╗██║   ██║    ██║  ██║█████╗     ██║
  ██║     ██║   ██║██║   ██║██║   ██║    ██║  ██║██╔══╝     ██║
  ███████╗╚██████╔╝╚██████╔╝╚██████╔╝    ██████╔╝███████╗   ██║
  ╚══════╝ ╚═════╝  ╚═════╝  ╚═════╝     ╚═════╝ ╚══════╝   ╚═╝
"""


def print_splash() -> None:
    """
    Print the ASCII art banner, version, and a quick-start summary.
    Clears the terminal first for a clean presentation.
    """
    # Clear terminal (works on Linux / macOS / Windows)
    os.system("cls" if os.name == "nt" else "clear")

    print(f"{Colors.BR_CYAN}{LOGO_ART}{Colors.RESET}")
    print(f"{Colors.BOLD}  ─── Logo Detection System  ·  v1.0.0  ·  TensorFlow/Keras ───{Colors.RESET}")
    print(f"  {Colors.DIM}Real-time logo recognition powered by a custom CNN{Colors.RESET}\n")
    print(DIVIDER_THICK)

    # Quick dataset status
    def _count(split_dir: str) -> int:
        total = 0
        for cn in CLASS_NAMES:
            d = os.path.join(split_dir, cn)
            if os.path.isdir(d):
                total += sum(
                    1 for f in os.listdir(d)
                    if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
                )
        return total

    n_train = _count(TRAIN_DIR)
    n_val   = _count(VAL_DIR)
    model_exists = os.path.exists(MODEL_PATH)

    status_color = Colors.BR_GREEN if model_exists else Colors.BR_YELLOW
    model_status = "Ready  ✔" if model_exists else "Not trained yet"

    print(f"\n  {Colors.BOLD}System Status{Colors.RESET}")
    print(f"  {'Classes':<22} {', '.join(CLASS_NAMES)}")
    print(f"  {'Train images':<22} {n_train}")
    print(f"  {'Val images':<22} {n_val}")
    print(f"  {'Trained model':<22} {status_color}{model_status}{Colors.RESET}")
    print()


# =============================================================================
# ─── MAIN MENU ────────────────────────────────────────────────────────────────
# =============================================================================

MENU_OPTIONS = {
    "1": ("Module 1 — Interactive Data Collector",   "Capture images from your webcam"),
    "2": ("Module 2 — Deep Learning CNN Trainer",    "Train the CNN on your dataset"),
    "3": ("Module 3 — Real-Time Inference Engine",   "Run live logo detection"),
    "4": ("Dataset Status",                          "Show detailed image counts"),
    "5": ("System Information",                      "Show environment & config details"),
    "q": ("Quit",                                    "Exit the application"),
}


def print_main_menu() -> None:
    """Render the interactive main menu."""
    print(f"\n{DIVIDER_THICK}")
    print(
        f"{Colors.CYAN}║{Colors.RESET}"
        f"{'  MAIN MENU':^63}"
        f"{Colors.CYAN}║{Colors.RESET}"
    )
    print(DIVIDER_THICK)
    print()

    for key, (title, subtitle) in MENU_OPTIONS.items():
        if key == "q":
            print(DIVIDER_THIN)
        key_str = f"{Colors.BR_YELLOW}[{key.upper()}]{Colors.RESET}"
        print(f"  {key_str}  {Colors.BOLD}{title}{Colors.RESET}")
        print(f"       {Colors.DIM}{subtitle}{Colors.RESET}")
        print()


def get_menu_choice() -> str:
    """Prompt the user for a menu selection and return the normalised key."""
    while True:
        try:
            raw = input(f"  {Colors.BOLD}Enter choice: {Colors.RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "q"

        if raw in MENU_OPTIONS:
            return raw

        warning(f"'{raw}' is not a valid option. Please choose from: "
                f"{', '.join(MENU_OPTIONS.keys())}")


# =============================================================================
# ─── DATASET STATUS SCREEN ────────────────────────────────────────────────────
# =============================================================================

def show_dataset_status() -> None:
    """
    Print a detailed per-class, per-split image count table,
    plus a summary of what to do if counts are too low.
    """
    print(section_header("Dataset Status"))

    print(f"\n  {'Class':<22} {'Train':>8} {'Val':>8} {'Total':>8}  {'Progress':>20}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}  {'-'*20}")

    all_sufficient = True
    for cn in CLASS_NAMES:
        train_dir = os.path.join(TRAIN_DIR, cn)
        val_dir   = os.path.join(VAL_DIR,   cn)

        def _c(d):
            if not os.path.isdir(d):
                return 0
            return sum(
                1 for f in os.listdir(d)
                if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
            )

        n_train = _c(train_dir)
        n_val   = _c(val_dir)
        n_total = n_train + n_val

        # Visual progress bar (out of IMAGES_PER_CLASS target)
        target       = IMAGES_PER_CLASS
        filled       = int(min(n_total / target, 1.0) * 18)
        bar          = "█" * filled + "░" * (18 - filled)
        pct          = min(int(n_total / target * 100), 100)

        color = Colors.BR_GREEN if n_total >= 10 else Colors.BR_RED
        if n_total < 10:
            all_sufficient = False

        print(
            f"  {color}{cn:<22}{Colors.RESET}"
            f"  {n_train:>6}  {n_val:>8}  {n_total:>8}"
            f"  {bar} {pct:>3}%"
        )

    print()
    if all_sufficient:
        success("All classes have sufficient images.")
        if os.path.exists(MODEL_PATH):
            success(f"Trained model exists → {os.path.relpath(MODEL_PATH)}")
        else:
            info("No trained model found yet. Run Module 2 to train.")
    else:
        warning("Some classes need more images. Use Module 1 to collect more.")

    input(f"\n  {Colors.DIM}Press ENTER to return to the menu…{Colors.RESET}")


# =============================================================================
# ─── SYSTEM INFORMATION SCREEN ────────────────────────────────────────────────
# =============================================================================

def show_system_info() -> None:
    """
    Print environment details: Python version, TensorFlow version, GPU info,
    and the active configuration from config.py.
    """
    print(section_header("System Information"))

    # ── Python & Platform ─────────────────────────────────────────────────────
    print(f"\n  {Colors.BOLD}Environment{Colors.RESET}")
    print(f"  {'Python version':<28} {sys.version.split()[0]}")
    print(f"  {'Platform':<28} {platform.system()} {platform.release()}")
    print(f"  {'Architecture':<28} {platform.machine()}")

    # ── TensorFlow & GPU ──────────────────────────────────────────────────────
    print(f"\n  {Colors.BOLD}TensorFlow{Colors.RESET}")
    try:
        import tensorflow as tf
        print(f"  {'TF version':<28} {tf.__version__}")

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                print(f"  {'GPU':<28} {Colors.BR_GREEN}{gpu.name}{Colors.RESET}")
        else:
            print(f"  {'GPU':<28} {Colors.BR_YELLOW}None detected (CPU mode){Colors.RESET}")
            info("Training will work on CPU but will be slower.")
            info("For GPU acceleration: https://www.tensorflow.org/install/gpu")

    except ImportError:
        error("TensorFlow is not installed!")
        info("Run:  pip install tensorflow")

    # ── OpenCV ────────────────────────────────────────────────────────────────
    print(f"\n  {Colors.BOLD}OpenCV{Colors.RESET}")
    try:
        import cv2
        print(f"  {'OpenCV version':<28} {cv2.__version__}")
    except ImportError:
        error("OpenCV is not installed!")
        info("Run:  pip install opencv-python")

    # ── Active Configuration ───────────────────────────────────────────────────
    print(f"\n  {Colors.BOLD}Active Configuration (config.py){Colors.RESET}")
    config_items = [
        ("Project root",      PROJECT_ROOT),
        ("Image size",        f"{IMG_SIZE[0]}×{IMG_SIZE[1]} px"),
        ("Batch size",        BATCH_SIZE),
        ("Max epochs",        EPOCHS),
        ("Learning rate",     LEARNING_RATE),
        ("Train directory",   os.path.relpath(TRAIN_DIR)),
        ("Val directory",     os.path.relpath(VAL_DIR)),
        ("Model path",        os.path.relpath(MODEL_PATH)),
        ("Webcam index",      WEBCAM_INDEX),
        ("Images per class",  IMAGES_PER_CLASS),
        ("Val split",         f"{int(VAL_SPLIT * 100)} %"),
        ("Classes",           ", ".join(CLASS_NAMES)),
    ]
    for label, value in config_items:
        print(f"  {label:<28} {Colors.DIM}{value}{Colors.RESET}")

    input(f"\n  {Colors.DIM}Press ENTER to return to the menu…{Colors.RESET}")


# =============================================================================
# ─── DEPENDENCY CHECKER ───────────────────────────────────────────────────────
# =============================================================================

def check_dependencies() -> bool:
    """
    Verify that all required packages are importable.
    Prints a clear error and install command for any missing package.

    Returns:
        True if all dependencies are present; False otherwise.
    """
    required = {
        "tensorflow": "pip install tensorflow",
        "cv2":        "pip install opencv-python",
        "numpy":      "pip install numpy",
    }
    optional = {
        "matplotlib": "pip install matplotlib",
    }

    all_ok = True

    for pkg, install_cmd in required.items():
        try:
            __import__(pkg)
        except ImportError:
            error(f"Required package '{pkg}' is not installed.")
            info(f"  Fix: {install_cmd}")
            all_ok = False

    for pkg, install_cmd in optional.items():
        try:
            __import__(pkg)
        except ImportError:
            warning(f"Optional package '{pkg}' is not installed — training curves will be skipped.")
            info(f"  Install (optional): {install_cmd}")

    return all_ok


# =============================================================================
# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
# =============================================================================

def main() -> None:
    """
    Application entry point.

    1. Check dependencies.
    2. Print the splash screen.
    3. Run the interactive menu loop until the user quits.
    """
    # Dependency check before anything else
    if not check_dependencies():
        print(f"\n{Colors.BR_RED}Cannot start: missing required packages (see above).{Colors.RESET}")
        sys.exit(1)

    while True:
        print_splash()
        print_main_menu()

        choice = get_menu_choice()

        # ── Route to the selected module ──────────────────────────────────────
        if choice == "1":
            # Lazy-import and run Module 1
            try:
                db = _lazy_import_dataset_builder()
                db.run_data_collection()
            except Exception as exc:
                error(f"Data collection error: {exc}")
            finally:
                _pause_before_menu()

        elif choice == "2":
            # Lazy-import and run Module 2
            try:
                mt = _lazy_import_model_trainer()
                mt.run_training()
            except Exception as exc:
                error(f"Training error: {exc}")
            finally:
                _pause_before_menu()

        elif choice == "3":
            # Lazy-import and run Module 3
            if not os.path.exists(MODEL_PATH):
                warning("No trained model found.")
                info(f"Expected at: {os.path.relpath(MODEL_PATH)}")
                info("Please train the model first (option 2).")
                _pause_before_menu()
                continue
            try:
                wi = _lazy_import_webcam_inference()
                wi.run_inference()
            except Exception as exc:
                error(f"Inference error: {exc}")
            finally:
                _pause_before_menu()

        elif choice == "4":
            show_dataset_status()

        elif choice == "5":
            show_system_info()

        elif choice == "q":
            _farewell()
            sys.exit(0)


def _pause_before_menu() -> None:
    """Wait for the user to press ENTER before re-drawing the main menu."""
    try:
        input(f"\n  {Colors.DIM}Press ENTER to return to the main menu…{Colors.RESET}")
    except (EOFError, KeyboardInterrupt):
        pass


def _farewell() -> None:
    """Print a friendly goodbye message."""
    print(f"\n{DIVIDER_THICK}")
    msg = "Thank you for using Logo Detection System!"
    pad = (63 - len(msg)) // 2
    print(
        f"{Colors.CYAN}║{Colors.RESET}"
        f"{' ' * pad}"
        f"{Colors.BOLD}{Colors.BR_WHITE}{msg}{Colors.RESET}"
        f"{' ' * (63 - pad - len(msg))}"
        f"{Colors.CYAN}║{Colors.RESET}"
    )
    print(DIVIDER_THICK)
    print(f"\n  {Colors.DIM}Goodbye.{Colors.RESET}\n")


# =============================================================================
if __name__ == "__main__":
    # Handle Ctrl+C gracefully at the top level
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n  {Colors.BR_YELLOW}Interrupted by user (Ctrl+C). Exiting.{Colors.RESET}\n")
        sys.exit(0)
