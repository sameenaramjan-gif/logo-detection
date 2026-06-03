# =============================================================================
# dataset_builder.py — Interactive Data Collector (Module 1)
# =============================================================================
# Responsibilities:
#   1. Guarantee that all required dataset directories exist before use.
#   2. Walk the user through a guided, class-by-class capture session using
#      the system webcam.
#   3. Apply real-time CLAHE lighting normalisation to each captured frame.
#   4. Automatically route captured images into train/ or val/ splits.
#   5. Provide clear, colourful console feedback throughout.
#
# Dependencies:
#   pip install opencv-python numpy
# =============================================================================

import os
import cv2
import time
import random
import numpy as np

from config import (
    CLASS_NAMES, TRAIN_DIR, VAL_DIR, IMG_SIZE,
    IMAGES_PER_CLASS, VAL_SPLIT, WEBCAM_INDEX,
    CAPTURE_COUNTDOWN, CAPTURE_DELAY,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE,
    Colors, section_header, info, success, warning, error, step,
    DIVIDER_THIN, DIVIDER_THICK,
)


# =============================================================================
# ─── DIRECTORY MANAGEMENT ────────────────────────────────────────────────────
# =============================================================================

def ensure_dataset_directories() -> None:
    """
    Create every required dataset folder if it does not already exist.

    Tree created:
        dataset/
        ├── train/
        │   ├── brand_apple/
        │   ├── brand_nike/
        │   └── brand_nike_air/
        └── val/
            ├── brand_apple/
            ├── brand_nike/
            └── brand_nike_air/

    This function is idempotent — safe to call multiple times.
    """
    print(section_header("Directory Initialisation"))
    created_any = False

    for split_dir in [TRAIN_DIR, VAL_DIR]:
        for class_name in CLASS_NAMES:
            target = os.path.join(split_dir, class_name)
            if not os.path.exists(target):
                os.makedirs(target, exist_ok=True)
                step(f"Created  →  {os.path.relpath(target)}")
                created_any = True

    if not created_any:
        success("All dataset directories already exist — nothing to create.")
    else:
        success("Directory tree is ready.")

    # Print a quick inventory of existing images
    _print_dataset_inventory()


def _print_dataset_inventory() -> None:
    """Print a per-class, per-split image count table to the console."""
    print(f"\n{DIVIDER_THIN}")
    print(f"  {'Class':<22} {'Train':>8} {'Val':>8} {'Total':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")

    for class_name in CLASS_NAMES:
        train_path = os.path.join(TRAIN_DIR, class_name)
        val_path   = os.path.join(VAL_DIR,   class_name)

        n_train = _count_images(train_path)
        n_val   = _count_images(val_path)
        n_total = n_train + n_val

        color = Colors.BR_GREEN if n_total > 0 else Colors.DIM
        print(
            f"  {color}{class_name:<22}{Colors.RESET}"
            f"  {n_train:>6}  {n_val:>8}  {n_total:>8}"
        )

    print(DIVIDER_THIN)


def _count_images(directory: str) -> int:
    """Return the number of .jpg / .jpeg / .png files in *directory*."""
    if not os.path.isdir(directory):
        return 0
    valid_exts = {".jpg", ".jpeg", ".png"}
    return sum(
        1 for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in valid_exts
    )


def _next_image_index(directory: str) -> int:
    """
    Return the next available numeric index for naming a captured image.
    Ensures we never overwrite existing files when resuming a session.
    """
    return _count_images(directory)


# =============================================================================
# ─── LIGHTING NORMALISATION ──────────────────────────────────────────────────
# =============================================================================

def apply_clahe(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to the
    luminance channel of a BGR frame, then convert back to BGR.

    CLAHE enhances local contrast without amplifying noise, making the
    captured images more consistent across different lighting conditions
    (offices, homes, outdoor bright light, etc.).

    Args:
        frame_bgr: Raw BGR frame from OpenCV VideoCapture.

    Returns:
        CLAHE-enhanced BGR frame (same shape and dtype as input).
    """
    # Convert BGR → LAB colour space (L = luminance, A/B = colour axes)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE only to the L (luminance) channel
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    l_enhanced = clahe.apply(l_channel)

    # Merge back and convert LAB → BGR
    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


# =============================================================================
# ─── ON-SCREEN HUD FOR CAPTURE WINDOW ────────────────────────────────────────
# =============================================================================

def _draw_capture_hud(
    frame: np.ndarray,
    class_name: str,
    captured: int,
    total_needed: int,
    status_msg: str = "",
    countdown: int = 0,
) -> np.ndarray:
    """
    Overlay a heads-up display on the live capture preview frame.

    Draws:
     • Semi-transparent top bar  → class name + progress
     • Semi-transparent bottom bar → keyboard shortcuts
     • Large countdown number (when counting down)
     • Optional status message

    Args:
        frame:         Current BGR camera frame.
        class_name:    The class currently being captured.
        captured:      Number of images saved so far.
        total_needed:  Total images to capture for this class.
        status_msg:    Short status line (e.g. "Saved!" or "Paused").
        countdown:     If > 0, draw a large countdown number.

    Returns:
        Annotated BGR frame (original is NOT modified in-place).
    """
    display = frame.copy()
    h, w = display.shape[:2]

    # ── Top info bar ────────────────────────────────────────────────────────
    bar_h = 50
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

    # Class name (left side)
    cv2.putText(
        display,
        f"Class: {class_name}",
        (12, 33),
        cv2.FONT_HERSHEY_DUPLEX, 0.65, (80, 230, 255), 1, cv2.LINE_AA,
    )

    # Progress counter (right side)
    pct = int(captured / total_needed * 100) if total_needed > 0 else 0
    prog_text = f"{captured}/{total_needed}  ({pct}%)"
    text_size, _ = cv2.getTextSize(prog_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
    cv2.putText(
        display, prog_text,
        (w - text_size[0] - 12, 33),
        cv2.FONT_HERSHEY_DUPLEX, 0.65, (80, 255, 150), 1, cv2.LINE_AA,
    )

    # Progress bar along the top edge
    bar_filled = int(w * captured / total_needed) if total_needed > 0 else 0
    cv2.rectangle(display, (0, bar_h - 4), (bar_filled, bar_h), (0, 220, 100), -1)

    # ── Bottom shortcuts bar ─────────────────────────────────────────────────
    bot_h = 38
    overlay2 = display.copy()
    cv2.rectangle(overlay2, (0, h - bot_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.75, display, 0.25, 0, display)

    shortcuts = "  [SPACE] Capture    [P] Pause    [Q] Quit class"
    cv2.putText(
        display, shortcuts,
        (8, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA,
    )

    # ── Status message (centre screen) ───────────────────────────────────────
    if status_msg:
        ts, _ = cv2.getTextSize(status_msg, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        tx = (w - ts[0]) // 2
        ty = (h + ts[1]) // 2
        # Shadow
        cv2.putText(display, status_msg, (tx+2, ty+2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, status_msg, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (80, 255, 80), 2, cv2.LINE_AA)

    # ── Countdown number ──────────────────────────────────────────────────────
    if countdown > 0:
        cd_text = str(countdown)
        ts, _ = cv2.getTextSize(cd_text, cv2.FONT_HERSHEY_DUPLEX, 5.0, 8)
        tx = (w - ts[0]) // 2
        ty = (h + ts[1]) // 2
        cv2.putText(display, cd_text, (tx+3, ty+3),
                    cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 0, 0), 12, cv2.LINE_AA)
        cv2.putText(display, cd_text, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 200, 255), 8, cv2.LINE_AA)

    return display


# =============================================================================
# ─── CAPTURE ENGINE ───────────────────────────────────────────────────────────
# =============================================================================

def _open_webcam() -> cv2.VideoCapture:
    """
    Open the webcam and raise a RuntimeError if it cannot be accessed.
    Sets a comfortable resolution; OpenCV will fall back gracefully if
    the hardware does not support it.
    """
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam at index {WEBCAM_INDEX}. "
            "Check that no other application is using it, or change "
            "WEBCAM_INDEX in config.py."
        )
    # Request 720p — camera may silently downgrade if unsupported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def _countdown(cap: cv2.VideoCapture, class_name: str, captured: int, total: int) -> None:
    """Show a live countdown overlay before capture begins."""
    for remaining in range(CAPTURE_COUNTDOWN, 0, -1):
        deadline = time.time() + 1.0
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = apply_clahe(frame)
            display = _draw_capture_hud(
                frame, class_name, captured, total,
                status_msg=f"Starting in...",
                countdown=remaining,
            )
            cv2.imshow("Logo Capture — Data Collector", display)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                return


def capture_class_images(class_name: str, target_count: int = IMAGES_PER_CLASS) -> int:
    """
    Run an interactive webcam capture session for a single class.

    The user can:
      • Press SPACE  — capture a single frame
      • Press A      — toggle auto-capture mode (captures every CAPTURE_DELAY s)
      • Press P      — pause / resume auto-capture
      • Press Q      — finish early and move on

    Captured images are saved as 224×224 JPEGs into either the train or val
    directory based on VAL_SPLIT.

    Args:
        class_name:   Must be one of the strings in CLASS_NAMES.
        target_count: How many images to capture (default: IMAGES_PER_CLASS).

    Returns:
        Number of images actually captured during this session.
    """
    print(section_header(f"Capturing:  {class_name}"))

    # Determine save paths
    train_path = os.path.join(TRAIN_DIR, class_name)
    val_path   = os.path.join(VAL_DIR,   class_name)

    info(f"Target images  : {target_count}")
    info(f"Train dir      : {os.path.relpath(train_path)}")
    info(f"Val dir        : {os.path.relpath(val_path)}")
    info(f"Existing images: {_count_images(train_path)} train / {_count_images(val_path)} val")
    print(DIVIDER_THIN)

    # Open the camera
    try:
        cap = _open_webcam()
    except RuntimeError as exc:
        error(str(exc))
        return 0

    success("Webcam opened successfully.")
    info("Controls: [SPACE] single capture  |  [A] toggle auto  |  [P] pause  |  [Q] quit")

    _countdown(cap, class_name, 0, target_count)

    captured      = 0
    auto_mode     = False
    paused        = False
    last_save_t   = 0.0
    status_msg    = ""
    status_expiry = 0.0

    while captured < target_count:
        ret, frame = cap.read()
        if not ret:
            warning("Frame read failed — skipping.")
            continue

        # Apply lighting normalisation
        frame = apply_clahe(frame)

        # Determine current status string (fades after 0.8 s)
        if time.time() < status_expiry:
            sm = status_msg
        else:
            sm = "AUTO" if (auto_mode and not paused) else ""

        display = _draw_capture_hud(frame, class_name, captured, target_count, sm)
        cv2.imshow("Logo Capture — Data Collector", display)

        key = cv2.waitKey(1) & 0xFF

        # ── Key handling ────────────────────────────────────────────────────
        if key == ord("q"):
            warning(f"Quit early — captured {captured} images for '{class_name}'.")
            break

        elif key == ord("a"):
            auto_mode = not auto_mode
            paused    = False
            status_msg    = "AUTO ON" if auto_mode else "AUTO OFF"
            status_expiry = time.time() + 1.2
            info(f"Auto-capture {'enabled' if auto_mode else 'disabled'}.")

        elif key == ord("p"):
            paused = not paused
            status_msg    = "PAUSED" if paused else "RESUMED"
            status_expiry = time.time() + 1.2

        should_capture = (
            (key == ord(" ")) or
            (auto_mode and not paused and (time.time() - last_save_t) >= CAPTURE_DELAY)
        )

        if should_capture:
            _save_frame(frame, class_name, train_path, val_path)
            captured    += 1
            last_save_t  = time.time()
            status_msg    = f"Saved #{captured}"
            status_expiry = time.time() + 0.6

    cap.release()
    cv2.destroyAllWindows()

    success(f"Capture complete — {captured} images saved for '{class_name}'.")
    _print_dataset_inventory()
    return captured


def _save_frame(
    frame: np.ndarray,
    class_name: str,
    train_path: str,
    val_path: str,
) -> None:
    """
    Resize *frame* to IMG_SIZE, decide train vs val, and write a JPEG.

    The train/val decision is made randomly on each frame so the split
    is approximate across the full session (law of large numbers).
    """
    # Resize to model input size
    resized = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # Decide split
    if random.random() < VAL_SPLIT:
        save_dir = val_path
        split    = "val"
    else:
        save_dir = train_path
        split    = "train"

    # Generate a timestamped filename for uniqueness
    ts       = int(time.time() * 1000)          # milliseconds
    rand_tag = random.randint(100, 999)
    filename = f"{class_name}_{split}_{ts}_{rand_tag}.jpg"
    filepath = os.path.join(save_dir, filename)

    cv2.imwrite(filepath, resized, [cv2.IMWRITE_JPEG_QUALITY, 95])


# =============================================================================
# ─── PUBLIC ENTRY POINT ───────────────────────────────────────────────────────
# =============================================================================

def run_data_collection() -> None:
    """
    Full interactive data collection workflow.

    1. Ensures all directories exist.
    2. Presents a menu so the user can choose which class(es) to capture.
    3. Runs the capture session for each selected class.
    4. Prints a final dataset inventory.
    """
    print(section_header("MODULE 1 — Interactive Data Collector"))

    # Guarantee directory tree exists
    ensure_dataset_directories()

    print(f"\n{DIVIDER_THIN}")
    info("You are about to capture training images using your webcam.")
    info("Hold the logo up to the camera clearly and steadily.")
    info(f"Target: {IMAGES_PER_CLASS} images per class  |  Val split: {int(VAL_SPLIT*100)}%")
    print(DIVIDER_THIN)

    # ── Class selection menu ─────────────────────────────────────────────────
    print(f"\n  {Colors.BOLD}Select classes to capture:{Colors.RESET}\n")
    for i, cn in enumerate(CLASS_NAMES, 1):
        existing = _count_images(os.path.join(TRAIN_DIR, cn)) + \
                   _count_images(os.path.join(VAL_DIR, cn))
        badge = f"{Colors.BR_GREEN}({existing} imgs){Colors.RESET}" if existing else ""
        print(f"    [{Colors.BR_YELLOW}{i}{Colors.RESET}] {cn}  {badge}")

    all_idx = len(CLASS_NAMES) + 1
    print(f"    [{Colors.BR_YELLOW}{all_idx}{Colors.RESET}] Capture ALL classes")
    print(f"    [{Colors.BR_YELLOW}0{Colors.RESET}] Back to main menu\n")

    raw = input(f"  {Colors.BOLD}Enter choice: {Colors.RESET}").strip()

    if raw == "0":
        info("Returning to main menu.")
        return

    if raw == str(all_idx):
        selected_classes = list(CLASS_NAMES)
    else:
        try:
            idx = int(raw) - 1
            if not (0 <= idx < len(CLASS_NAMES)):
                raise ValueError
            selected_classes = [CLASS_NAMES[idx]]
        except ValueError:
            error("Invalid selection. Returning to main menu.")
            return

    # ── Per-class capture loop ────────────────────────────────────────────────
    for class_name in selected_classes:
        # Ask for a custom count
        print(f"\n  {Colors.BOLD}How many images for '{class_name}'?{Colors.RESET}")
        raw_count = input(
            f"  (Press ENTER for default {IMAGES_PER_CLASS}): "
        ).strip()

        try:
            count = int(raw_count) if raw_count else IMAGES_PER_CLASS
            if count <= 0:
                raise ValueError
        except ValueError:
            warning(f"Invalid number — using default {IMAGES_PER_CLASS}.")
            count = IMAGES_PER_CLASS

        capture_class_images(class_name, count)

    print(section_header("Data Collection — Complete"))
    success("All selected classes have been captured.")
    info("You may now proceed to Module 2 to train the model.")
