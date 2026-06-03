# =============================================================================
# webcam_inference.py — Real-Time Live Inference Engine (Module 3)
# =============================================================================
# Responsibilities:
#   1. Load the trained Keras model from disk.
#   2. Open the webcam and process a continuous video feed.
#   3. Crop the Region-of-Interest (ROI) from each frame.
#   4. Preprocess the ROI and run inference.
#   5. Render a polished, colour-coded HUD overlay on the OpenCV window.
#
# HUD elements:
#   • ROI bounding box  — green / amber / red based on confidence
#   • Prediction label  — class name + percentage, inside a filled box
#   • Top-3 probability bar chart — drawn directly on the frame
#   • FPS counter       — rendered in the top-right corner
#   • Keyboard shortcut legend at the bottom
#
# Dependencies:
#   pip install tensorflow opencv-python numpy
# =============================================================================

import os
import time
import collections

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress TF info logs

import tensorflow as tf

from config import (
    CLASS_NAMES, NUM_CLASSES, MODEL_PATH,
    IMG_SIZE,
    WEBCAM_INDEX,
    HIGH_CONF_THRESHOLD, MED_CONF_THRESHOLD,
    COLOR_HIGH, COLOR_MED, COLOR_LOW,
    ROI_TOP_LEFT_RATIO, ROI_BOTTOM_RIGHT_RATIO,
    OSD_FONT_SCALE, OSD_FONT_THICKNESS, OSD_PADDING,
    WINDOW_TITLE,
    Colors, section_header, info, success, warning, error, step,
    DIVIDER_THIN,
)


# =============================================================================
# ─── MODEL LOADER ─────────────────────────────────────────────────────────────
# =============================================================================

def load_model() -> tf.keras.Model:
    """
    Load the trained Keras model from MODEL_PATH.

    Raises:
        FileNotFoundError — if the model file does not exist yet.
        RuntimeError       — if the model fails to load.

    Returns:
        Loaded, ready-to-use keras.Model.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at:\n    {MODEL_PATH}\n\n"
            "Please train the model first using Module 2."
        )

    step(f"Loading model from {os.path.relpath(MODEL_PATH)} …")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        success("Model loaded successfully.")
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc


# =============================================================================
# ─── PREPROCESSING ────────────────────────────────────────────────────────────
# =============================================================================

def preprocess_roi(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Prepare a BGR ROI crop for model inference.

    Steps:
        1. Resize to model input size (IMG_SIZE).
        2. Convert BGR → RGB (TensorFlow/Keras uses RGB).
        3. Normalise pixel values to [0.0, 1.0].
        4. Add batch dimension → shape (1, H, W, 3).

    Args:
        roi_bgr: Cropped region of interest as a BGR NumPy array.

    Returns:
        Float32 tensor of shape (1, 224, 224, 3).
    """
    resized = cv2.resize(roi_bgr, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm    = rgb.astype(np.float32) / 255.0
    return np.expand_dims(norm, axis=0)   # (1, H, W, 3)


# =============================================================================
# ─── COLOUR SELECTOR ─────────────────────────────────────────────────────────
# =============================================================================

def _confidence_color(confidence: float) -> tuple[int, int, int]:
    """
    Map a confidence score to a BGR colour tuple.

    ≥ HIGH_CONF_THRESHOLD → bright green  (confident)
    ≥ MED_CONF_THRESHOLD  → amber         (moderate)
    <  MED_CONF_THRESHOLD → red           (uncertain)
    """
    if confidence >= HIGH_CONF_THRESHOLD:
        return COLOR_HIGH
    elif confidence >= MED_CONF_THRESHOLD:
        return COLOR_MED
    else:
        return COLOR_LOW


# =============================================================================
# ─── HUD RENDERING UTILITIES ─────────────────────────────────────────────────
# =============================================================================

def _draw_filled_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple,
    font_scale: float = OSD_FONT_SCALE,
    thickness: int = OSD_FONT_THICKNESS,
    padding: int = OSD_PADDING,
    above: bool = True,
) -> None:
    """
    Draw *text* inside a filled, semi-transparent rectangle on *frame*.

    Args:
        frame:      The frame to draw on (modified in-place).
        text:       The string to render.
        x, y:       Top-left corner of the associated bounding box.
        color:      BGR fill colour for the label box.
        font_scale: OpenCV font scale.
        thickness:  Text line thickness.
        padding:    Pixel padding around the text inside the box.
        above:      If True, position the label above (x, y); else below.
    """
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    box_w = tw + 2 * padding
    box_h = th + baseline + 2 * padding

    if above:
        box_y1 = y - box_h
        box_y2 = y
    else:
        box_y1 = y
        box_y2 = y + box_h

    box_x1 = x
    box_x2 = x + box_w

    # Clamp to frame bounds
    h_frame, w_frame = frame.shape[:2]
    box_x1 = max(0, box_x1)
    box_y1 = max(0, box_y1)
    box_x2 = min(w_frame, box_x2)
    box_y2 = min(h_frame, box_y2)

    # Semi-transparent fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), color, -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # White text on top
    text_x = box_x1 + padding
    text_y = box_y2 - padding - baseline
    cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_roi_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
    thickness: int = 2,
    corner_len: int = 20,
) -> None:
    """
    Draw a stylised bounding box with corner brackets (not a full rectangle).

    Corner brackets look more professional than a plain rectangle, and are
    common in modern AR / HUD interfaces.

    Args:
        frame:      Frame to draw on (in-place).
        x1,y1:      Top-left corner.
        x2,y2:      Bottom-right corner.
        color:      BGR line colour.
        thickness:  Line thickness.
        corner_len: Pixel length of each corner bracket arm.
    """
    # Top-left corner
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)

    # Top-right corner
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)

    # Bottom-left corner
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)

    # Bottom-right corner
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    # Thin dashed inner rectangle for depth — approximate with dotted lines
    # (full rect at lower opacity via overlay trick)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def _draw_probability_bars(
    frame: np.ndarray,
    probabilities: np.ndarray,
    origin_x: int,
    origin_y: int,
    bar_width: int = 160,
    bar_height: int = 16,
    gap: int = 6,
) -> None:
    """
    Draw horizontal probability bars for all classes.

    Each bar shows:
        [class name]  ████████░░░░ 87.3%

    Args:
        frame:         Frame to draw on (in-place).
        probabilities: 1-D float array of length NUM_CLASSES.
        origin_x:      Left edge of the bar panel.
        origin_y:      Top edge of the first bar.
        bar_width:     Maximum bar length in pixels.
        bar_height:    Height of each bar in pixels.
        gap:           Pixel gap between rows.
    """
    font          = cv2.FONT_HERSHEY_SIMPLEX
    label_font_sc = 0.42
    val_font_sc   = 0.42

    panel_h = NUM_CLASSES * (bar_height + gap) + gap + 6
    label_col_w = 120   # reserved pixels for class name text

    # Semi-transparent background panel
    total_w = label_col_w + bar_width + 60
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (origin_x - 6, origin_y - 6),
        (origin_x + total_w, origin_y + panel_h),
        (15, 15, 15), -1,
    )
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    for i, (cn, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
        row_y = origin_y + i * (bar_height + gap)

        # Shorten long class names to fit
        display_name = cn.replace("brand_", "")
        display_name = display_name[:14]

        # Background track
        cv2.rectangle(
            frame,
            (origin_x + label_col_w, row_y),
            (origin_x + label_col_w + bar_width, row_y + bar_height),
            (50, 50, 50), -1,
        )

        # Filled bar — colour depends on confidence
        filled_w = int(prob * bar_width)
        bar_color = _confidence_color(float(prob))
        cv2.rectangle(
            frame,
            (origin_x + label_col_w, row_y),
            (origin_x + label_col_w + filled_w, row_y + bar_height),
            bar_color, -1,
        )

        # Class name label
        cv2.putText(frame, display_name,
                    (origin_x, row_y + bar_height - 4),
                    font, label_font_sc, (200, 200, 200), 1, cv2.LINE_AA)

        # Percentage
        pct_text = f"{prob * 100:.1f}%"
        cv2.putText(frame, pct_text,
                    (origin_x + label_col_w + bar_width + 6, row_y + bar_height - 4),
                    font, val_font_sc, (220, 220, 220), 1, cv2.LINE_AA)


def _draw_fps(frame: np.ndarray, fps: float) -> None:
    """Render an FPS counter in the top-right corner."""
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 1)
    x = w - tw - 10
    y = th + 10
    cv2.putText(frame, text, (x + 1, y + 1), font, 0.55, (0, 0, 0),   2, cv2.LINE_AA)
    cv2.putText(frame, text, (x,     y    ), font, 0.55, (200, 255, 200), 1, cv2.LINE_AA)


def _draw_bottom_bar(frame: np.ndarray) -> None:
    """Draw a semi-transparent footer with keyboard shortcuts."""
    h, w = frame.shape[:2]
    bar_h = 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    shortcuts = "  [Q] Quit    [S] Screenshot    [F] Freeze frame"
    cv2.putText(frame, shortcuts,
                (8, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


# =============================================================================
# ─── MAIN INFERENCE LOOP ──────────────────────────────────────────────────────
# =============================================================================

def run_inference() -> None:
    """
    Main entry point for Module 3: Real-Time Live Inference Engine.

    Loop:
        1. Read a frame from the webcam.
        2. Compute the ROI pixel coordinates from the frame size.
        3. Crop and preprocess the ROI.
        4. Run model.predict() on the ROI.
        5. Render the full HUD on the display frame.
        6. Show the frame in an OpenCV window.
        7. Handle keyboard input (Q = quit, S = screenshot, F = freeze).
    """
    print(section_header("MODULE 3 — Real-Time Live Inference"))

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model = load_model()
    except (FileNotFoundError, RuntimeError) as exc:
        error(str(exc))
        return

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        error(
            f"Cannot open webcam at index {WEBCAM_INDEX}. "
            "Check that no other application is using it, or change "
            "WEBCAM_INDEX in config.py."
        )
        return

    # Request 720p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    success("Webcam opened. Starting inference loop.")
    info("Hold a logo inside the bracket box to detect it.")
    print(DIVIDER_THIN)

    # ── FPS tracking ─────────────────────────────────────────────────────────
    fps_buffer       = collections.deque(maxlen=30)
    prev_time        = time.time()

    # ── State ─────────────────────────────────────────────────────────────────
    frozen           = False          # Freeze-frame toggle
    frozen_frame     = None           # Buffer for the frozen display frame
    screenshot_dir   = "screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)

    # Smoothing: keep a short rolling buffer of predictions to reduce flicker
    pred_buffer      = collections.deque(maxlen=5)

    try:
        while True:
            # ── Capture frame ────────────────────────────────────────────────
            if not frozen:
                ret, frame = cap.read()
                if not ret:
                    warning("Frame read failed — retrying…")
                    continue

                h_frame, w_frame = frame.shape[:2]

                # ── Compute ROI coordinates ───────────────────────────────────
                x1 = int(ROI_TOP_LEFT_RATIO[0]     * w_frame)
                y1 = int(ROI_TOP_LEFT_RATIO[1]     * h_frame)
                x2 = int(ROI_BOTTOM_RIGHT_RATIO[0] * w_frame)
                y2 = int(ROI_BOTTOM_RIGHT_RATIO[1] * h_frame)

                roi_bgr = frame[y1:y2, x1:x2]

                # ── Inference ────────────────────────────────────────────────
                if roi_bgr.size > 0:
                    input_tensor = preprocess_roi(roi_bgr)
                    raw_probs    = model.predict(input_tensor, verbose=0)[0]   # shape (NUM_CLASSES,)
                    pred_buffer.append(raw_probs)

                # Smoothed probabilities (mean over rolling window)
                if pred_buffer:
                    smoothed_probs = np.mean(np.stack(pred_buffer), axis=0)
                else:
                    smoothed_probs = np.ones(NUM_CLASSES) / NUM_CLASSES

                top_idx    = int(np.argmax(smoothed_probs))
                top_conf   = float(smoothed_probs[top_idx])
                top_class  = CLASS_NAMES[top_idx]

                # ── Choose colours based on confidence ───────────────────────
                box_color = _confidence_color(top_conf)

                # ── FPS calculation ───────────────────────────────────────────
                now      = time.time()
                fps_buffer.append(1.0 / max(now - prev_time, 1e-6))
                prev_time = now
                fps       = float(np.mean(fps_buffer))

                # ── Build display frame ───────────────────────────────────────
                display = frame.copy()

                # Dark vignette on the periphery outside ROI (subtle depth cue)
                vignette = display.copy()
                cv2.rectangle(vignette, (0, 0), (w_frame, h_frame), (0, 0, 0), -1)
                # Punch out the ROI area so it stays bright
                vignette[y1:y2, x1:x2] = display[y1:y2, x1:x2]
                cv2.addWeighted(vignette, 0.18, display, 0.82, 0, display)

                # ROI bounding box with corner brackets
                _draw_roi_box(display, x1, y1, x2, y2, box_color, thickness=2)

                # Main prediction label (above the ROI box)
                # Format: "brand_nike  94.7%"
                clean_name  = top_class.replace("_", " ").title()
                label_text  = f"{clean_name}  {top_conf * 100:.1f}%"
                _draw_filled_label(
                    display, label_text,
                    x1, y1,
                    box_color,
                    font_scale=0.85,
                    thickness=2,
                    padding=10,
                    above=True,
                )

                # Probability bar chart — bottom-left corner
                _draw_probability_bars(
                    display, smoothed_probs,
                    origin_x=12,
                    origin_y=h_frame - (NUM_CLASSES * 22 + 50),
                )

                # FPS counter
                _draw_fps(display, fps)

                # Bottom shortcuts bar
                _draw_bottom_bar(display)

                # Freeze indicator
                if frozen:
                    cv2.putText(display, "[ FROZEN ]",
                                (w_frame // 2 - 60, 40),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)

                frozen_frame = display.copy()

            else:
                # Show the frozen frame when paused
                display = frozen_frame.copy() if frozen_frame is not None else np.zeros((480, 640, 3), np.uint8)
                cv2.putText(display, "[ FROZEN — Press F to resume ]",
                            (20, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA)

            # ── Show frame ───────────────────────────────────────────────────
            cv2.imshow(WINDOW_TITLE, display)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:   # Q or ESC to quit
                info("Quit requested.")
                break

            elif key == ord("f"):              # F — freeze / unfreeze
                frozen = not frozen
                status = "Frozen" if frozen else "Live"
                info(f"Display {status}.")

            elif key == ord("s"):              # S — save screenshot
                ts         = int(time.time())
                shot_path  = os.path.join(screenshot_dir, f"screenshot_{ts}.jpg")
                cv2.imwrite(shot_path, display)
                success(f"Screenshot saved → {shot_path}")

    except KeyboardInterrupt:
        warning("Inference interrupted by user (Ctrl+C).")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        success("Webcam released. Inference engine shut down cleanly.")
