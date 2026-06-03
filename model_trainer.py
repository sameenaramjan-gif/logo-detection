# =============================================================================
# model_trainer.py — Deep Learning CNN Trainer (Module 2)
# =============================================================================
# Responsibilities:
#   1. Validate that the dataset directory contains enough images to train.
#   2. Build tf.data pipelines with real-time augmentation for the train set.
#   3. Construct a custom CNN architecture suited for 224×224 logo detection.
#   4. Train with callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
#   5. Save the final model to disk and print a training summary.
#
# Dependencies:
#   pip install tensorflow numpy matplotlib
# =============================================================================

import os
import math
import numpy as np

# Suppress TensorFlow info / warning logs — only show errors
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from config import (
    CLASS_NAMES, NUM_CLASSES,
    TRAIN_DIR, VAL_DIR, MODELS_DIR, MODEL_PATH, LOGS_DIR,
    IMG_SIZE, IMG_SHAPE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, LR_DECAY_FACTOR, LR_PATIENCE,
    EARLY_STOP_PAT, MIN_LR, RANDOM_SEED,
    AUG_ROTATION_RANGE, AUG_WIDTH_SHIFT_RANGE, AUG_HEIGHT_SHIFT_RANGE,
    AUG_SHEAR_RANGE, AUG_ZOOM_RANGE, AUG_BRIGHTNESS_RANGE,
    AUG_HORIZONTAL_FLIP, AUG_FILL_MODE,
    DROPOUT_RATE, L2_REG,
    Colors, section_header, info, success, warning, error, step,
    DIVIDER_THIN,
)

# Attempt to import matplotlib for the optional training-curve plot
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — safe on all platforms
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# ─── DATASET VALIDATION ───────────────────────────────────────────────────────
# =============================================================================

def _count_images_in_split(split_dir: str) -> dict[str, int]:
    """Return a dict mapping class_name → image count for *split_dir*."""
    counts = {}
    for cn in CLASS_NAMES:
        class_dir = os.path.join(split_dir, cn)
        if os.path.isdir(class_dir):
            counts[cn] = sum(
                1 for f in os.listdir(class_dir)
                if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
            )
        else:
            counts[cn] = 0
    return counts


def validate_dataset() -> bool:
    """
    Check that every class has at least a few images in both splits.

    Returns:
        True if the dataset looks usable; False otherwise.
    """
    print(f"\n{DIVIDER_THIN}")
    info("Validating dataset…")

    train_counts = _count_images_in_split(TRAIN_DIR)
    val_counts   = _count_images_in_split(VAL_DIR)

    min_train = 10   # arbitrary minimum for a meaningful gradient step
    min_val   = 2

    ok = True
    print(f"\n  {'Class':<22} {'Train':>8} {'Val':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8}")

    for cn in CLASS_NAMES:
        t, v = train_counts[cn], val_counts[cn]
        tag  = ""
        if t < min_train:
            tag = f"  {Colors.BR_RED}⚠ Need ≥ {min_train} train images{Colors.RESET}"
            ok  = False
        elif v < min_val:
            tag = f"  {Colors.BR_YELLOW}⚠ Need ≥ {min_val} val images{Colors.RESET}"
            ok  = False
        print(f"  {cn:<22}  {t:>6}  {v:>8}{tag}")

    print(DIVIDER_THIN)

    if not ok:
        error("Dataset is insufficient. Run Module 1 first to collect more images.")
    else:
        success("Dataset validation passed.")

    return ok


# =============================================================================
# ─── DATA PIPELINE ────────────────────────────────────────────────────────────
# =============================================================================

def _build_augmentation_layer() -> keras.Sequential:
    """
    Build a Keras Sequential model that applies random augmentations.

    All augmentations are applied ONLY during training (via training=True).
    During inference the layers act as pass-through identity transforms.

    Augmentation strategy:
      • Random horizontal flip      — logos can appear on any side
      • Random rotation ±20°        — angled shots / tilted phones
      • Random zoom ±20%            — logo at varying distances
      • Random translation ±15%     — logo not always centred
      • Random contrast / brightness— indoor / outdoor lighting variety
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(
                AUG_ROTATION_RANGE / 360.0,   # Keras uses fractions of a full turn
                fill_mode=AUG_FILL_MODE,
            ),
            layers.RandomZoom(
                height_factor=(-AUG_ZOOM_RANGE, AUG_ZOOM_RANGE),
                fill_mode=AUG_FILL_MODE,
            ),
            layers.RandomTranslation(
                height_factor=AUG_HEIGHT_SHIFT_RANGE,
                width_factor=AUG_WIDTH_SHIFT_RANGE,
                fill_mode=AUG_FILL_MODE,
            ),
            layers.RandomContrast(factor=0.25),
            layers.RandomBrightness(
                factor=(AUG_BRIGHTNESS_RANGE[0] - 1.0, AUG_BRIGHTNESS_RANGE[1] - 1.0)
            ),
        ],
        name="augmentation",
    )


def build_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """
    Create optimised tf.data Datasets for training and validation.

    Processing pipeline:
        load_image → decode_jpeg → resize → normalise → (augment if train)
        → batch → prefetch

    Returns:
        (train_ds, val_ds, steps_per_epoch, validation_steps)
    """
    # ── Gather file paths and integer labels ─────────────────────────────────
    def _collect_paths_and_labels(split_dir: str):
        paths, labels = [], []
        for label_idx, cn in enumerate(CLASS_NAMES):
            class_dir = os.path.join(split_dir, cn)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if os.path.splitext(fname)[1].lower() in {".jpg", ".jpeg", ".png"}:
                    paths.append(os.path.join(class_dir, fname))
                    labels.append(label_idx)
        return paths, labels

    train_paths, train_labels = _collect_paths_and_labels(TRAIN_DIR)
    val_paths,   val_labels   = _collect_paths_and_labels(VAL_DIR)

    info(f"Train samples : {len(train_paths)}")
    info(f"Val samples   : {len(val_paths)}")

    if len(train_paths) == 0:
        raise ValueError("No training images found. Collect data first (Module 1).")

    # ── tf.data map function: load + decode + resize + normalise ─────────────
    @tf.function
    def load_and_preprocess(path: tf.Tensor, label: tf.Tensor):
        raw    = tf.io.read_file(path)
        image  = tf.image.decode_jpeg(raw, channels=3)
        image  = tf.image.resize(image, IMG_SIZE)
        image  = tf.cast(image, tf.float32) / 255.0   # scale to [0, 1]
        return image, label

    # ── Build raw datasets ────────────────────────────────────────────────────
    tf.random.set_seed(RANDOM_SEED)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .shuffle(len(train_paths), seed=RANDOM_SEED, reshuffle_each_iteration=True)
        .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    steps_per_epoch  = math.ceil(len(train_paths) / BATCH_SIZE)
    validation_steps = math.ceil(len(val_paths)   / BATCH_SIZE)

    return train_ds, val_ds, steps_per_epoch, validation_steps


# =============================================================================
# ─── CNN ARCHITECTURE ─────────────────────────────────────────────────────────
# =============================================================================

def build_model() -> keras.Model:
    """
    Build and return a custom CNN for multi-class logo classification.

    Architecture overview
    ─────────────────────
    Input (224×224×3)
        │
        ├─ Augmentation block (random flip / rotate / zoom / brightness)
        │    — active during training, identity during inference
        │
        ├─ Block 1: Conv(32, 3×3) → BN → ReLU → MaxPool(2×2)
        ├─ Block 2: Conv(64, 3×3) → BN → ReLU → MaxPool(2×2)
        ├─ Block 3: Conv(128,3×3) → BN → ReLU → MaxPool(2×2)
        ├─ Block 4: Conv(256,3×3) → BN → ReLU → GlobalAvgPool
        │
        ├─ Dense(512) → BN → ReLU → Dropout(0.5)
        └─ Dense(NUM_CLASSES, softmax)   ← output probabilities

    Design decisions
    ────────────────
    • GlobalAveragePooling (not Flatten) — fewer params, less overfitting.
    • BatchNormalisation after every Conv — faster convergence, more stable.
    • L2 kernel regularisation — penalises large weights, reduces overfitting.
    • Augmentation baked into the graph — no separate preprocessing step
      needed at inference time; just feed a raw [0,1]-normalised image.

    Returns:
        Compiled keras.Model ready for model.fit().
    """
    reg = regularizers.L2(L2_REG)

    inputs = keras.Input(shape=IMG_SHAPE, name="input_image")

    # ── Data augmentation (training only) ────────────────────────────────────
    x = _build_augmentation_layer()(inputs, training=True)  # NOTE: see below *

    # ─────────────────────────────────────────────────────────────────────────
    # * We pass training=True as a STATIC argument here so that Keras treats
    #   augmentation as "always training" when called via model.fit(), but the
    #   layers themselves check the global learning_phase flag and silently
    #   become no-ops during model.predict() / model.evaluate().
    # ─────────────────────────────────────────────────────────────────────────

    # ── Convolutional Block 1 ─────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg,
                      use_bias=False, name="conv1_1")(x)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="act1_1")(x)

    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg,
                      use_bias=False, name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="act1_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # ── Convolutional Block 2 ─────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg,
                      use_bias=False, name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="act2_1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg,
                      use_bias=False, name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="act2_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # ── Convolutional Block 3 ─────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg,
                      use_bias=False, name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="act3_1")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg,
                      use_bias=False, name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="act3_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # ── Convolutional Block 4 ─────────────────────────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=reg,
                      use_bias=False, name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu", name="act4_1")(x)

    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=reg,
                      use_bias=False, name="conv4_2")(x)
    x = layers.BatchNormalization(name="bn4_2")(x)
    x = layers.Activation("relu", name="act4_2")(x)

    # Global Average Pooling replaces Flatten — much fewer parameters
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # ── Fully Connected Head ──────────────────────────────────────────────────
    x = layers.Dense(512, kernel_regularizer=reg, use_bias=False, name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc")(x)
    x = layers.Activation("relu", name="act_fc")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="LogoCNN")

    # ── Compilation ───────────────────────────────────────────────────────────
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =============================================================================
# ─── CALLBACKS ────────────────────────────────────────────────────────────────
# =============================================================================

def build_callbacks(checkpoint_path: str) -> list:
    """
    Return a list of Keras callbacks for the training loop.

    Callbacks included:
    ┌─────────────────────────┬────────────────────────────────────────────┐
    │ ModelCheckpoint          │ Save the best val_accuracy weights to disk │
    │ EarlyStopping            │ Stop if val_loss doesn't improve for N ep  │
    │ ReduceLROnPlateau        │ Halve LR when val_loss plateaus            │
    │ TensorBoard (optional)   │ Log metrics for browser visualisation      │
    └─────────────────────────┴────────────────────────────────────────────┘
    """
    callbacks = []

    # Best-model checkpoint
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )
    )

    # Early stopping
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PAT,
            restore_best_weights=True,
            verbose=1,
        )
    )

    # Learning-rate reduction on plateau
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_DECAY_FACTOR,
            patience=LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        )
    )

    # TensorBoard (safe to skip if logs dir cannot be created)
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=LOGS_DIR,
                histogram_freq=1,
                update_freq="epoch",
            )
        )
        info(f"TensorBoard logs → {LOGS_DIR}")
        info("  Run: tensorboard --logdir logs/")
    except Exception:
        warning("TensorBoard callback skipped — could not create logs directory.")

    return callbacks


# =============================================================================
# ─── TRAINING CURVE PLOT ─────────────────────────────────────────────────────
# =============================================================================

def _save_training_plot(history: keras.callbacks.History) -> None:
    """
    Save a four-panel training-curve figure to the models directory.
    Skipped gracefully if Matplotlib is not installed.
    """
    if not MATPLOTLIB_AVAILABLE:
        warning("Matplotlib not installed — training curve plot skipped.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History — Logo Detection CNN", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train acc",  color="#2196F3")
    axes[0].plot(history.history["val_accuracy"], label="Val acc",    color="#4CAF50")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train loss", color="#F44336")
    axes[1].plot(history.history["val_loss"], label="Val loss",   color="#FF9800")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(MODELS_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    success(f"Training curves saved → {os.path.relpath(plot_path)}")


# =============================================================================
# ─── PUBLIC ENTRY POINT ───────────────────────────────────────────────────────
# =============================================================================

def run_training() -> None:
    """
    Full training workflow:

    1. Validate the dataset.
    2. Build tf.data pipelines.
    3. Construct the CNN.
    4. Train with callbacks.
    5. Evaluate on the validation set.
    6. Save the model.
    7. Optionally plot training curves.
    """
    print(section_header("MODULE 2 — Deep Learning CNN Trainer"))

    # ── Step 1: validate dataset ──────────────────────────────────────────────
    if not validate_dataset():
        return

    # ── Step 2: build data pipelines ─────────────────────────────────────────
    print(f"\n{DIVIDER_THIN}")
    step("Building tf.data pipelines…")
    try:
        train_ds, val_ds, steps_per_epoch, val_steps = build_datasets()
    except ValueError as exc:
        error(str(exc))
        return

    # ── Step 3: build model ───────────────────────────────────────────────────
    step("Constructing CNN architecture…")
    model = build_model()

    # Print a clean model summary
    print()
    model.summary(line_length=70, expand_nested=False)
    print()

    total_params = model.count_params()
    info(f"Total parameters : {total_params:,}")

    # ── Step 4: prepare output directory ─────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODELS_DIR, "best_checkpoint.keras")

    # ── Step 5: build callbacks ───────────────────────────────────────────────
    callbacks = build_callbacks(checkpoint_path)

    # ── Step 6: print training config ────────────────────────────────────────
    print(f"\n{DIVIDER_THIN}")
    info(f"Batch size        : {BATCH_SIZE}")
    info(f"Max epochs        : {EPOCHS}")
    info(f"Initial LR        : {LEARNING_RATE}")
    info(f"Early-stop pat.   : {EARLY_STOP_PAT}")
    info(f"Steps / epoch     : {steps_per_epoch}")
    info(f"Val steps         : {val_steps}")
    print(DIVIDER_THIN)

    confirm = input(
        f"\n  {Colors.BOLD}Start training? [Y/n]: {Colors.RESET}"
    ).strip().lower()

    if confirm not in ("", "y", "yes"):
        warning("Training cancelled.")
        return

    print(f"\n{section_header('Training Loop')}\n")

    # ── Step 7: train ─────────────────────────────────────────────────────────
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1,
        )
    except KeyboardInterrupt:
        warning("\nTraining interrupted by user.")
        return

    # ── Step 8: final evaluation ──────────────────────────────────────────────
    print(f"\n{DIVIDER_THIN}")
    step("Running final evaluation on validation set…")
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    success(f"Validation loss     : {val_loss:.4f}")
    success(f"Validation accuracy : {val_acc * 100:.2f}%")

    # ── Step 9: save model ────────────────────────────────────────────────────
    step(f"Saving model → {os.path.relpath(MODEL_PATH)}")
    model.save(MODEL_PATH)
    success(f"Model saved successfully.")

    # ── Step 10: training curves ──────────────────────────────────────────────
    _save_training_plot(history)

    print(section_header("Training Complete"))
    success(f"Best model: {os.path.relpath(MODEL_PATH)}")
    info("Proceed to Module 3 for real-time inference.")
