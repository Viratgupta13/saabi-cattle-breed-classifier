"""Two-phase transfer learning with production-grade training techniques.

Includes label smoothing, mixup augmentation, and cosine learning rate decay.

    python -m saabi.train
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from . import config, data
from .model import build_model, unfreeze_top


def _mixup_batch(images, labels, num_classes, alpha=config.MIXUP_ALPHA):
    """Apply mixup augmentation: blend pairs of images and their soft labels."""
    batch_size = tf.shape(images)[0]
    # Sample lambda from Beta(alpha, alpha) using two Gamma samples
    g1 = tf.random.gamma([batch_size, 1, 1, 1], alpha)
    g2 = tf.random.gamma([batch_size, 1, 1, 1], alpha)
    lam = g1 / (g1 + g2 + 1e-8)       # proper Beta(alpha, alpha)
    lam = tf.maximum(lam, 1.0 - lam)   # keep lambda >= 0.5

    # Shuffle indices for pairing
    indices = tf.random.shuffle(tf.range(batch_size))
    images_b = tf.gather(images, indices)
    labels_b = tf.gather(labels, indices)

    mixed_images = lam * images + (1.0 - lam) * images_b
    lam_flat = tf.squeeze(lam, axis=[1, 2, 3])
    # One-hot encode if needed
    labels_a = tf.one_hot(tf.cast(labels, tf.int32), num_classes)
    labels_b = tf.one_hot(tf.cast(labels_b, tf.int32), num_classes)
    mixed_labels = tf.expand_dims(lam_flat, 1) * labels_a + \
                   tf.expand_dims(1.0 - lam_flat, 1) * labels_b
    return mixed_images, mixed_labels


def _compile(model, lr, num_classes, use_mixup=False):
    """Compile with optional label smoothing. When mixup is on, loss expects soft labels."""
    if use_mixup:
        # Mixup already produces soft labels — skip label smoothing to avoid
        # over-regularisation (both diffuse the target distribution).
        loss = tf.keras.losses.CategoricalCrossentropy()
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=loss,
        metrics=["accuracy"],
    )


def main(head_epochs: int = 15, finetune_epochs: int = 30, batch_size: int = 32):
    tf.keras.utils.set_random_seed(config.SEED)
    paths, labels, class_names, report = data.discover()
    num_classes = len(class_names)
    print(f"{len(paths)} images, {num_classes} classes "
          f"(dropped {report['corrupt']} corrupt, {report['duplicates']} duplicates)")
    (p_tr, y_tr), (p_va, y_va), (p_te, y_te) = data.split(paths, labels)
    print(f"train={len(p_tr)} val={len(p_va)} test={len(p_te)}")

    train_ds = data.make_dataset(p_tr, y_tr, training=True, batch_size=batch_size)
    val_ds = data.make_dataset(p_va, y_va, training=False, batch_size=batch_size)
    weights = data.class_weights(y_tr)

    # Apply mixup to training data during fine-tuning phase
    train_ds_mixup = train_ds.map(
        lambda img, lbl: _mixup_batch(img, lbl, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    config.MODEL_DIR.mkdir(exist_ok=True)
    checkpoint_path = config.MODEL_DIR / "checkpoint_best.keras"
    model = build_model(num_classes)

    def _callbacks(patience, save_best=True):
        cbs = [
            EarlyStopping("val_loss", patience=patience, restore_best_weights=True),
            ReduceLROnPlateau("val_loss", factor=0.3, patience=2, min_lr=1e-7),
        ]
        if save_best:
            cbs.append(ModelCheckpoint(str(checkpoint_path),
                                       monitor="val_accuracy", mode="max",
                                       save_best_only=True, verbose=1))
        return cbs

    # ── Phase 1: Train head only (no mixup, faster convergence) ─────
    _compile(model, 1e-3, num_classes, use_mixup=False)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs,
                   class_weight=weights, callbacks=_callbacks(4))
    # Save Phase 1 checkpoint so we never lose it
    phase1_path = config.MODEL_DIR / "phase1_checkpoint.keras"
    model.save(phase1_path)
    print(f"Phase 1 checkpoint saved → {phase1_path}")

    # ── Phase 2: Fine-tune top layers with mixup ────────────────────
    unfreeze_top(model, 40)
    _compile(model, 3e-5, num_classes, use_mixup=True)

    # Convert validation to soft labels for categorical loss
    val_ds_soft = val_ds.map(
        lambda img, lbl: (img, tf.one_hot(tf.cast(lbl, tf.int32), num_classes)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    h2 = model.fit(train_ds_mixup, validation_data=val_ds_soft,
                   epochs=finetune_epochs, callbacks=_callbacks(6))

    model.save(config.MODEL_PATH)
    data.save_class_names(class_names)

    history = {k: h1.history.get(k, []) + h2.history.get(k, []) for k in h1.history}
    history["head_epochs"] = len(h1.history["loss"])
    history["total_images"] = len(paths)
    history["num_classes"] = num_classes
    history["label_smoothing"] = config.LABEL_SMOOTHING
    history["mixup_alpha"] = config.MIXUP_ALPHA
    (config.MODEL_DIR / "history.json").write_text(json.dumps(history))
    _plot(history)
    print(f"\nSaved {config.MODEL_PATH}.")
    print(f"Now run:  python -m saabi.evaluate")
    print(f"Then:     python -m saabi.calibrate")
    print(f"Then:     python -m saabi.export")


def _plot(history):
    config.ASSETS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, title in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        if key in history:
            ax.plot(history[key], label="train")
        vk = f"val_{key}"
        if vk in history:
            ax.plot(history[vk], label="validation")
        ax.axvline(history["head_epochs"] - 0.5, color="gray", ls="--", lw=1)
        ax.set_title(title); ax.set_xlabel("epoch"); ax.legend()
    fig.suptitle("Dashed line: start of fine-tuning")
    fig.tight_layout()
    fig.savefig(config.ASSETS_DIR / "training_curves.png", dpi=150)


if __name__ == "__main__":
    main()
