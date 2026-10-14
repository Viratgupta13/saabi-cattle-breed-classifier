"""Two-phase transfer learning: train the head, then fine-tune the top backbone layers.

    python -m saabi.train
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from . import config, data
from .model import build_model, unfreeze_top


def _compile(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def main(head_epochs: int = 12, finetune_epochs: int = 25, batch_size: int = 32):
    tf.keras.utils.set_random_seed(config.SEED)
    paths, labels, class_names, report = data.discover()
    print(f"{len(paths)} images, {len(class_names)} classes "
          f"(dropped {report['corrupt']} corrupt, {report['duplicates']} duplicates)")
    (p_tr, y_tr), (p_va, y_va), (p_te, y_te) = data.split(paths, labels)
    print(f"train={len(p_tr)} val={len(p_va)} test={len(p_te)}")

    train_ds = data.make_dataset(p_tr, y_tr, training=True, batch_size=batch_size)
    val_ds = data.make_dataset(p_va, y_va, training=False, batch_size=batch_size)
    weights = data.class_weights(y_tr)

    model = build_model(len(class_names))
    callbacks = lambda patience: [
        EarlyStopping("val_loss", patience=patience, restore_best_weights=True),
        ReduceLROnPlateau("val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]

    _compile(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs,
                   class_weight=weights, callbacks=callbacks(4))

    unfreeze_top(model, 40)
    _compile(model, 3e-5)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=finetune_epochs,
                   class_weight=weights, callbacks=callbacks(6))

    config.MODEL_DIR.mkdir(exist_ok=True)
    model.save(config.MODEL_PATH)
    data.save_class_names(class_names)

    history = {k: h1.history[k] + h2.history[k] for k in h1.history}
    history["head_epochs"] = len(h1.history["loss"])
    (config.MODEL_DIR / "history.json").write_text(json.dumps(history))
    _plot(history)
    print(f"Saved {config.MODEL_PATH}. Now run: python -m saabi.evaluate")


def _plot(history):
    config.ASSETS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, title in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(history[key], label="train")
        ax.plot(history[f"val_{key}"], label="validation")
        ax.axvline(history["head_epochs"] - 0.5, color="gray", ls="--", lw=1)
        ax.set_title(title); ax.set_xlabel("epoch"); ax.legend()
    fig.suptitle("Dashed line: start of fine-tuning")
    fig.tight_layout()
    fig.savefig(config.ASSETS_DIR / "training_curves.png", dpi=150)


if __name__ == "__main__":
    main()
