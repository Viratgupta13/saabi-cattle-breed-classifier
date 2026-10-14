"""Evaluate on the held-out test split (never used for training or model selection).

    python -m saabi.evaluate
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from . import config, data


def main():
    paths, labels, class_names, _ = data.discover()
    _, _, (p_te, y_te) = data.split(paths, labels)
    model = tf.keras.models.load_model(config.MODEL_PATH)
    probs = model.predict(data.make_dataset(p_te, y_te, training=False), verbose=0)
    pred = probs.argmax(1)

    top3 = float(np.mean([y in np.argsort(p)[-3:] for p, y in zip(probs, y_te)]))
    metrics = {
        "test_images": int(len(y_te)),
        "accuracy": float((pred == y_te).mean()),
        "top3_accuracy": top3,
        "macro_f1": float(f1_score(y_te, pred, average="macro")),
        "per_class": classification_report(
            y_te, pred, target_names=class_names, output_dict=True, zero_division=0
        ),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(classification_report(y_te, pred, target_names=class_names, zero_division=0))
    print(f"accuracy={metrics['accuracy']:.4f}  top3={top3:.4f}  macro_f1={metrics['macro_f1']:.4f}")
    _confusion_plot(y_te, pred, class_names)


def _confusion_plot(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Test-set confusion matrix")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    config.ASSETS_DIR.mkdir(exist_ok=True)
    fig.savefig(config.ASSETS_DIR / "confusion_matrix.png", dpi=150)


if __name__ == "__main__":
    main()
