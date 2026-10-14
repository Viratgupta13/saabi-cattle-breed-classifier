"""Temperature scaling + a confidence threshold for rejecting uncertain/unknown images.

Fit on the validation split only; the test split is used just to report results.

    python -m saabi.calibrate
"""
import json

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize_scalar

from . import config, data
from .scoring import cosine_to_centroids, softmax_t  # noqa: F401  (re-exported)


def nll(probs, labels):
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1))))


def fit_temperature(probs, labels) -> float:
    res = minimize_scalar(lambda t: nll(softmax_t(probs, t), labels),
                          bounds=(0.3, 5.0), method="bounded")
    return float(res.x)


def expected_calibration_error(probs, labels, bins: int = 10) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def pick_threshold(probs, labels, target: float = config.TARGET_SELECTIVE_ACCURACY) -> float:
    """Lowest confidence cutoff whose accepted predictions reach the target accuracy."""
    conf, correct = probs.max(1), probs.argmax(1) == labels
    for t in np.unique(conf):
        accepted = conf >= t
        if accepted.sum() >= 10 and correct[accepted].mean() >= target:
            return float(t)
    return float(conf.max())


def selective_report(probs, labels, threshold):
    conf, correct = probs.max(1), probs.argmax(1) == labels
    accepted = conf >= threshold
    return {
        "coverage": float(accepted.mean()),
        "accuracy_on_accepted": float(correct[accepted].mean()) if accepted.any() else None,
    }


def feature_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """Model returning (pooled backbone features, class probabilities)."""
    gap = next(l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
    return tf.keras.Model(model.input, [gap.output, model.output])


def main():
    paths, labels, class_names, _ = data.discover()
    (p_tr, y_tr), (p_va, y_va), (p_te, y_te) = data.split(paths, labels)
    model = tf.keras.models.load_model(config.MODEL_PATH)
    extractor = feature_extractor(model)

    def run(p, y):
        feats, probs = extractor.predict(data.make_dataset(p, y, training=False), verbose=0)
        return feats, probs

    tr_feats, _ = run(p_tr, y_tr)
    va_feats, val_probs = run(p_va, y_va)
    te_feats, test_probs = run(p_te, y_te)

    centroids = np.stack([tr_feats[y_tr == c].mean(0) for c in range(len(class_names))])
    np.save(config.OOD_PATH, centroids)
    ood_threshold = float(np.quantile(cosine_to_centroids(va_feats, centroids),
                                      1 - config.OOD_ACCEPT_FRACTION))

    temperature = fit_temperature(val_probs, y_va)
    val_cal, test_cal = softmax_t(val_probs, temperature), softmax_t(test_probs, temperature)
    threshold = pick_threshold(val_cal, y_va)

    result = {
        "temperature": temperature,
        "threshold": threshold,
        "ood_threshold": ood_threshold,
        "test": {
            "in_distribution_accepted": float(
                (cosine_to_centroids(te_feats, centroids) >= ood_threshold).mean()),
            "ece_before": expected_calibration_error(test_probs, y_te),
            "ece_after": expected_calibration_error(test_cal, y_te),
            "nll_before": nll(test_probs, y_te),
            "nll_after": nll(test_cal, y_te),
            **selective_report(test_cal, y_te, threshold),
        },
    }
    config.CALIBRATION_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
