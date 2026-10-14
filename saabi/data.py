"""Dataset discovery, cleaning, splitting and tf.data pipelines."""
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from . import config


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def discover(data_dir: Path = config.DATA_DIR):
    """Return (paths, labels, class_names, report).

    Unreadable files and exact byte-for-byte duplicates are dropped so the
    same picture can never land in both the train and test splits.
    """
    data_dir = Path(data_dir)
    class_names = sorted(
        d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    paths, labels = [], []
    seen = set()
    report = {"corrupt": 0, "duplicates": 0}
    for idx, breed in enumerate(class_names):
        for p in sorted((data_dir / breed).iterdir()):
            if p.suffix.lower() not in config.IMAGE_EXTENSIONS:
                continue
            if not _is_valid_image(p):
                report["corrupt"] += 1
                continue
            digest = hashlib.md5(p.read_bytes()).hexdigest()
            if digest in seen:
                report["duplicates"] += 1
                continue
            seen.add(digest)
            paths.append(str(p))
            labels.append(idx)
    return np.array(paths), np.array(labels), class_names, report


def split(paths, labels, seed: int = config.SEED):
    """Stratified train / val / test split."""
    holdout = config.VAL_FRACTION + config.TEST_FRACTION
    p_tr, p_ho, y_tr, y_ho = train_test_split(
        paths, labels, test_size=holdout, random_state=seed, stratify=labels
    )
    p_va, p_te, y_va, y_te = train_test_split(
        p_ho, y_ho, test_size=config.TEST_FRACTION / holdout,
        random_state=seed, stratify=y_ho,
    )
    return (p_tr, y_tr), (p_va, y_va), (p_te, y_te)


def _load(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    return img, label


def make_dataset(paths, labels, training: bool, batch_size: int = 32):
    """Yield (RGB float32 in 0..255, int label). Scaling happens inside the model."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_weights(labels):
    """Inverse-frequency weights to offset class imbalance."""
    counts = np.bincount(labels)
    return {i: float(len(labels) / (len(counts) * c)) for i, c in enumerate(counts)}


def save_class_names(class_names):
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.CLASSES_PATH.write_text(json.dumps(class_names, indent=2))


def load_class_names():
    return json.loads(config.CLASSES_PATH.read_text())
