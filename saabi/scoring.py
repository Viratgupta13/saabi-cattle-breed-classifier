"""NumPy-only scoring helpers, so the inference server never needs TensorFlow."""
import numpy as np


def softmax_t(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Re-temper softmax outputs. log(p) equals the logits up to a per-row constant,
    which softmax ignores, so this is exactly softmax(logits / T)."""
    z = np.log(np.clip(probs, 1e-12, 1)) / temperature
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cosine_to_centroids(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Highest cosine similarity to any breed centroid; low means 'unlike every known breed'."""
    a = features / np.linalg.norm(features, axis=1, keepdims=True)
    b = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    return (a @ b.T).max(axis=1)
