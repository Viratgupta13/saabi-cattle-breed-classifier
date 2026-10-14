import numpy as np

from saabi import calibrate


def test_temperature_one_is_identity():
    p = np.array([[0.7, 0.2, 0.1]])
    assert np.allclose(calibrate.softmax_t(p, 1.0), p)


def test_higher_temperature_flattens():
    p = np.array([[0.9, 0.05, 0.05]])
    assert calibrate.softmax_t(p, 2.0).max() < 0.9


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 500)
    logits = rng.normal(size=(500, 3)) * 3  # noisy
    logits[np.arange(500), labels] += 2
    probs = calibrate.softmax_t(np.exp(logits) / np.exp(logits).sum(1, keepdims=True), 0.3)  # over-sharpened
    assert calibrate.fit_temperature(probs, labels) > 1.0


def test_ece_perfect_and_bad():
    labels = np.array([0, 1])
    assert calibrate.expected_calibration_error(np.array([[1.0, 0.0], [0.0, 1.0]]), labels) < 1e-9
    assert calibrate.expected_calibration_error(np.array([[1.0, 0.0], [1.0, 0.0]]), labels) > 0.4


def test_cosine_scores_far_vectors_lower():
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
    near = calibrate.cosine_to_centroids(np.array([[2.0, 0.1]]), centroids)[0]
    far = calibrate.cosine_to_centroids(np.array([[-1.0, -1.0]]), centroids)[0]
    assert near > 0.9 > far


def test_threshold_meets_target_accuracy():
    probs = np.array([[0.95, 0.05]] * 20 + [[0.55, 0.45]] * 20)
    labels = np.array([0] * 20 + [1] * 20)  # confident ones right, unsure ones wrong
    t = calibrate.pick_threshold(probs, labels, target=0.95)
    assert 0.55 < t <= 0.95
