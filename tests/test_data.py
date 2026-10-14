import numpy as np

from saabi import data


def _fake(n_per_class=20, n_classes=4):
    paths = np.array([f"c{c}_{i}.jpg" for c in range(n_classes) for i in range(n_per_class)])
    labels = np.repeat(np.arange(n_classes), n_per_class)
    return paths, labels


def test_splits_are_disjoint_and_complete():
    paths, labels = _fake()
    (a, _), (b, _), (c, _) = data.split(paths, labels)
    assert not set(a) & set(b) and not set(a) & set(c) and not set(b) & set(c)
    assert len(a) + len(b) + len(c) == len(paths)


def test_splits_are_stratified():
    paths, labels = _fake()
    for _, y in data.split(paths, labels):
        assert set(y) == set(labels)


def test_split_is_deterministic():
    paths, labels = _fake()
    first = data.split(paths, labels)[2][0]
    second = data.split(paths, labels)[2][0]
    assert list(first) == list(second)


def test_class_weights_favour_rare_classes():
    w = data.class_weights(np.array([0] * 100 + [1] * 20))
    assert w[1] > w[0]
