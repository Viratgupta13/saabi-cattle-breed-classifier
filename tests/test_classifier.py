import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from saabi import config

pytestmark = pytest.mark.skipif(not config.TFLITE_PATH.exists(), reason="trained model not present")


def test_classifier_output_shape_and_probabilities():
    from saabi.predict import Classifier
    clf = Classifier()
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8))
    out = clf.predict(img, top_k=3)
    assert out["status"] in {"ok", "uncertain", "unsupported"}
    confs = [p["confidence"] for p in out["predictions"]]
    assert len(confs) == 3 and confs == sorted(confs, reverse=True)
    assert 0 < sum(confs) <= 1.0 + 1e-6


def test_noise_image_is_not_accepted_as_a_breed():
    from saabi.predict import Classifier
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (224, 224, 3), dtype=np.uint8))
    assert Classifier().predict(img)["status"] != "ok"


def test_serving_path_does_not_import_tensorflow():
    code = "import sys, saabi.predict, app; assert 'tensorflow' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=config.ROOT)
