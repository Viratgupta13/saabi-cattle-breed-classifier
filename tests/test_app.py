import io

import pytest
from PIL import Image

from app import create_app


class StubClassifier:
    class_names = ["Gir", "Murrah"]

    def predict(self, img, top_k=3):
        assert img.mode in ("RGB", "RGBA", "L", "P")
        return {"predictions": [{"breed": "Gir", "confidence": 0.9},
                                {"breed": "Murrah", "confidence": 0.1}],
                "status": self.status}

    status = "ok"


@pytest.fixture
def client():
    return create_app(StubClassifier()).test_client()


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "brown").save(buf, "PNG")
    buf.seek(0)
    return buf


def test_health(client):
    assert client.get("/api/health").get_json()["classes"] == ["Gir", "Murrah"]


def test_home_renders(client):
    assert b"SAABI" in client.get("/").data


def test_predict_ok(client):
    r = client.post("/api/predict", data={"image": (_png(), "cow.png")})
    body = r.get_json()
    assert r.status_code == 200
    assert body["predictions"][0]["breed"] == "Gir"
    assert body["status"] == "ok"
    assert body["info"]["origin"] == "Gujarat"


def test_predict_passes_through_unsupported_status():
    stub = StubClassifier()
    stub.status = "unsupported"
    c = create_app(stub).test_client()
    body = c.post("/api/predict", data={"image": (_png(), "x.png")}).get_json()
    assert body["status"] == "unsupported"


def test_predict_missing_file(client):
    assert client.post("/api/predict").status_code == 400


def test_predict_rejects_wrong_extension(client):
    r = client.post("/api/predict", data={"image": (io.BytesIO(b"x"), "notes.txt")})
    assert r.status_code == 400


def test_predict_rejects_corrupt_image(client):
    r = client.post("/api/predict", data={"image": (io.BytesIO(b"not an image"), "a.jpg")})
    assert r.status_code == 400
