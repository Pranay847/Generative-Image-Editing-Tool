"""Offline tests for HFClient: token check, caching, fallbacks (no network)."""
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hf_client  # noqa: E402
from hf_client import HFClient  # noqa: E402


class FakeInference:
    """Stands in for huggingface_hub.InferenceClient; counts calls."""

    instances = []

    def __init__(self, model, token, timeout):
        self.model, self.calls = model, []
        FakeInference.instances.append(self)

    def text_to_image(self, **kw):
        self.calls.append(("t2i", kw))
        return Image.new("RGB", (4, 4), (10, 20, 30))

    def image_to_image(self, **kw):
        self.calls.append(("i2i", kw))
        raise RuntimeError("provider down")


@pytest.fixture
def client(monkeypatch):
    FakeInference.instances = []
    monkeypatch.setattr(hf_client, "InferenceClient", FakeInference)
    return HFClient(token="test-token")


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_API_TOKEN"):
        HFClient()


def test_text_to_image_cached_on_repeat(client):
    a = client.text_to_image("a cat", seed=1)
    b = client.text_to_image("a cat", seed=1)
    assert a.tobytes() == b.tobytes()
    assert len(FakeInference.instances) == 1
    assert len(FakeInference.instances[0].calls) == 1
    client.text_to_image("a cat", seed=2)  # different seed -> new call
    assert len(FakeInference.instances[0].calls) == 2


def test_negative_prompt_and_seed_only_sent_when_set(client):
    client.text_to_image("x")
    kw = FakeInference.instances[0].calls[0][1]
    assert "negative_prompt" not in kw and "seed" not in kw
    client.text_to_image("y", negative_prompt="blur", seed=0)
    kw = FakeInference.instances[0].calls[1][1]
    assert kw["negative_prompt"] == "blur" and kw["seed"] == 0


def test_img2img_falls_back_to_text2img_and_caches(client):
    src = Image.new("RGB", (4, 4), (255, 0, 0))
    out = client.image_to_image(src, "make it blue")
    assert out.size == (4, 4)
    models = {fi.model: [c[0] for c in fi.calls] for fi in FakeInference.instances}
    assert models[hf_client.MODELS["img2img"]] == ["i2i"]
    assert models[hf_client.MODELS["text2img"]] == ["t2i"]
    client.image_to_image(src, "make it blue")  # served from cache
    assert sum(len(fi.calls) for fi in FakeInference.instances) == 2


def test_cache_key_depends_on_image_pixels(client):
    red = Image.new("RGB", (4, 4), (255, 0, 0))
    green = Image.new("RGB", (4, 4), (0, 255, 0))
    assert client._key("i2i", red, "p") != client._key("i2i", green, "p")
    assert client._key("i2i", red, "p") == client._key("i2i", red.copy(), "p")


def test_to_pil_rejects_unexpected_type():
    with pytest.raises(RuntimeError, match="Unexpected response type"):
        HFClient._to_pil({"not": "an image"})
