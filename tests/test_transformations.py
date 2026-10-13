"""Offline tests: transformation routing (fake client) and OpenCV mask helpers."""
import inspect
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cv_utils import feather_mask, grabcut_mask, threshold_mask  # noqa: E402
from transformations import TRANSFORMATIONS  # noqa: E402


class FakeClient:
    """Records which pipeline was called instead of hitting Hugging Face."""

    def __init__(self):
        self.calls = []

    def text_to_image(self, prompt, **kw):
        self.calls.append(("text2img", prompt, None))
        return Image.new("RGB", (8, 8))

    def image_to_image(self, image, prompt, strength=0.6, **kw):
        self.calls.append(("img2img", prompt, image))
        return image

    def inpaint(self, image, mask, prompt, **kw):
        self.calls.append(("inpaint", prompt, image))
        return image


def _run(key, **kw):
    client = FakeClient()
    t = TRANSFORMATIONS[key]
    args = {"prompt": "a red balloon"}
    if t.needs_image:
        args["image"] = Image.new("RGB", (8, 8), (200, 30, 30))
    if t.needs_mask:
        args["mask"] = Image.new("L", (8, 8), 255)
    args.update(kw)
    t.run(client, **args)
    assert len(client.calls) == 1
    return client.calls[0]


def test_registry_keys_match_and_flags_match_signatures():
    for key, t in TRANSFORMATIONS.items():
        assert t.key == key
        params = inspect.signature(t.run).parameters
        assert ("image" in params) == t.needs_image, key
        assert ("mask" in params) == t.needs_mask, key


def test_every_transformation_routes_to_expected_pipeline():
    for key, t in TRANSFORMATIONS.items():
        pipeline, prompt, _ = _run(key)
        expected = "text2img" if not t.needs_image else "inpaint" if t.needs_mask else "img2img"
        assert pipeline == expected, key
        assert prompt.strip(), key


def test_user_prompt_and_style_reach_the_model_prompt():
    assert "a red balloon" in _run("insert")[1]
    assert "Monet" in _run("style", style="Monet")[1]


def test_colorize_sends_grayscale_rgb():
    _, _, sent = _run("colorize")
    arr = np.array(sent)
    assert sent.mode == "RGB"
    assert (arr[..., 0] == arr[..., 1]).all() and (arr[..., 1] == arr[..., 2]).all()


def test_grabcut_degenerate_rect_returns_empty_full_size_mask():
    img = Image.new("RGB", (20, 10))
    m = grabcut_mask(img, (5, 5, 6, 6))
    assert m.size == (20, 10) and m.mode == "L"
    assert np.array(m).max() == 0


def test_threshold_mask_is_binary():
    arr = np.zeros((10, 10, 3), np.uint8)
    arr[:, 5:] = 220
    m = np.array(threshold_mask(Image.fromarray(arr)))
    assert set(np.unique(m)) == {0, 255}


def test_feather_mask_accepts_even_radius_and_keeps_size():
    mask = Image.new("L", (16, 16), 0)
    mask.paste(255, (4, 4, 12, 12))
    out = feather_mask(mask, radius=4)
    vals = np.unique(np.array(out))
    assert out.size == (16, 16)
    assert len(vals) > 2  # edges got softened
