"""
Hugging Face client using the official huggingface_hub InferenceClient.
Auto-routes to whichever provider serves each model through the new
router.huggingface.co endpoint. In-memory SHA-256 cache for fast repeats.
"""
from __future__ import annotations

import hashlib
import io
import os
from typing import Optional

from PIL import Image
from huggingface_hub import InferenceClient

MODELS = {
    "text2img": "black-forest-labs/FLUX.1-schnell",
    "img2img":  "stabilityai/stable-diffusion-xl-refiner-1.0",
    "inpaint":  "stabilityai/stable-diffusion-2-inpainting",
}


class HFClient:
    def __init__(self, token: Optional[str] = None, timeout: int = 120):
        self.token = token or os.getenv("HF_API_TOKEN", "")
        if not self.token:
            raise RuntimeError("HF_API_TOKEN not set. Add it to your .env file.")
        self.timeout = timeout
        self._cache: dict[str, bytes] = {}
        self._clients: dict[str, InferenceClient] = {}

    def _client_for(self, model: str) -> InferenceClient:
        if model not in self._clients:
            self._clients[model] = InferenceClient(model=model, token=self.token, timeout=self.timeout)
        return self._clients[model]

    def _key(self, *parts) -> str:
        h = hashlib.sha256()
        for p in parts:
            if isinstance(p, Image.Image):
                buf = io.BytesIO(); p.save(buf, format="PNG"); h.update(buf.getvalue())
            elif isinstance(p, (bytes, bytearray)):
                h.update(p)
            else:
                h.update(str(p).encode())
        return h.hexdigest()

    @staticmethod
    def _img_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

    @staticmethod
    def _to_pil(obj) -> Image.Image:
        if isinstance(obj, Image.Image):
            return obj.convert("RGB")
        if isinstance(obj, (bytes, bytearray)):
            return Image.open(io.BytesIO(obj)).convert("RGB")
        raise RuntimeError(f"Unexpected response type: {type(obj)}")

    def text_to_image(self, prompt, negative_prompt="", steps=30, guidance=7.5,
                      width=768, height=768, seed=None):
        key = self._key("t2i", MODELS["text2img"], prompt, negative_prompt, steps, guidance, width, height, seed)
        if key in self._cache:
            return self._to_pil(self._cache[key])
        client = self._client_for(MODELS["text2img"])
        kwargs = dict(prompt=prompt, num_inference_steps=steps,
                      guidance_scale=guidance, width=width, height=height)
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if seed is not None:
            kwargs["seed"] = seed
        try:
            img = client.text_to_image(**kwargs)
        except TypeError:
            img = client.text_to_image(prompt=prompt)
        img = self._to_pil(img)
        self._cache[key] = self._img_bytes(img)
        return img

    def image_to_image(self, image, prompt, strength=0.6, guidance=7.5, steps=30):
        key = self._key("i2i", MODELS["img2img"], image, prompt, strength, guidance, steps)
        if key in self._cache:
            return self._to_pil(self._cache[key])
        client = self._client_for(MODELS["img2img"])
        try:
            img = client.image_to_image(
                image=image, prompt=prompt,
                num_inference_steps=steps, guidance_scale=guidance, strength=strength,
            )
            img = self._to_pil(img)
        except Exception:
            img = self.text_to_image(prompt, steps=steps, guidance=guidance)
        self._cache[key] = self._img_bytes(img)
        return img

    def inpaint(self, image, mask, prompt, steps=30, guidance=7.5):
        key = self._key("inpaint", MODELS["inpaint"], image, mask, prompt, steps, guidance)
        if key in self._cache:
            return self._to_pil(self._cache[key])
        client = self._client_for(MODELS["inpaint"])
        try:
            img = client.image_to_image(
                image=image, prompt=prompt, mask_image=mask.convert("L"),
                num_inference_steps=steps, guidance_scale=guidance,
            )
            img = self._to_pil(img)
        except Exception:
            img = self.text_to_image(prompt, steps=steps, guidance=guidance)
        self._cache[key] = self._img_bytes(img)
        return img
