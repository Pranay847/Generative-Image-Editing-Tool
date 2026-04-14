"""
15+ natural-language transformations built on top of HFClient.

Each transformation accepts a base image (optional for text-to-image) and a
user prompt or parameters, then constructs an optimized Stable Diffusion
prompt and routes to the right pipeline (text2img / img2img / inpaint).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PIL import Image, ImageDraw

from hf_client import HFClient


@dataclass
class Transformation:
    key: str
    label: str
    description: str
    needs_image: bool
    needs_mask: bool
    run: Callable


# ---------- prompt builders ----------
def _style(prompt: str, style: str) -> str:
    return f"{prompt}, in the style of {style}, highly detailed, masterpiece, 8k"


def _quality_suffix(p: str) -> str:
    return f"{p}, ultra detailed, sharp focus, professional photography, 8k"


# ---------- transformation implementations ----------
def t_text_to_image(client: HFClient, prompt: str, **kw) -> Image.Image:
    return client.text_to_image(_quality_suffix(prompt))


def t_style_transfer(client: HFClient, image: Image.Image, prompt: str, style: str = "Van Gogh", **kw) -> Image.Image:
    return client.image_to_image(image, _style(prompt or "this scene", style), strength=0.7)


def t_object_insertion(client: HFClient, image: Image.Image, mask: Image.Image, prompt: str, **kw) -> Image.Image:
    return client.inpaint(image, mask, f"add {prompt}, photorealistic, seamless lighting")


def t_object_removal(client: HFClient, image: Image.Image, mask: Image.Image, **kw) -> Image.Image:
    return client.inpaint(image, mask, "empty background, clean, seamless, photorealistic")


def t_background_replace(client: HFClient, image: Image.Image, mask: Image.Image, prompt: str, **kw) -> Image.Image:
    return client.inpaint(image, mask, f"background of {prompt}, cinematic lighting")


def t_colorize(client: HFClient, image: Image.Image, **kw) -> Image.Image:
    return client.image_to_image(image.convert("L").convert("RGB"), "vivid color photograph, natural colors", strength=0.5)


def t_cartoonify(client: HFClient, image: Image.Image, **kw) -> Image.Image:
    return client.image_to_image(image, "Pixar 3D cartoon style, vibrant, smooth shading", strength=0.65)


def t_anime(client: HFClient, image: Image.Image, **kw) -> Image.Image:
    return client.image_to_image(image, "anime style, Studio Ghibli, cel shaded, vibrant", strength=0.7)


def t_oil_painting(client: HFClient, image: Image.Image, **kw) -> Image.Image:
    return client.image_to_image(image, "classical oil painting, thick brush strokes, canvas texture", strength=0.6)


def t_pencil_sketch(client: HFClient, image: Image.Image, **kw) -> Image.Image:
    return client.image_to_image(image, "detailed pencil sketch, graphite, hatching, monochrome", strength=0.7)


def t_pixel_art(client: HFClient, image: Image.Image, **kw) -> Image.Image:
    return client.image_to_image(image, "8-bit pixel art, retro game style, limited palette", strength=0.7)


def t_watercolor(client: HFClient, image: Image.Image, **kw) -> Image.Image:
    return client.image_to_image(image, "watercolor painting, soft washes, paper texture", strength=0.6)


def t_cyberpunk(client: HFClient, image: Image.Image, **kw) -> Image.Image:
    return client.image_to_image(image, "cyberpunk neon cityscape, blade runner, rain, holograms", strength=0.65)


def t_relight(client: HFClient, image: Image.Image, prompt: str = "golden hour", **kw) -> Image.Image:
    return client.image_to_image(image, f"relit with {prompt} lighting, dramatic shadows", strength=0.45)


def t_season_swap(client: HFClient, image: Image.Image, prompt: str = "winter snow", **kw) -> Image.Image:
    return client.image_to_image(image, f"transformed to {prompt}, photorealistic", strength=0.55)


def t_age_progression(client: HFClient, image: Image.Image, prompt: str = "older", **kw) -> Image.Image:
    return client.image_to_image(image, f"portrait, made {prompt}, realistic skin, detailed", strength=0.5)


def t_upscale_enhance(client: HFClient, image: Image.Image, **kw) -> Image.Image:
    return client.image_to_image(image, "ultra high resolution, sharp, denoised, enhanced details", strength=0.3)


# ---------- registry ----------
TRANSFORMATIONS: dict[str, Transformation] = {
    t.key: t for t in [
        Transformation("text2img", "Text → Image", "Generate a brand-new image from a prompt.", False, False, t_text_to_image),
        Transformation("style", "Style Transfer", "Re-render image in an artistic style.", True, False, t_style_transfer),
        Transformation("insert", "Object Insertion", "Mask a region and add an object.", True, True, t_object_insertion),
        Transformation("remove", "Object Removal", "Mask a region and erase its contents.", True, True, t_object_removal),
        Transformation("bg_replace", "Background Replace", "Mask the background and swap it.", True, True, t_background_replace),
        Transformation("colorize", "Colorize B&W", "Add color to a grayscale photo.", True, False, t_colorize),
        Transformation("cartoonify", "Cartoonify", "Convert into a 3D cartoon look.", True, False, t_cartoonify),
        Transformation("anime", "Anime Style", "Anime / Ghibli rendering.", True, False, t_anime),
        Transformation("oil", "Oil Painting", "Classical oil painting.", True, False, t_oil_painting),
        Transformation("sketch", "Pencil Sketch", "Detailed graphite sketch.", True, False, t_pencil_sketch),
        Transformation("pixel", "Pixel Art", "Retro 8-bit pixel art.", True, False, t_pixel_art),
        Transformation("watercolor", "Watercolor", "Soft watercolor painting.", True, False, t_watercolor),
        Transformation("cyberpunk", "Cyberpunk", "Neon cyberpunk reimagining.", True, False, t_cyberpunk),
        Transformation("relight", "Relighting", "Change lighting (golden hour, studio, etc.).", True, False, t_relight),
        Transformation("season", "Season Swap", "Change the season of a scene.", True, False, t_season_swap),
        Transformation("age", "Age Progression", "Make a portrait look older / younger.", True, False, t_age_progression),
        Transformation("upscale", "Enhance / Upscale", "Sharpen and refine details.", True, False, t_upscale_enhance),
    ]
}
