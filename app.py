"""
Generative Image Editing Tool — Streamlit UI
"""
from __future__ import annotations

import io
import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw

from hf_client import HFClient
from transformations import TRANSFORMATIONS
from cv_utils import grabcut_mask, threshold_mask, face_mask, feather_mask

load_dotenv()

st.set_page_config(
    page_title="Gen Image Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp {
        background: radial-gradient(circle at 20% 0%, #1e1b4b 0%, #0f172a 40%, #020617 100%);
      }
      h1, h2, h3 { color: #f8fafc !important; letter-spacing: -0.01em; }
      .gradient-title {
        background: linear-gradient(90deg, #a855f7 0%, #ec4899 50%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.6rem;
        margin-bottom: 0.2rem;
      }
      .subtitle { color: #94a3b8; font-size: 1rem; margin-bottom: 1.5rem; }
      .stButton > button {
        background: linear-gradient(90deg, #8b5cf6 0%, #ec4899 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.4rem;
        font-weight: 600;
        font-size: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.35);
      }
      .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 28px rgba(236, 72, 153, 0.45);
      }
      .stDownloadButton > button {
        background: rgba(148, 163, 184, 0.15);
        color: #f8fafc;
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 10px;
      }
      .stTextArea textarea, .stTextInput input {
        background: rgba(15, 23, 42, 0.6) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
      }
      section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
      }
      .empty-canvas {
        border: 2px dashed rgba(148, 163, 184, 0.3);
        border-radius: 18px;
        padding: 4rem 1rem;
        text-align: center;
        color: #64748b;
      }
      #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state():
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("result_label", None)
    st.session_state.setdefault("history", [])

_init_state()


with st.sidebar:
    st.markdown("### 🎨 Gen Image Studio")
    st.caption("Stable Diffusion · 17 AI transformations")
    st.divider()

    st.markdown("**Choose a tool**")
    transform_keys = list(TRANSFORMATIONS.keys())
    transform_labels = [TRANSFORMATIONS[k].label for k in transform_keys]
    chosen_idx = st.selectbox(
        "Transformation",
        range(len(transform_keys)),
        format_func=lambda i: transform_labels[i],
        label_visibility="collapsed",
    )
    chosen_key = transform_keys[chosen_idx]
    xform = TRANSFORMATIONS[chosen_key]
    st.caption(xform.description)

    st.divider()
    st.markdown("**Tips**")
    st.caption(
        "• Be descriptive in your prompt\n\n"
        "• Try styles like *'cinematic lighting'*\n\n"
        "• For edits, upload an image first\n\n"
        "• For inpainting, draw a mask"
    )


st.markdown('<div class="gradient-title">Generative Image Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Create and edit images with natural language · powered by Stable Diffusion</div>',
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("#### ✏️ Create")

    image: Optional[Image.Image] = None
    mask: Optional[Image.Image] = None

    if xform.needs_image:
        upload = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )
        if upload:
            image = Image.open(upload).convert("RGB")
            image.thumbnail((1024, 1024))
            st.image(image, caption="Source image", width="stretch")

    if xform.needs_mask and image is not None:
        with st.expander("🎯 Smart mask (OpenCV)", expanded=True):
            mode = st.radio(
                "Mask mode",
                ["Box", "GrabCut", "Otsu", "Face detect"],
                horizontal=True,
                label_visibility="collapsed",
            )
            w, h = image.size
            c1, c2 = st.columns(2)
            with c1:
                x0 = st.slider("x0", 0, w, w // 4)
                y0 = st.slider("y0", 0, h, h // 4)
            with c2:
                x1 = st.slider("x1", 0, w, 3 * w // 4)
                y1 = st.slider("y1", 0, h, 3 * h // 4)
            if mode == "Box":
                mask = Image.new("L", (w, h), 0)
                ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], fill=255)
            elif mode == "GrabCut":
                mask = grabcut_mask(image, (x0, y0, x1, y1))
            elif mode == "Otsu":
                mask = threshold_mask(image)
            else:
                mask = face_mask(image)
            feather = st.slider("Feather edges", 0, 31, 9, step=2)
            if feather > 0:
                mask = feather_mask(mask, radius=feather)
            st.image(mask, caption="Mask preview", width="stretch")

    prompt = st.text_area(
        "Prompt",
        placeholder="A cozy cabin under the northern lights, cinematic lighting, 4k…",
        height=110,
    )

    with st.expander("⚙️ Advanced settings"):
        style_name = st.text_input("Artistic style (for style transfer)", "Van Gogh")
        c1, c2 = st.columns(2)
        with c1:
            steps = st.slider("Quality (steps)", 10, 60, 30)
        with c2:
            guidance = st.slider("Prompt adherence", 1.0, 15.0, 7.5)
        seed_str = st.text_input("Seed (optional)", "", placeholder="random")
        seed = int(seed_str) if seed_str.strip().isdigit() else None

    generate = st.button("✨  Generate", type="primary", use_container_width=True)

with col_right:
    st.markdown("#### 🖼️ Result")
    if st.session_state.result is not None:
        st.image(
            st.session_state.result,
            caption=st.session_state.result_label,
            width="stretch",
        )
        buf = io.BytesIO()
        st.session_state.result.save(buf, format="PNG")
        st.download_button(
            "⬇️  Download PNG",
            buf.getvalue(),
            file_name=f"{chosen_key}.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.markdown(
            '<div class="empty-canvas">🪄<br><br>'
            'Your generated image will appear here.<br>'
            'Pick a tool, write a prompt, and click <b>Generate</b>.</div>',
            unsafe_allow_html=True,
        )


if generate:
    token = os.getenv("HF_API_TOKEN", "")
    if not token:
        st.error("No Hugging Face token found. Add `HF_API_TOKEN=hf_...` to your `.env` file.")
        st.stop()
    if xform.needs_image and image is None:
        st.warning("Please upload an image for this tool.")
        st.stop()
    if xform.needs_mask and mask is None:
        st.warning("Please create a mask for this tool.")
        st.stop()
    if not xform.needs_image and not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    try:
        client = HFClient(token=token)
        with st.spinner(f"Generating with {xform.label}… this can take 10–30 seconds"):
            kwargs = dict(prompt=prompt, style=style_name, steps=steps, guidance=guidance, seed=seed)
            if xform.needs_image:
                kwargs["image"] = image
            if xform.needs_mask:
                kwargs["mask"] = mask
            result = xform.run(client, **{k: v for k, v in kwargs.items() if v is not None})
        st.session_state.result = result
        st.session_state.result_label = xform.label
        st.session_state.history.insert(0, (xform.label, result))
        st.rerun()
    except Exception as e:
        st.error(f"Generation failed: {e}")


if st.session_state.history:
    st.divider()
    st.markdown("#### 🕘 Recent creations")
    cols = st.columns(4)
    for i, (label, img) in enumerate(st.session_state.history[:8]):
        with cols[i % 4]:
            st.image(img, caption=label, width="stretch")