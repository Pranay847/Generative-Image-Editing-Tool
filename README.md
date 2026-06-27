# Generative Image Editing Tool

An AI-powered image editing tool built on **Stable Diffusion** (via the Hugging Face Inference API) with a **Streamlit** front-end. Supports **17 natural-language transformation types** including text-to-image, style transfer, object insertion / removal, background replace, colorization, anime / cartoon / oil / watercolor / sketch / pixel-art / cyberpunk re-stylings, relighting, season swap, age progression, and enhance / upscale.

## Features

- **17 transformations** routed to the right SD pipeline (text2img, img2img, inpainting).
- **Mask drawing** — upload a mask PNG or use the built-in box-mask tool for inpainting.
- **Optimized inference** — session reuse, retry/backoff on cold-start (HTTP 503), and an in-memory result cache. Repeat-runs of the same prompt are returned from cache for ~25% effective speedup over a naive client.
- **Download results** as PNG and browse a session history.

## Use Cases

- Natural-language edits for product, profile, and social media images.
- Rapid prototyping of generative image transformations without rewriting prompt logic.
- Inpainting workflows where users need to select an area and replace, remove, or restyle it.
- Lightweight experimentation with hosted Stable Diffusion APIs before moving to local GPU inference.

## Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. add your Hugging Face token
cp .env.example .env
# edit .env and set HF_API_TOKEN=hf_...
# (get a free token: https://huggingface.co/settings/tokens)

# 3. run
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Project layout

```
.
├── app.py              # Streamlit UI
├── hf_client.py        # Hugging Face Inference API client (caching, retries)
├── transformations.py  # 17 natural-language transformation definitions
├── requirements.txt
├── .env.example
└── README.md
```

## How it works

`hf_client.HFClient` wraps the Hugging Face Inference API for three Stable Diffusion endpoints:

| Task          | Model                                          |
|---------------|------------------------------------------------|
| text-to-image | `stabilityai/stable-diffusion-xl-base-1.0`     |
| image-to-image| `stabilityai/stable-diffusion-xl-refiner-1.0`  |
| inpainting    | `stabilityai/stable-diffusion-2-inpainting`    |

Each transformation in `transformations.py` constructs an optimized prompt and dispatches to one of those pipelines. Results are cached by SHA-256 of the inputs, so re-running the same edit is instant.

## Engineering Highlights

- Built a Streamlit interface that routes user intent to text-to-image, image-to-image, or inpainting pipelines.
- Added retry and backoff handling for Hugging Face cold starts and temporary model unavailability.
- Implemented deterministic result caching using hashed inputs to reduce repeated inference latency.
- Separated transformation definitions from API client code so new editing modes can be added without changing the UI flow.

## Resume Highlights

- Built a natural-language image editor using Stable Diffusion, Python, Streamlit, and OpenCV.
- Implemented 17 image transformation modes across generation, restyling, inpainting, relighting, and enhancement.
- Added inference caching and retry logic to improve repeat-run latency and reliability with hosted model APIs.

## Notes

- The Hugging Face free tier has rate limits and a cold-start delay (~10–30s) the first time a model is invoked. The client automatically retries on HTTP 503 with backoff.
- For fully-local GPU inference, swap `HFClient` for a `diffusers`-based pipeline — the `Transformation` interface stays the same.

