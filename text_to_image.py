"""Text-to-image generation using FLUX or Stable Diffusion XL."""

import time
from pathlib import Path
from typing import Optional

_pipeline = None
_pipeline_name = None


MODELS = {
    "flux-schnell": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "description": "FLUX.1 Schnell — fast 4-step generation",
        "pipeline_class": "FluxPipeline",
        "default_steps": 4,
        "default_guidance": 0.0,
        "min_vram_gb": 12,
    },
    "flux-dev": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "description": "FLUX.1 Dev — high quality, 20-50 steps",
        "pipeline_class": "FluxPipeline",
        "default_steps": 28,
        "default_guidance": 3.5,
        "min_vram_gb": 24,
    },
    "sdxl": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "description": "Stable Diffusion XL — reliable, well-supported",
        "pipeline_class": "StableDiffusionXLPipeline",
        "default_steps": 30,
        "default_guidance": 7.5,
        "min_vram_gb": 8,
    },
    "z-image": {
        "repo": "Tongyi-MAI/Z-Image-Turbo",
        "description": "Z-Image Turbo — 6B param DiT, fast and high quality",
        "pipeline_class": "ZImagePipeline",
        "default_steps": 4,
        "default_guidance": 0.0,
        "min_vram_gb": 16,
    },
    "playground": {
        "repo": "playgroundai/playground-v2.5-1024px-aesthetic",
        "description": "Playground v2.5 — aesthetic tuned, great for objects",
        "pipeline_class": "StableDiffusionXLPipeline",
        "default_steps": 30,
        "default_guidance": 3.0,
        "min_vram_gb": 8,
    },
}


def available_models() -> list[dict]:
    results = []
    for name, info in MODELS.items():
        results.append({
            "name": name,
            "description": info["description"],
            "default_steps": info["default_steps"],
            "default_guidance": info["default_guidance"],
            "min_vram_gb": info["min_vram_gb"],
        })
    return results


def is_available() -> bool:
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def load(model_name: str = "playground", low_vram: bool = False):
    global _pipeline, _pipeline_name

    if _pipeline is not None and _pipeline_name == model_name:
        return

    unload()

    import torch
    from diffusers import FluxPipeline, StableDiffusionXLPipeline, DiffusionPipeline

    info = MODELS[model_name]
    dtype = torch.float16

    pipeline_classes = {
        "FluxPipeline": FluxPipeline,
        "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
    }

    cls_name = info["pipeline_class"]
    if cls_name in pipeline_classes:
        pipe = pipeline_classes[cls_name].from_pretrained(info["repo"], torch_dtype=dtype)
    else:
        pipe = DiffusionPipeline.from_pretrained(
            info["repo"], torch_dtype=dtype, trust_remote_code=True
        )

    if low_vram:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    _pipeline = pipe
    _pipeline_name = model_name


def generate(
    prompt: str,
    output_path: Path,
    model_name: str = "playground",
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
    width: int = 1024,
    height: int = 1024,
    low_vram: bool = False,
) -> dict:
    """Generate an image from a text prompt. Returns metadata dict."""
    import torch

    load(model_name, low_vram=low_vram)

    info = MODELS[model_name]
    num_steps = steps or info["default_steps"]
    guidance = guidance_scale if guidance_scale is not None else info["default_guidance"]

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    t0 = time.time()

    kwargs = {
        "prompt": prompt,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height,
        "generator": generator,
    }

    image = _pipeline(**kwargs).images[0]
    elapsed = time.time() - t0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    return {
        "output_path": str(output_path),
        "model": model_name,
        "prompt": prompt,
        "steps": num_steps,
        "guidance_scale": guidance,
        "seed": seed,
        "width": width,
        "height": height,
        "elapsed_seconds": elapsed,
    }


def unload():
    global _pipeline, _pipeline_name
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        _pipeline_name = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
