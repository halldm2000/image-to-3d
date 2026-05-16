"""Hunyuan3D-2.1 backend — highest quality, PBR textures, ~29GB VRAM."""

import sys
from pathlib import Path

from models import register
from models.base import BaseModel, GenerationConfig, GenerationResult

REPO_DIR = Path(__file__).parent.parent / "Hunyuan3D-2.1"
MODEL_ID = "tencent/Hunyuan3D-2.1"


@register
class Hunyuan3DModel(BaseModel):
    name = "hunyuan3d"
    description = "Hunyuan3D-2.1 — highest quality, PBR textures (albedo/metallic/roughness)"
    url = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1"
    min_vram_gb = 10
    supports_texture = True
    supports_pbr = True

    def __init__(self):
        self._shape_pipeline = None
        self._paint_pipeline = None

    def is_available(self) -> bool:
        return (REPO_DIR / "hy3dshape").is_dir()

    def load(self, low_vram: bool = False):
        self._ensure_paths()
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        kwargs = {}
        if low_vram:
            kwargs["device"] = "cpu"

        self._shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_ID, subfolder="hunyuan3d-dit-v2-1", **kwargs
        )
        if low_vram:
            self._shape_pipeline.to("cuda")

        self._low_vram = low_vram

    def generate(self, image_path: Path, output_path: Path,
                 config: GenerationConfig) -> GenerationResult:
        self._ensure_paths()
        if self._shape_pipeline is None:
            self.load(low_vram=config.low_vram)

        def _run(img_path, out_path, cfg):
            import torch

            gen_kwargs = {
                "image": str(img_path),
                "num_inference_steps": cfg.steps,
                "guidance_scale": cfg.guidance_scale,
                "octree_resolution": cfg.octree_resolution,
            }
            if cfg.seed is not None:
                gen_kwargs["generator"] = torch.Generator(device="cuda").manual_seed(cfg.seed)

            mesh = self._shape_pipeline(**gen_kwargs)[0]

            if not cfg.texture:
                mesh.export(str(out_path))
                return

            untextured = out_path.with_suffix(".untextured.glb")
            mesh.export(str(untextured))

            if self._low_vram:
                del self._shape_pipeline
                self._shape_pipeline = None
                torch.cuda.empty_cache()

            from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
            paint_config = Hunyuan3DPaintConfig(
                max_num_view=cfg.texture_views,
                resolution=cfg.texture_resolution,
            )
            paint_pipeline = Hunyuan3DPaintPipeline(paint_config)
            textured_mesh = paint_pipeline(str(untextured), image_path=str(img_path))
            textured_mesh.export(str(out_path))

        return self._timed_generate(_run, image_path, output_path, config)

    def unload(self):
        import torch
        self._shape_pipeline = None
        self._paint_pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_paths(self):
        shape_path = str(REPO_DIR / "hy3dshape")
        paint_path = str(REPO_DIR / "hy3dpaint")
        if shape_path not in sys.path:
            sys.path.insert(0, shape_path)
        if paint_path not in sys.path:
            sys.path.insert(0, paint_path)
