"""TripoSG backend — high-quality geometry, optional texture baking, ~8GB VRAM."""

from pathlib import Path

from models import register
from models.base import BaseModel, GenerationConfig, GenerationResult

MODEL_ID = "VAST-AI/TripoSG"


@register
class TripoSGModel(BaseModel):
    name = "triposg"
    description = "TripoSG — excellent geometry, optional texture bake, 1.5B params (~8 GB)"
    url = "https://github.com/VAST-AI-Research/TripoSG"
    min_vram_gb = 6
    supports_texture = True
    supports_pbr = False

    def __init__(self):
        self._pipeline = None

    def is_available(self) -> bool:
        try:
            from triposg.pipelines import TripoSGPipeline  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self, low_vram: bool = False):
        from triposg.pipelines import TripoSGPipeline

        self._pipeline = TripoSGPipeline.from_pretrained(MODEL_ID)
        self._pipeline.to("cuda")

    def generate(self, image_path: Path, output_path: Path,
                 config: GenerationConfig) -> GenerationResult:
        if self._pipeline is None:
            self.load(low_vram=config.low_vram)

        def _run(img_path, out_path, cfg):
            from PIL import Image

            image = Image.open(img_path).convert("RGB")

            mesh = self._pipeline(
                image,
                mc_resolution=cfg.octree_resolution,
                bake_texture=cfg.texture,
                texture_resolution=cfg.texture_resolution,
            )

            mesh.export(str(out_path))

        return self._timed_generate(_run, image_path, output_path, config)

    def unload(self):
        self._pipeline = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
