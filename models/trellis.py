"""TRELLIS backend — high-quality structured 3D with textured meshes, ~16GB VRAM."""

from pathlib import Path

from models import register
from models.base import BaseModel, GenerationConfig, GenerationResult

MODEL_ID = "JeffreyXiang/TRELLIS-image-large"


@register
class TrellisModel(BaseModel):
    name = "trellis"
    description = "TRELLIS — high quality, textured GLB, structured latent representation (~16 GB)"
    url = "https://github.com/microsoft/TRELLIS"
    min_vram_gb = 16
    supports_texture = True
    supports_pbr = False

    def __init__(self):
        self._pipeline = None

    def is_available(self) -> bool:
        try:
            from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self, low_vram: bool = False, log=None):
        if log is None:
            from models.base import _noop_log
            log = _noop_log
        from trellis.pipelines import TrellisImageTo3DPipeline

        log("Loading TRELLIS weights...")
        self._pipeline = TrellisImageTo3DPipeline.from_pretrained(MODEL_ID)
        log("Moving model to GPU...")
        self._pipeline.cuda()
        log("TRELLIS ready")

    def generate(self, image_path: Path, output_path: Path,
                 config: GenerationConfig) -> GenerationResult:
        if self._pipeline is None:
            self.load(low_vram=config.low_vram)

        def _run(img_path, out_path, cfg):
            from PIL import Image
            from trellis.utils import postprocessing_utils

            image = Image.open(img_path).convert("RGB")

            outputs = self._pipeline.run(image, seed=cfg.seed or 0)

            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=0.95,
                texture_size=cfg.texture_resolution,
            )
            glb.export(str(out_path))

        return self._timed_generate(_run, image_path, output_path, config)

    def unload(self):
        self._pipeline = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
