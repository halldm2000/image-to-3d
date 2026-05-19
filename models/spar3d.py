"""SPAR3D backend — fast (~0.7s), UV-textured meshes, successor to SF3D, ~10GB VRAM."""

from pathlib import Path

from models import register
from models.base import BaseModel, GenerationConfig, GenerationResult

MODEL_ID = "stabilityai/stable-point-aware-3d"


@register
class SPAR3DModel(BaseModel):
    name = "spar3d"
    description = "SPAR3D — fast (0.7s), UV-textured output, point-aware geometry (~10 GB)"
    url = "https://github.com/Stability-AI/stable-point-aware-3d"
    min_vram_gb = 7
    supports_texture = True
    supports_pbr = False

    def __init__(self):
        self._model = None

    def is_available(self) -> bool:
        try:
            from spar3d.system import SPAR3D  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self, low_vram: bool = False, log=None):
        if log is None:
            from models.base import _noop_log
            log = _noop_log
        import os
        from spar3d.system import SPAR3D

        if low_vram:
            os.environ["SPAR3D_LOW_VRAM"] = "1"

        log("Loading SPAR3D weights...")
        self._model = SPAR3D.from_pretrained(
            MODEL_ID,
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        log("Moving model to GPU...")
        self._model.to("cuda")
        self._model.eval()
        log("SPAR3D ready")

    def generate(self, image_path: Path, output_path: Path,
                 config: GenerationConfig) -> GenerationResult:
        if self._model is None:
            self.load(low_vram=config.low_vram)

        def _run(img_path, out_path, cfg):
            import torch
            from PIL import Image

            image = Image.open(img_path).convert("RGB")

            with torch.no_grad():
                mesh = self._model.run_image(
                    image,
                    bake_resolution=cfg.texture_resolution,
                )

            mesh.export(str(out_path))

        return self._timed_generate(_run, image_path, output_path, config)

    def unload(self):
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
