"""TripoSR backend — fast single-image 3D, ~4GB VRAM, no texture."""

from pathlib import Path

from models import register
from models.base import BaseModel, GenerationConfig, GenerationResult

MODEL_ID = "stabilityai/TripoSR"


@register
class TripoSRModel(BaseModel):
    name = "triposr"
    description = "TripoSR — fast (~5s), low VRAM (~4GB), geometry only (vertex colors)"
    url = "https://github.com/VAST-AI-Research/TripoSR"
    min_vram_gb = 4
    supports_texture = False
    supports_pbr = False

    def __init__(self):
        self._model = None

    def is_available(self) -> bool:
        try:
            import tsr  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self, low_vram: bool = False, log=None):
        if log is None:
            from models.base import _noop_log
            log = _noop_log
        from tsr.system import TSR

        log("Loading TripoSR weights...")
        self._model = TSR.from_pretrained(
            MODEL_ID,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self._model.renderer.set_chunk_size(8192 if not low_vram else 2048)
        log("Moving model to GPU...")
        self._model.to("cuda")
        log("TripoSR ready")

    def generate(self, image_path: Path, output_path: Path,
                 config: GenerationConfig, log=None) -> GenerationResult:
        if self._model is None:
            self.load(low_vram=config.low_vram)

        def _run(img_path, out_path, cfg):
            from PIL import Image
            import numpy as np

            image = Image.open(img_path).convert("RGB")

            with __import__("torch").no_grad():
                scene_codes = self._model([image], device="cuda")

            mesh = self._model.extract_mesh(
                scene_codes,
                resolution=cfg.octree_resolution,
            )[0]

            if str(out_path).endswith(".glb"):
                mesh.export(str(out_path))
            elif str(out_path).endswith(".obj"):
                mesh.export(str(out_path))
            else:
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
