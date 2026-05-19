"""Base class for image-to-3D model backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import time

_noop_log = lambda msg: None


@dataclass
class GenerationResult:
    output_path: Path
    model_name: str
    elapsed_seconds: float
    vertex_count: Optional[int] = None
    face_count: Optional[int] = None
    has_texture: bool = False
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        parts = [
            f"Model: {self.model_name}",
            f"Time: {self.elapsed_seconds:.1f}s",
            f"Output: {self.output_path}",
        ]
        if self.vertex_count:
            parts.append(f"Vertices: {self.vertex_count:,}")
        if self.face_count:
            parts.append(f"Faces: {self.face_count:,}")
        parts.append(f"Textured: {'yes' if self.has_texture else 'no'}")
        return " | ".join(parts)


@dataclass
class GenerationConfig:
    steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    octree_resolution: int = 256
    texture: bool = True
    texture_resolution: int = 512
    texture_views: int = 6
    low_vram: bool = False
    remove_background: bool = True


class BaseModel(ABC):
    """Abstract base for all image-to-3D model backends."""

    name: str = "base"
    description: str = ""
    url: str = ""
    min_vram_gb: float = 0
    supports_texture: bool = False
    supports_pbr: bool = False

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this model's dependencies are installed and weights are accessible."""

    def is_loaded(self) -> bool:
        """Return True if model weights are currently in memory."""
        return False

    @abstractmethod
    def load(self, low_vram: bool = False, log: Callable = _noop_log):
        """Load model weights into memory."""

    @abstractmethod
    def generate(self, image_path: Path, output_path: Path,
                 config: GenerationConfig, log: Callable = _noop_log) -> GenerationResult:
        """Run inference: image → 3D mesh file."""

    def unload(self):
        """Free GPU memory. Override if cleanup is needed."""
        pass

    def _count_mesh_stats(self, path: Path) -> tuple[Optional[int], Optional[int]]:
        """Try to read vertex/face counts from a mesh file."""
        try:
            import trimesh
            mesh = trimesh.load(str(path), force="mesh")
            return len(mesh.vertices), len(mesh.faces)
        except Exception:
            return None, None

    def _timed_generate(self, fn, image_path: Path, output_path: Path,
                        config: GenerationConfig) -> GenerationResult:
        """Wrap a generation function with timing and stats."""
        t0 = time.time()
        fn(image_path, output_path, config)
        elapsed = time.time() - t0

        verts, faces = self._count_mesh_stats(output_path)
        return GenerationResult(
            output_path=output_path,
            model_name=self.name,
            elapsed_seconds=elapsed,
            vertex_count=verts,
            face_count=faces,
            has_texture=self.supports_texture and config.texture,
            metadata={
                "steps": config.steps,
                "guidance_scale": config.guidance_scale,
                "seed": config.seed,
            },
        )
