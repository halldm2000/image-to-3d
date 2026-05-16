"""Model registry for image-to-3D backends."""

from models.base import BaseModel, GenerationConfig, GenerationResult

_REGISTRY: dict[str, type[BaseModel]] = {}


def register(cls: type[BaseModel]) -> type[BaseModel]:
    _REGISTRY[cls.name] = cls
    return cls


def get_model(name: str) -> BaseModel:
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _REGISTRY[name]()


def list_models() -> list[dict]:
    results = []
    for name, cls in _REGISTRY.items():
        m = cls()
        results.append({
            "name": name,
            "description": cls.description,
            "available": m.is_available(),
            "min_vram_gb": cls.min_vram_gb,
            "supports_texture": cls.supports_texture,
            "supports_pbr": cls.supports_pbr,
            "url": cls.url,
        })
    return results


def available_models() -> list[str]:
    return [name for name, cls in _REGISTRY.items() if cls().is_available()]


# Import backends to trigger registration
from models import hunyuan3d  # noqa: F401, E402
from models import triposr  # noqa: F401, E402
from models import trellis  # noqa: F401, E402
from models import spar3d  # noqa: F401, E402
from models import triposg  # noqa: F401, E402
