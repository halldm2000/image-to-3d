"""Image preprocessing for 3D generation: background removal, cleanup, padding."""

from pathlib import Path
from PIL import Image
import io
import numpy as np


def remove_background(image: Image.Image) -> Image.Image:
    """Remove background using rembg, returning RGBA image with transparent bg."""
    from rembg import remove
    output = remove(image)
    return output.convert("RGBA")


def center_and_pad(image: Image.Image, size: int = 512, padding_ratio: float = 0.85) -> Image.Image:
    """Center the subject and pad to a square with consistent margins."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    alpha = np.array(image)[:, :, 3]
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)

    if not rows.any() or not cols.any():
        return image.resize((size, size), Image.LANCZOS)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    cropped = image.crop((x_min, y_min, x_max + 1, y_max + 1))

    max_dim = max(cropped.size)
    target_dim = int(size * padding_ratio)
    scale = target_dim / max_dim

    new_w = int(cropped.size[0] * scale)
    new_h = int(cropped.size[1] * scale)
    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    canvas.paste(resized, (x_offset, y_offset), resized)

    return canvas


def add_white_background(image: Image.Image) -> Image.Image:
    """Replace transparent background with white."""
    if image.mode != "RGBA":
        return image
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    bg.paste(image, (0, 0), image)
    return bg.convert("RGB")


def preprocess(image_path: Path, output_path: Path = None,
               size: int = 512, skip_bg_removal: bool = False) -> Path:
    """Full preprocessing pipeline: bg removal → center/pad → save.

    Returns path to the preprocessed image.
    """
    image = Image.open(image_path).convert("RGBA")

    if not skip_bg_removal:
        image = remove_background(image)

    image = center_and_pad(image, size=size)

    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_preprocessed.png"

    image.save(str(output_path))
    return output_path
