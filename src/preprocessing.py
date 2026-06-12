from __future__ import annotations

from pathlib import Path
from PIL import Image

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

def load_image(path: str | Path, size: tuple[int, int] = (512, 512)) -> Image.Image:
    path = Path(path)
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported image format: {path.suffix}")
    img = Image.open(path).convert("RGB")
    return img.resize(size)

def basic_quality_flag(path: str | Path) -> str:
    name = Path(path).name.lower()
    if "uncertain" in name or "limited" in name:
        return "limited"
    return "good"