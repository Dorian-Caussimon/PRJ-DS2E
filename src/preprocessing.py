from __future__ import annotations

from pathlib import Path
from PIL import Image

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

# Une radiographie thoracique est quasi monochrome (niveaux de gris).
# Au-delà de ce seuil de saturation moyenne (0-255), l'image est très
# probablement une photo couleur, une capture d'écran ou un dessin.
MAX_XRAY_SATURATION = 18

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

def looks_like_xray(image: Image.Image) -> bool:
    """Heuristique non-IA : une radio thoracique est quasi monochrome.

    Rejette grossièrement les photos couleur, captures d'écran ou dessins,
    qui ont une saturation bien supérieure à celle d'un vrai radiogramme.
    """
    _, saturation, _ = image.convert("HSV").split()
    avg_saturation = sum(saturation.getdata()) / (image.width * image.height)
    return avg_saturation <= MAX_XRAY_SATURATION