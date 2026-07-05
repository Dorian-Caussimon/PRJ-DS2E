from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

from .preprocessing import basic_quality_flag, looks_like_xray

WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

ALLOWED_CLASSES = {"normal", "suspected_opacity", "uncertain"}
ALLOWED_QUALITY = {"good", "limited", "poor"}
REQUIRED_KEYS = {
    "image_quality",
    "predicted_class",
    "confidence",
    "visual_evidence",
    "justification",
    "limitations",
    "warning",
}

# ---------------------------------------------------------------------------
# MedGemma — configuration
#
# Modèle : google/medgemma-4b-it  (vision-language, instruction-tuned)
# Accès  : Hugging Face Inference API (serverless ou dédié)
# Clé    : variable d'environnement HF_TOKEN  (jamais dans le code)
#
# Pour utiliser ce fichier :
#   export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
#   pip install huggingface_hub Pillow
#
# Accès au modèle :
#   Le modèle MedGemma est soumis à des conditions d'usage spécifiques Google.
#   Acceptez les conditions sur https://huggingface.co/google/medgemma-4b-it
#   avant d'utiliser ce code, et citez la model card dans votre rapport.
# ---------------------------------------------------------------------------
_HF_MODEL = os.getenv("MEDGEMMA_MODEL", "google/medgemma-4b-it")


# ---------------------------------------------------------------------------
# Chargeur de prompts : prompts/prompt_{mode}_v1.txt  (source de vérité)
# ---------------------------------------------------------------------------
_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

_FALLBACK_PROMPT = (
    "You are an educational radiology assistant. Not a clinician.\n"
    "Analyze the frontal chest X-ray. Return ONLY valid JSON, no markdown, no extra text.\n"
    '{"image_quality":"good|limited|poor","predicted_class":"normal|suspected_opacity|uncertain",'
    '"confidence":0.0,"visual_evidence":["observation"],"justification":"short reasoning",'
    '"limitations":["factor"],"warning":"Educational prototype only. Not for diagnosis."}\n'
    "Rules: no diagnosis language; if confidence<0.60 use uncertain; if quality poor use uncertain."
)


def _load_prompt(mode: str) -> str:
    """Charge prompts/prompt_{mode}_v1.txt.
    Fallback inline si le fichier est absent pour ne jamais bloquer le pipeline.
    """
    prompt_candidates = [
        _PROMPTS_DIR / f"prompt_{mode}_v1.txt",
        _PROMPTS_DIR / f"{mode}_prompt.txt",
    ]
    for prompt_file in prompt_candidates:
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")
    import warnings
    warnings.warn(
        f"[inference] Prompt file not found for mode={mode}. Using inline fallback.",
        stacklevel=3,
    )
    return _FALLBACK_PROMPT


# ---------------------------------------------------------------------------
# Appel MedGemma via Hugging Face Inference API
# ---------------------------------------------------------------------------
def _encode_image_b64(image_path: Path) -> tuple[str, str]:
    """Encode l'image en base64 et retourne (données, media_type)."""
    suffix = image_path.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(suffix, "image/jpeg")
    data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    return data, media_type


# Cache du pipeline MedGemma (chargé une seule fois, réutilisé ensuite)
# @st.cache_resource si Streamlit est disponible : garantit la persistance
# du modèle en mémoire entre les reruns du script (chaque interaction UI
# relance streamlit_app.py, sans ce cache le modèle serait rechargé à
# chaque upload d'image).
try:
    import streamlit as _st
    _cache_resource = _st.cache_resource
except ImportError:
    def _cache_resource(func):
        return func

_medgemma_pipeline = None


@_cache_resource
def _get_pipeline():
    """Charge MedGemma localement via transformers (une seule fois).

    Utilise le GPU si disponible, sinon CPU (lent mais fonctionnel).
    Mis en cache via st.cache_resource (ou une variable globale en
    dehors de Streamlit) pour éviter de le recharger à chaque prédiction.

    Prérequis :
        pip install transformers torch accelerate Pillow
        huggingface-cli login   (token avec accès à google/medgemma-4b-it)
    """
    global _medgemma_pipeline
    if _medgemma_pipeline is not None:
        return _medgemma_pipeline

    import sys
    _log = lambda msg: print(f"[VLM] {msg}", file=sys.stderr, flush=True)

    try:
        if not os.getenv("HF_TOKEN"):
            _log("HF_TOKEN absent : MedGemma non charge, repli uncertain.")
            return None

        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        device_label = "GPU (cuda)" if device == 0 else "CPU (lent, ~30-60 s/image)"
        torch_dtype = torch.bfloat16 if device == 0 else torch.float32
        _log(f"⏳ Chargement MedGemma sur {device_label}…")
        _log("   (premier chargement ~1-5 min selon connexion et machine)")

        _medgemma_pipeline = pipeline(
            "image-text-to-text",
            model=_HF_MODEL,
            device=device,
            torch_dtype=torch_dtype,
            token=os.environ["HF_TOKEN"],
        )
        _log("✅ MedGemma chargé en mémoire")
        return _medgemma_pipeline

    except ImportError as exc:
        _log(f"❌ Dépendance manquante : {exc}")
        _log("   Corrigez avec : pip install transformers torch accelerate")
        return None
    except Exception as exc:
        _log(f"❌ Chargement MedGemma échoué : {type(exc).__name__}: {exc}")
        return None


def _call_vlm_backend(image_path: Path, prompt: str) -> str | None:
    """Appelle MedGemma en local via transformers.

    Les logs [VLM] apparaissent dans le terminal Streamlit (stderr).
    Retourne le texte brut de la réponse, ou None en cas d'échec.
    """
    import sys
    from PIL import Image as PILImage

    _log = lambda msg: print(f"[VLM] {msg}", file=sys.stderr, flush=True)
    _log(f"→ image={image_path.name}  modèle={_HF_MODEL}")

    # ÉTAPE 1 : charger le pipeline (mis en cache après le 1er appel)
    pipe = _get_pipeline()
    if pipe is None:
        return None

    # ÉTAPE 2 : ouvrir l'image
    try:
        image = PILImage.open(image_path).convert("RGB")
        _log(f"✅ image ouverte ({image.size[0]}×{image.size[1]} px)")
        # Borne la résolution pour éviter une explosion du nombre de tokens
        # visuels (et des temps d'inférence de plusieurs minutes) sur des
        # images bien plus grandes que ce que le modèle attend.
        max_side = 896
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), PILImage.LANCZOS)
            _log(f"↘️ image redimensionnée ({image.size[0]}×{image.size[1]} px)")
    except Exception as exc:
        _log(f"❌ ouverture image échouée : {exc}")
        return None

    # ÉTAPE 3 : inférence MedGemma
    try:
        _log("⏳ Inférence MedGemma en cours…")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]
        outputs = pipe(text=messages, max_new_tokens=1000)
        # Le pipeline retourne une liste ; on extrait le texte généré
        raw = outputs[0]["generated_text"][-1]["content"]
        _log(f"✅ Réponse reçue ({len(raw)} caractères)")
        _log(f"   Aperçu : {raw[:120].replace(chr(10), ' ')}")
        return raw

    except Exception as exc:
        _log(f"❌ Inférence échouée : {type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Helpers : validation et nettoyage de la réponse du modèle
# ---------------------------------------------------------------------------
def _extract_json_block(text: str) -> str:
    """Retire les balises markdown et extrait le premier objet JSON du texte."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start: end + 1]
    return cleaned.strip()


def _validate_schema(data: Any) -> bool:
    """Vérifie strictement que la réponse respecte le schéma attendu."""
    if not isinstance(data, dict):
        return False
    if not REQUIRED_KEYS.issubset(data.keys()):
        return False
    if data.get("image_quality") not in ALLOWED_QUALITY:
        return False
    if data.get("predicted_class") not in ALLOWED_CLASSES:
        return False
    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return False
    if not (0.0 <= float(confidence) <= 1.0):
        return False
    if not isinstance(data.get("visual_evidence"), list) or not all(
        isinstance(i, str) for i in data["visual_evidence"]
    ):
        return False
    if not isinstance(data.get("justification"), str):
        return False
    if not isinstance(data.get("limitations"), list) or not all(
        isinstance(i, str) for i in data["limitations"]
    ):
        return False
    if not isinstance(data.get("warning"), str):
        return False
    return True


def _safe_uncertain_fallback(
    reason: str,
    quality: str = "limited",
    model_name: str = "vlm-fallback",
    prompt_version: str = "unknown",
) -> dict[str, Any]:
    """Retourne toujours 'uncertain' quand MedGemma échoue ou renvoie un schéma invalide."""
    return {
        "image_quality": quality if quality in ALLOWED_QUALITY else "limited",
        "predicted_class": "uncertain",
        "confidence": 0.5,
        "visual_evidence": ["analyse automatique indisponible ou non fiable"],
        "justification": f"Repli de sécurité : {reason}.",
        "limitations": [
            "échec ou réponse invalide du modèle VLM",
            "aucune interprétation clinique possible",
        ],
        "warning": WARNING,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "latency_ms": 0,
    }


# ---------------------------------------------------------------------------
# Fonction publique principale
# ---------------------------------------------------------------------------
def vlm_predict(
    image_path: str | Path,
    mode: str = "baseline",
    prompt: str | None = None,
) -> dict[str, Any]:
    """Analyse une radiographie via MedGemma avec le prompt du mode choisi.

    Flux :
      1. Charge prompts/prompt_{mode}_v1.txt  (baseline ou improved)
      2. Encode l'image en base64 et l'envoie avec le prompt à MedGemma
      3. Parse et valide le JSON retourné par le modèle
      4. En cas d'échec à n'importe quelle étape → retombe sur uncertain

    Variables d'environnement requises :
      HF_TOKEN  — token Hugging Face avec accès à google/medgemma-4b-it

    Le champ 'warning' est toujours réécrit par le code,
    jamais laissé tel que renvoyé par le modèle.
    """
    start = time.perf_counter()
    image_path = Path(image_path)
    quality = basic_quality_flag(image_path)
    prompt_version = f"{mode}_v1"
    used_prompt = prompt if prompt is not None else _load_prompt(mode)

    # --- Garde-fou technique (non-IA) : rejette les images qui ne
    # ressemblent pas à une radiographie avant même d'appeler MedGemma,
    # car le modèle peut halluciner un résultat plausible sur n'importe
    # quelle image plutôt que de refuser de répondre. ---
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(image_path) as _img:
            is_xray = looks_like_xray(_img.convert("RGB"))
    except Exception:
        is_xray = True  # ne bloque pas sur une erreur de lecture ; MedGemma gérera l'échec
    if not is_xray:
        return _safe_uncertain_fallback(
            reason="image non reconnue comme une radiographie thoracique (contrôle technique pré-modèle)",
            quality=quality,
            model_name=f"medgemma-{mode}",
            prompt_version=prompt_version,
        )

    # --- Appel MedGemma ---
    try:
        raw_output = _call_vlm_backend(image_path, used_prompt)
    except Exception:
        raw_output = None

    if raw_output is None:
        return _safe_uncertain_fallback(
            reason="appel MedGemma échoué (token absent, modèle inaccessible ou timeout)",
            quality=quality,
            model_name=f"medgemma-{mode}",
            prompt_version=prompt_version,
        )

    # --- Parsing JSON ---
    try:
        cleaned = _extract_json_block(raw_output)
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return _safe_uncertain_fallback(
            reason="réponse MedGemma non parsable en JSON",
            quality=quality,
            model_name=f"medgemma-{mode}",
            prompt_version=prompt_version,
        )

    # --- Validation schéma ---
    if not _validate_schema(data):
        return _safe_uncertain_fallback(
            reason="schéma JSON invalide ou valeur hors liste autorisée",
            quality=quality,
            model_name=f"medgemma-{mode}",
            prompt_version=prompt_version,
        )

    # --- Post-traitement ---
    data["confidence"] = round(float(data["confidence"]), 3)
    data["warning"] = WARNING                            # toujours réécrit par le code
    data["model_name"] = f"medgemma-4b-it-{mode}"       # traçable dans la DB
    data["prompt_version"] = prompt_version              # baseline_v1 ou improved_v1
    data["latency_ms"] = int((time.perf_counter() - start) * 1000)
    return data


# ---------------------------------------------------------------------------
# Rétrocompatibilité
# ---------------------------------------------------------------------------
def toy_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Backend local deterministe pour les smoke tests et la demo sans modele VLM."""
    start = time.perf_counter()
    image_path = Path(image_path)
    name = image_path.name.lower()
    quality = basic_quality_flag(image_path)

    if "suspected_opacity" in name or "opacity" in name:
        predicted_class = "suspected_opacity"
        confidence = 0.78 if mode == "baseline" else 0.86
        visual_evidence = ["synthetic opacity marker detected in filename"]
        justification = "Toy backend uses the synthetic case naming convention for smoke tests."
    elif "normal" in name:
        predicted_class = "normal"
        confidence = 0.76 if mode == "baseline" else 0.84
        visual_evidence = ["synthetic normal marker detected in filename"]
        justification = "Toy backend uses the synthetic case naming convention for smoke tests."
    else:
        predicted_class = "uncertain"
        confidence = 0.5
        visual_evidence = ["no reliable synthetic class marker detected"]
        justification = "Toy backend cannot infer a reliable class from this filename."

    if quality == "limited":
        predicted_class = "uncertain"
        confidence = min(confidence, 0.5)

    return {
        "image_quality": quality,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 3),
        "visual_evidence": visual_evidence,
        "justification": justification,
        "limitations": [
            "toy backend based on synthetic filenames only",
            "not a validated medical model",
        ],
        "warning": WARNING,
        "model_name": f"toy-{mode}",
        "prompt_version": f"{mode}_toy",
        "latency_ms": int((time.perf_counter() - start) * 1000),
    }


def vlm_predict_placeholder(image_path: str | Path, prompt: str) -> dict[str, Any]:
    """Conservé pour compatibilité ascendante."""
    return vlm_predict(image_path, mode="baseline", prompt=prompt)
