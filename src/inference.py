from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any

from .preprocessing import basic_quality_flag

WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

ALLOWED_CLASSES = {"normal", "suspected_opacity", "uncertain"}
ALLOWED_QUALITY = {"good", "limited"}
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
# Prompt amélioré (voir suggestions dans la réponse de chat)
# ---------------------------------------------------------------------------
VLM_PROMPT_TEMPLATE = """Vous êtes un assistant radiologue virtuel strictement pédagogique.
Vous travaillez exclusivement sur des images synthétiques de validation de pipeline.
Votre rôle est d'analyser cette radiographie thoracique frontale et de produire UNIQUEMENT
un objet JSON valide, sans aucun texte avant ou après, et sans balises markdown (pas de ```).

Schéma obligatoire (n'ajoutez aucune clé supplémentaire) :
{
  "image_quality": "good" ou "limited" (flou, mauvais cadrage, artefact, sur/sous-exposition => limited),
  "predicted_class": "normal" OU "suspected_opacity" OU "uncertain" (exactement ces chaînes, aucune autre langue/casse),
  "confidence": nombre décimal entre 0.0 et 1.0,
  "visual_evidence": ["signes visuels observés ou absence d'anomalie, décrits factuellement"],
  "justification": "une phrase courte reliant les signes visuels à la classe prédite",
  "limitations": ["facteurs limitant cette analyse"],
  "warning": "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."
}

Règles strictes :
- Ne devinez jamais. Si l'image est de mauvaise qualité, ambiguë, ou si le signe visuel n'est pas clair, utilisez "uncertain".
- La confiance doit rester inférieure ou égale à 0.6 si image_quality est "limited" ou si le signe est ambigu.
- N'inventez aucun diagnostic en dehors d'une suspicion d'opacité : pas de nom de pathologie précis, pas de stade, pas de localisation anatomique fine non observable.
- Ne proposez aucune conduite à tenir, traitement, examen complémentaire ou pronostic.
- Si l'image ne ressemble pas à une radiographie thoracique frontale, retournez tout de même "uncertain" avec image_quality "limited" et expliquez-le dans la justification, sans analyser une autre modalité.

Exemple de format de sortie attendu (cas incertain) :
{"image_quality": "limited", "predicted_class": "uncertain", "confidence": 0.5, "visual_evidence": ["qualité d'image insuffisante pour conclure"], "justification": "L'image ne permet pas de distinguer un signe fiable.", "limitations": ["qualité d'image limitée"], "warning": "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."}

Répondez maintenant UNIQUEMENT avec le JSON, rien avant, rien après.
"""


def toy_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Deterministic toy predictor used to validate the repo pipeline.
    It reads synthetic labels from filenames. This is not medical inference.
    """
    start = time.perf_counter()
    name = Path(image_path).name.lower()
    quality = basic_quality_flag(image_path)

    if "suspected_opacity" in name:
        pred = "suspected_opacity"
        conf = 0.78 if mode == "baseline" else 0.72
        evidence = ["synthetic opacity-like area visible in the lung field"]
        justification = (
            "The synthetic image contains a localized brighter region compatible with the "
            "toy opacity class. This is a pipeline validation result, not a medical interpretation."
        )
    elif "normal" in name:
        pred = "normal"
        conf = 0.72 if mode == "baseline" else 0.68
        evidence = ["no synthetic opacity marker detected"]
        justification = (
            "The synthetic image does not contain the opacity marker used by the toy generator. "
            "This conclusion is limited to the synthetic validation setting."
        )
    else:
        pred = "uncertain"
        conf = 0.52
        evidence = ["limited synthetic image quality"]
        justification = (
            "The image is treated as limited quality in the toy catalog. "
            "The safe output is uncertainty rather than a forced class."
        )

    # Improved mode is more conservative.
    if mode == "improved" and quality != "good":
        pred = "uncertain"
        conf = min(conf, 0.55)

    latency_ms = int((time.perf_counter() - start) * 1000)

    return {
        "image_quality": quality,
        "predicted_class": pred,
        "confidence": round(float(conf), 3),
        "visual_evidence": evidence,
        "justification": justification,
        "limitations": ["synthetic toy image", "no clinical context", "not a validated medical model"],
        "warning": WARNING,
        "model_name": f"toy-rule-{mode}",
        "prompt_version": f"{mode}_v1",
        "latency_ms": latency_ms,
    }


def _safe_uncertain_fallback(reason: str, quality: str = "limited", model_name: str = "vlm-fallback") -> dict[str, Any]:
    """Repli de sécurité utilisé chaque fois que le VLM échoue, renvoie un texte
    non parsable, ou viole le schéma attendu. Toujours 'uncertain'."""
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
        "prompt_version": "vlm_v1",
        "latency_ms": 0,
    }


def _validate_schema(data: Any) -> bool:
    """Vérifie strictement que la réponse du modèle respecte le schéma attendu."""
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
        isinstance(item, str) for item in data["visual_evidence"]
    ):
        return False
    if not isinstance(data.get("justification"), str):
        return False
    if not isinstance(data.get("limitations"), list) or not all(
        isinstance(item, str) for item in data["limitations"]
    ):
        return False
    if not isinstance(data.get("warning"), str):
        return False

    return True


def _extract_json_block(text: str) -> str:
    """Nettoie une réponse de VLM qui pourrait contenir des balises markdown
    ou du texte autour du JSON, avant de tenter le parsing."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned.strip()


def _call_vlm_backend(image_path: Path, prompt: str) -> str | None:
    """Point d'extension pour un vrai appel au VLM (MedGemma / Gemma multimodal
    via l'API d'inférence Hugging Face, ou un pipeline transformers local).

    Dans cet environnement pédagogique, aucun backend n'est câblé par défaut :
    la fonction renvoie None, ce qui fait retomber `vlm_predict` sur le
    prédicteur déterministe `toy_predict`, afin de garder le pipeline testable
    sans dépendre d'un modèle ni d'une clé d'API.

    Pour les étudiants : remplacez le corps de cette fonction par un vrai appel,
    par exemple via `huggingface_hub.InferenceClient.chat_completion(...)` ou
    `transformers.pipeline("image-text-to-text", model="google/medgemma-4b-it")`,
    en passant `image_path` et `prompt`, et en renvoyant le texte brut de la
    réponse du modèle (à parser ensuite en JSON par `vlm_predict`).
    """
    return None


def vlm_predict(image_path: str | Path, mode: str = "baseline", prompt: str | None = None) -> dict[str, Any]:
    """Prédiction via un VLM (MedGemma / Gemma), avec repli de sécurité strict.

    Garanties :
    - Le schéma de sortie reste identique à `toy_predict`.
    - En l'absence de backend configuré, ou en cas de réponse non parsable /
      hors schéma, le résultat retombe sur `uncertain` plutôt que d'inventer
      une classe.
    - Le champ `warning` n'est jamais celui renvoyé par le modèle : il est
      toujours réécrit par le code, pour ne jamais dépendre du LLM sur ce point.
    """
    start = time.perf_counter()
    image_path = Path(image_path)
    quality = basic_quality_flag(image_path)
    used_prompt = prompt or VLM_PROMPT_TEMPLATE

    try:
        raw_output = _call_vlm_backend(image_path, used_prompt)
    except Exception:
        raw_output = None

    if raw_output is None:
        # Aucun backend VLM configuré : on garde un comportement déterministe
        # et testable en repassant par le prédicteur toy.
        result = toy_predict(image_path, mode=mode)
        result["model_name"] = f"vlm-toy-fallback-{mode}"
        result["prompt_version"] = "vlm_v1"
        return result

    try:
        cleaned = _extract_json_block(raw_output)
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return _safe_uncertain_fallback("réponse du modèle non parsable en JSON", quality)

    if not _validate_schema(data):
        return _safe_uncertain_fallback("schéma JSON invalide ou champ hors valeurs autorisées", quality)

    data["confidence"] = round(float(data["confidence"]), 3)
    data["warning"] = WARNING  # ne jamais faire confiance au modèle sur ce champ
    data.setdefault("model_name", f"vlm-{mode}")
    data.setdefault("prompt_version", "vlm_v1")
    data["latency_ms"] = int((time.perf_counter() - start) * 1000)
    return data