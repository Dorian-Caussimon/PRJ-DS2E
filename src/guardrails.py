from __future__ import annotations

from typing import Any

ALLOWED_CLASSES = {"normal", "suspected_opacity", "uncertain"}
REQUIRED_KEYS = {"image_quality", "predicted_class", "confidence", "visual_evidence", "justification", "limitations", "warning"}
WARNING_TEXT = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."
NON_CLINICAL_LIMITATION = "not a validated medical model"


def validate_prediction(pred: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    missing = REQUIRED_KEYS - set(pred)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if pred.get("predicted_class") not in ALLOWED_CLASSES:
        errors.append("invalid predicted_class")
    try:
        conf = float(pred.get("confidence", -1))
        if not 0 <= conf <= 1:
            errors.append("confidence outside [0,1]")
    except Exception:
        errors.append("confidence is not numeric")
    if not pred.get("warning"):
        errors.append("warning missing")
    return not errors, errors


def apply_safety_guardrails(pred: dict[str, Any]) -> dict[str, Any]:
    valid, errors = validate_prediction(pred)
    if not valid:
        pred["predicted_class"] = "uncertain"
        pred["confidence"] = min(float(pred.get("confidence", 0.0) or 0.0), 0.5)
        pred.setdefault("limitations", []).append("guardrail triggered: invalid output schema")
    pred.setdefault("limitations", [])
    if NON_CLINICAL_LIMITATION not in pred["limitations"]:
        pred["limitations"].append(NON_CLINICAL_LIMITATION)
    if pred.get("image_quality") in {"limited", "poor"} and float(pred.get("confidence", 0)) < 0.6:
        pred["predicted_class"] = "uncertain"
    if any("not recognized as a chest x-ray" in str(item).lower() for item in pred["limitations"]):
        pred["predicted_class"] = "uncertain"
        pred["confidence"] = min(float(pred.get("confidence", 0.0) or 0.0), 0.0)
    pred["warning"] = WARNING_TEXT
    pred["guardrail_errors"] = errors
    return pred
