from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Journal de prédictions :
# - une ligne console lisible et alignée pour le suivi en direct (uvicorn / streamlit)
# - une trace JSONL append-only pour l'analyse d'erreurs / le registre de 20-30 cas
#   demandé dans l'appel d'offre.
#
# NB : si le dépôt possède déjà un module de métriques/SQLite (cf. README,
# "src/ # ... métriques, SQLite"), branchez-le ici en complément de
# (ou à la place de) l'écriture JSONL ci-dessous, pour ne pas dupliquer la
# logique d'évaluation déjà en place dans eval/run_evaluation.py.
# ---------------------------------------------------------------------------

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
JSONL_PATH = LOG_DIR / "predictions.jsonl"

_CLASS_TAG = {
    "normal": "NORMAL      ",
    "suspected_opacity": "SUSPECTED_OP",
    "uncertain": "UNCERTAIN   ",
}


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("assistant_radio")
    if logger.handlers:  # éviter les doublons de handlers au rechargement (uvicorn --reload)
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = _build_logger()


def log_prediction(
    *,
    request_id: str,
    source_name: str,
    mode: str,
    backend: str,
    result: dict[str, Any],
) -> None:
    """Journalise une prédiction de façon lisible (console) et traçable (JSONL)."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tag = _CLASS_TAG.get(result.get("predicted_class", ""), "UNKNOWN     ")
    confidence = result.get("confidence", 0.0)
    quality = str(result.get("image_quality", "")).ljust(7)
    latency = result.get("latency_ms", 0)

    line = (
        f"[{timestamp}] req={request_id} class={tag} "
        f"conf={confidence:.2f} quality={quality} "
        f"mode={mode:<8} backend={backend:<4} latency={latency}ms "
        f"file={source_name}"
    )
    logger.info(line)

    record = {
        "timestamp": timestamp,
        "request_id": request_id,
        "source_name": source_name,
        "mode": mode,
        "backend": backend,
        **result,
    }
    with JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
