from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.inference import WARNING, toy_predict, vlm_predict
from src.guardrails import apply_safety_guardrails
from api.logging_utils import log_prediction

app = FastAPI(
    title="Assistant radiologue virtuel — API de démonstration",
    description=(
        "Prototype pédagogique d'IA médicale multimodale. "
        "Non destiné au diagnostic. Validation par un professionnel qualifié requise."
    ),
    version="0.1.0",
)


class PredictionResponse(BaseModel):
    """Schéma de sortie — identique à celui affiché par l'app Streamlit."""

    image_quality: Literal["good", "limited"]
    predicted_class: Literal["normal", "suspected_opacity", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    visual_evidence: list[str]
    justification: str
    limitations: list[str]
    warning: str
    model_name: str
    prompt_version: str
    latency_ms: int
    request_id: str


@app.get("/")
def root() -> dict:
    """Point d'entrée informatif : rappelle le statut non clinique du prototype."""
    return {
        "name": "Assistant radiologue virtuel — API de démonstration",
        "status": "ok",
        "warning": WARNING,
        "endpoints": {
            "predict": "POST /predict (multipart/form-data : file, mode, backend)",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "scope": "educational prototype, not diagnosis"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Radiographie thoracique frontale (png/jpg)."),
    mode: Literal["baseline", "improved"] = Form("baseline"),
    backend: Literal["toy", "vlm"] = Form("toy"),
) -> PredictionResponse:
    """Analyse une radiographie et renvoie le même schéma JSON que l'app Streamlit.

    Exemple :
        curl -X POST "http://127.0.0.1:8000/predict" \\
             -F "file=@data/sample_images/CXR_SYN_002_suspected_opacity.png"
    """
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Format d'image non supporté (png/jpg uniquement).")

    request_id = uuid.uuid4().hex[:8]

    # On garde le nom de fichier original : toy_predict() lit la classe
    # synthétique dans ce nom (cf. data/sample_images/*.png). Un nom aléatoire
    # ferait toujours retomber la prédiction sur "uncertain".
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    tmp_path.write_bytes(await file.read())

    predict_fn = vlm_predict if backend == "vlm" else toy_predict
    result = apply_safety_guardrails(predict_fn(tmp_path, mode=mode))

    log_prediction(
        request_id=request_id,
        source_name=file.filename,
        mode=mode,
        backend=backend,
        result=result,
    )

    return PredictionResponse(**result, request_id=request_id)
