from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from PIL import Image

from .preprocessing import basic_quality_flag

WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."


def toy_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    start = time.perf_counter()
    name = Path(image_path).name.lower()
    quality = basic_quality_flag(image_path)

    if "suspected_opacity" in name:
        pred = "suspected_opacity"
        conf = 0.78 if mode == "baseline" else 0.72
        evidence = ["synthetic opacity-like area visible in the lung field"]
        justification = "The synthetic image contains a localized brighter region compatible with the toy opacity class. This is a pipeline validation result, not a medical interpretation."
    elif "normal" in name:
        pred = "normal"
        conf = 0.72 if mode == "baseline" else 0.68
        evidence = ["no synthetic opacity marker detected"]
        justification = "The synthetic image does not contain the opacity marker used by the toy generator. This conclusion is limited to the synthetic validation setting."
    else:
        pred = "uncertain"
        conf = 0.52
        evidence = ["limited synthetic image quality"]
        justification = "The image is treated as limited quality in the toy catalog. The safe output is uncertainty rather than a forced class."

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


def vlm_predict(image_path: str | Path, model_id: str = "google/medgemma-4b-it") -> dict[str, Any]:
    start = time.perf_counter()

    # imports ici pour ne pas casser les tests sans dependances VLM
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": "<image>\nDécris cette radiographie médicale de manière prudente et concise.",
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    result = processor.decode(outputs[0], skip_special_tokens=True)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "image_quality": basic_quality_flag(image_path),
        "predicted_class": "uncertain",
        "confidence": 0.5,
        "visual_evidence": [result],
        "justification": "The VLM output is kept as supporting text and the class stays uncertain until it is parsed and validated.",
        "limitations": ["raw VLM output", "no clinical context", "not a validated medical model"],
        "warning": WARNING,
        "model_name": model_id,
        "prompt_version": "vlm_v1",
        "latency_ms": latency_ms,
    }
