import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image


def vlm_predict(image_path, model_id="google/medgemma-4b-it"):
    # 1. Chargement du processeur et du modèle (c'est ici qu'ils vont !)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 2. Préparation de l'image
    image = Image.open(image_path).convert("RGB")

    # 3. Préparation du prompt avec le template spécial
    messages = [
        {"role": "user",
         "content": "<image>\nDécris cette radiographie médicale de manière précise et professionnelle."}
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # 4. Envoi au modèle
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    # 5. Génération
    outputs = model.generate(**inputs, max_new_tokens=512)

    # 6. Décodage
    result = processor.decode(outputs[0], skip_special_tokens=True)

    return {
        "predicted_class": result,
        "model_used": model_id,
        "status": "success"
    }