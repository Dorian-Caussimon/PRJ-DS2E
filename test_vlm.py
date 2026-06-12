from huggingface_hub import login
import json

# 1. On se connecte à Hugging Face directement ici !
#Rajouter votre token ici !
#login()

# 2. On importe notre fonction
from src.inference import vlm_predict

print("🤖 Démarrage du test MedGemma...")
print("⏳ (Le téléchargement va commencer, laissez tourner...)")

# Assurez-vous que ce chemin pointe bien vers une image de votre dossier
chemin_image = "data/sample_images/CXR_SYN_002_suspected_opacity.png"

try:
    resultat = vlm_predict(chemin_image)

    print("\n✅ --- RÉSULTAT DU MODÈLE ---")
    print(json.dumps(resultat, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"\n❌ Erreur lors du test : {e}")