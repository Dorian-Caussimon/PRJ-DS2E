from src.inference import vlm_predict


def test_medgemma_baseline_uncertainty():
    """Vérifie que la baseline VLM retourne bien la classe 'uncertain' par défaut."""

    # Données simulées pour le contexte
    patient_name = "James"
    chemin_image = "data/sample_images/CXR_SYN_002_suspected_opacity.png"

    # Exécution de ta fonction
    resultat = vlm_predict(chemin_image)

    import json
    print("\n🚀 RÉSULTAT DU MODÈLE :")
    print(json.dumps(resultat, indent=2, ensure_ascii=False))

    # Assertions (ce que pytest va vérifier)
    assert resultat is not None
    assert resultat["predicted_class"] == "uncertain", "La classe par défaut doit être uncertain !"
    assert "warning" in resultat, "Le warning médical obligatoire est manquant."