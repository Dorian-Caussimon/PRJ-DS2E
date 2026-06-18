import streamlit as st
from PIL import Image
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="Radiologue IA", page_icon="🩺", layout="wide")

# --- INJECTION DE CSS PERSONNALISÉ (La magie opère ici) ---
st.markdown("""
<style>
    /* Cacher le menu Streamlit et le footer pour faire plus "Application" */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Titre principal stylisé avec dégradé */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        padding-top: 1rem;
    }
    .sub-title {
        text-align: center;
        color: #a0aab5;
        font-size: 1.2rem;
        margin-bottom: 40px;
        font-style: italic;
    }

    /* Bannière éducative stylisée */
    .edu-warning {
        background-color: rgba(255, 75, 75, 0.15);
        border-left: 5px solid #ff4b4b;
        padding: 15px 20px;
        border-radius: 5px;
        color: #ffcccc;
        margin-bottom: 20px;
        font-size: 1.1rem;
    }

    /* Style pour les images */
    img {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- EN-TÊTE STYLISÉ ---
st.markdown(
    '<div class="edu-warning">⚠️ <b>PROTOTYPE ÉDUCATIF</b> : Ce système n\'a aucune valeur de diagnostic clinique. Les résultats doivent être validés par un professionnel de santé.</div>',
    unsafe_allow_html=True)
st.markdown('<h1 class="main-title">🩺 Assistant Radiologue Virtuel</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Prototype d\'IA médicale multimodale — MasterCamp DAI EFREI</p>',
            unsafe_allow_html=True)

# --- ONGLETS ---
tab1, tab2, tab3, tab4 = st.tabs(["📁 Cas Patient", "🔍 Analyse IA", "🧠 Apprentissage", "📊 Suivi & Logs"])

# --- ONGLET 1 : CAS ---
with tab1:
    st.write("### 📥 Importer une radiographie")
    st.write("Veuillez charger une radiographie thoracique frontale (format JPG ou PNG) pour débuter l'analyse.")

    st.write("")  # Espacement

    # On utilise des colonnes pour centrer l'uploader de fichier pour un look plus épuré
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state['current_image'] = image

            st.markdown("<br>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.success("✅ Image chargée avec succès dans le pipeline. Passez à l'onglet **🔍 Analyse IA**.")

# --- ONGLET 2 : ANALYSE ---
with tab2:
    if 'current_image' not in st.session_state:
        st.info("ℹ️ En attente d'une image. Veuillez charger une radiographie dans l'onglet **📁 Cas Patient**.")
    else:
        # Layout avec la radio à gauche et les résultats à droite
        col_img, spacer, col_res = st.columns([1, 0.1, 1.5])

        with col_img:
            st.write("#### Image originale")
            st.image(st.session_state['current_image'], use_container_width=True)
            st.write("")
            btn = st.button("🚀 Lancer l'inférence (MedGemma 4B)", type="primary", use_container_width=True)

        with col_res:
            st.write("#### Rapport d'analyse généré")
            if btn:
                with st.spinner("🧠 Raisonnement IA en cours (Simulation)..."):
                    time.sleep(2)

                    # Faux JSON basé sur les exigences du document (Section 3.5)
                    mock_json = {
                        "predicted_class": "suspicion d'opacité",
                        "confidence": 0.82,
                        "visual_evidence": "Opacité linéaire basale gauche. Perte de volume et rétraction des structures.",
                        "justification": "La perte de volume lobaire inférieur gauche associée à la rétraction des structures est fortement évocatrice d'une atélectasie basale probable.",
                        "warning": "Résultat incertain, corrélation clinique et contrôle radiologique nécessaires."
                    }

                    # Affichage façon Dashboard
                    st.markdown(f"### 🎯 Conclusion IA : **{mock_json['predicted_class'].upper()}**")

                    # Jauge de confiance
                    conf = int(mock_json['confidence'] * 100)
                    st.progress(mock_json['confidence'], text=f"Indice de confiance : {conf}%")

                    st.divider()

                    # Boîtes d'information stylisées
                    st.info(f"**👁️ Preuves visuelles :**\n\n{mock_json['visual_evidence']}")
                    st.success(f"**🧠 Raisonnement clinique :**\n\n{mock_json['justification']}")
                    st.error(f"**⚠️ Garde-fous (Avertissement) :**\n\n{mock_json['warning']}")

# --- ONGLET 3 & 4 (Placeholders propres) ---
with tab3:
    st.write("### 📈 Comparaison des modèles")
    st.info(
        "Cet onglet accueillera les comparaisons entre la baseline et le prompt amélioré, prévues en S3 de la roadmap.")

with tab4:
    st.write("### 🗄️ Traçabilité (SQLite)")
    st.info(
        "Ici s'afficheront les logs de requêtes, la matrice de confusion et le registre d'erreurs exigé par le cahier des charges.")