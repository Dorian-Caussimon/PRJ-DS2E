from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import streamlit as st
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.inference import toy_predict
from src.guardrails import apply_safety_guardrails

st.set_page_config(
    page_title="Assistant radiologue virtuel — prototype pédagogique",
    page_icon="🩻",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Identité visuelle
# Palette clinique sobre : fond ivoire, accent sarcelle, statuts nuancés
# (vert = normal, ambre = suspicion, ardoise = incertain — jamais de rouge
# alarmiste, ce n'est qu'un outil pédagogique, pas un dispositif médical).
# Typo : Spectral (serif académique) pour les titres, Inter pour le reste.
#
# NB technique : les sélecteurs [data-testid="..."] ci-dessous ciblent le
# DOM interne de Streamlit. Ils sont stables depuis plusieurs versions, mais
# si Streamlit change sa structure interne, il faudra ajuster ces sélecteurs.
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Spectral:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root{
  --bg:#F6F7F4;
  --card-bg:#FFFFFF;
  --ink:#1F2A2E;
  --muted:#5B6B6F;
  --border:#E4E7E5;
  --accent:#0E7C86;
  --accent-soft:#E3F2F1;
  --normal:#2F7D5B;
  --normal-soft:#E6F4EC;
  --opacity:#C2660B;
  --opacity-soft:#FBEBD9;
  --uncertain:#5B6472;
  --uncertain-soft:#ECEEF0;
}

.stApp{ background:var(--bg); }
html, body, [class*="css"]{ font-family:'Inter', sans-serif; color:var(--ink); }

/* Cartes : on stylise toutes les zones st.container(border=True) de la page */
[data-testid="stVerticalBlockBorderWrapper"] > div{
  background:var(--card-bg);
  border:1px solid var(--border);
  border-radius:14px;
  padding:1.4rem 1.6rem;
  box-shadow:0 1px 3px rgba(31,42,46,0.06);
  animation:fadeInUp .35s ease both;
}
@keyframes fadeInUp{ from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:translateY(0);} }

.eyebrow{
  text-transform:uppercase; letter-spacing:.14em; font-size:.72rem;
  color:var(--muted); font-weight:600; margin:0;
}
.hero-title{
  font-family:'Spectral', serif; font-weight:600; font-size:2.05rem;
  color:var(--ink); margin:.15rem 0 0 0; line-height:1.15;
}
.hero-sub{ color:var(--muted); font-size:.95rem; margin:.5rem 0 0 0; max-width:46rem; }

.notice-banner{
  display:flex; gap:.65rem; align-items:flex-start;
  background:var(--accent-soft); border:1px solid var(--accent);
  border-radius:10px; padding:.8rem 1rem; margin:1.1rem 0 1.6rem 0;
  font-size:.86rem; line-height:1.45;
}
.notice-banner .dot{ flex-shrink:0; width:8px; height:8px; border-radius:50%; background:var(--accent); margin-top:.35rem; }

.section-label{
  font-size:.72rem; text-transform:uppercase; letter-spacing:.12em;
  color:var(--muted); font-weight:600; margin:0 0 .5rem 0;
}

.badge{
  display:inline-flex; align-items:center; gap:.45rem;
  padding:.4rem .9rem; border-radius:999px; font-weight:600; font-size:.92rem;
}
.badge .dot{ width:8px; height:8px; border-radius:50%; }
.badge-normal{ background:var(--normal-soft); color:var(--normal); }
.badge-normal .dot{ background:var(--normal); }
.badge-suspected_opacity{ background:var(--opacity-soft); color:var(--opacity); }
.badge-suspected_opacity .dot{ background:var(--opacity); }
.badge-uncertain{ background:var(--uncertain-soft); color:var(--uncertain); }
.badge-uncertain .dot{ background:var(--uncertain); }

.tag{
  display:inline-flex; align-items:center; gap:.35rem;
  padding:.2rem .6rem; border-radius:7px; font-size:.78rem; font-weight:500;
}
.tag .dot{ width:6px; height:6px; border-radius:50%; }
.tag-good{ background:var(--normal-soft); color:var(--normal); }
.tag-good .dot{ background:var(--normal); }
.tag-limited{ background:var(--opacity-soft); color:var(--opacity); }
.tag-limited .dot{ background:var(--opacity); }

.gauge-row{ display:flex; align-items:center; gap:.8rem; margin-top:.3rem; }
.gauge-track{ flex:1; position:relative; height:10px; border-radius:6px; background:var(--border); overflow:hidden; }
.gauge-fill{ position:absolute; left:0; top:0; height:100%; border-radius:6px; }
.gauge-value{ font-weight:700; font-size:.95rem; min-width:3.2rem; text-align:right; }

.evidence-list{ margin:.2rem 0 0 0; padding-left:1.15rem; font-size:.93rem; line-height:1.6; color:var(--ink); }

.justification-quote{
  border-left:3px solid var(--accent); background:var(--accent-soft);
  border-radius:0 8px 8px 0; padding:.6rem .95rem; margin:.3rem 0 0 0;
  font-size:.92rem; font-style:italic; color:var(--ink);
}

.limitation-chip{
  display:inline-block; background:var(--uncertain-soft); color:var(--muted);
  font-size:.78rem; padding:.28rem .7rem; border-radius:8px; margin:.2rem .35rem .2rem 0;
}

.meta-caption{ color:var(--muted); font-size:.78rem; margin-top:1rem; }
"""

st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
HEADER_MARK = (
    '<svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="18" cy="18" r="15.5" stroke="#0E7C86" stroke-width="2" stroke-dasharray="68 28" '
    'stroke-linecap="round" transform="rotate(-90 18 18)"/>'
    '<circle cx="18" cy="18" r="4" fill="#0E7C86"/>'
    "</svg>"
)

header_html = (
    '<div style="display:flex; align-items:center; gap:.8rem;">'
    + HEADER_MARK
    + '<div><p class="eyebrow">Module de validation pédagogique — EFREI</p>'
    + '<h1 class="hero-title">Assistant radiologue virtuel</h1></div></div>'
    + '<p class="hero-sub">Pipeline de prédiction (toy / VLM) sur radiographies thoraciques synthétiques — '
    + "destiné à valider l'enchaînement du code, pas à produire un avis médical. "
    + "Position non clinique : ce dépôt n'est pas un dispositif médical.</p>"
    + '<div class="notice-banner"><span class="dot"></span>'
    + '<div><strong>Prototype pédagogique.</strong> Non destiné au diagnostic. '
    + "Validation par un professionnel qualifié requise.</div></div>"
)
st.markdown(header_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Barre latérale
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Paramètres")
    mode = st.selectbox("Mode de prédiction", ["baseline", "improved"])
    st.caption("« improved » est plus conservateur : il bascule en *uncertain* dès que la qualité d'image est limitée.")

    with st.expander("Comment tester ce prototype ?"):
        st.markdown(
            "Utilisez les images synthétiques du dossier `data/sample_images`. "
            "Leur nom de fichier encode la classe attendue "
            "(`..._normal.png`, `..._suspected_opacity.png`, `..._uncertain.png`)."
        )

    st.markdown("---")
    st.caption("Code pédagogique sous licence MIT. Les datasets/modèles externes conservent leurs licences propres.")


# ---------------------------------------------------------------------------
# Fonctions d'affichage
# ---------------------------------------------------------------------------
CLASS_LABELS = {
    "normal": "Normal",
    "suspected_opacity": "Suspicion d'opacité",
    "uncertain": "Incertain",
}
CLASS_COLOR_VARS = {
    "normal": "var(--normal)",
    "suspected_opacity": "var(--opacity)",
    "uncertain": "var(--uncertain)",
}


def render_class_badge(predicted_class: str) -> str:
    label = CLASS_LABELS.get(predicted_class, predicted_class)
    return (
        f'<span class="badge badge-{predicted_class}">'
        f'<span class="dot"></span>{label}</span>'
    )


def render_quality_tag(image_quality: str) -> str:
    css_class = "tag-good" if image_quality == "good" else "tag-limited"
    label = "Qualité d'image : bonne" if image_quality == "good" else "Qualité d'image : limitée"
    return f'<span class="tag {css_class}"><span class="dot"></span>{label}</span>'


def render_confidence_gauge(confidence: float, predicted_class: str) -> str:
    pct = max(0, min(100, round(confidence * 100)))
    color = CLASS_COLOR_VARS.get(predicted_class, "var(--accent)")
    return (
        '<div class="gauge-row"><div class="gauge-track">'
        f'<div class="gauge-fill" style="width:{pct}%; background:{color};"></div>'
        "</div>"
        f'<div class="gauge-value" style="color:{color};">{pct}%</div>'
        "</div>"
    )


def render_evidence_list(items: list[str]) -> str:
    rows = "".join(f"<li>{item}</li>" for item in items)
    return f'<ul class="evidence-list">{rows}</ul>'


def render_limitation_chips(items: list[str]) -> str:
    return "".join(f'<span class="limitation-chip">{item}</span>' for item in items)


# ---------------------------------------------------------------------------
# Corps principal
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("Déposer une radiographie thoracique frontale", type=["png", "jpg", "jpeg"])

if uploaded:
    # On conserve le nom de fichier original : toy_predict() lit la classe
    # synthétique dans ce nom, un nom aléatoire ferait toujours retomber sur "uncertain".
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / uploaded.name
    tmp_path.write_bytes(uploaded.read())

    pred = apply_safety_guardrails(toy_predict(tmp_path, mode=mode))

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        with st.container(border=True):
            st.markdown('<p class="section-label">Image source</p>', unsafe_allow_html=True)
            st.image(Image.open(tmp_path), use_container_width=True)
            st.markdown(render_quality_tag(pred["image_quality"]), unsafe_allow_html=True)
            st.caption(uploaded.name)

    with col2:
        with st.container(border=True):
            st.markdown('<p class="section-label">Résultat de l\'analyse</p>', unsafe_allow_html=True)
            st.markdown(render_class_badge(pred["predicted_class"]), unsafe_allow_html=True)

            st.markdown('<p class="section-label" style="margin-top:1.1rem;">Confiance</p>', unsafe_allow_html=True)
            st.markdown(render_confidence_gauge(pred["confidence"], pred["predicted_class"]), unsafe_allow_html=True)

            st.markdown('<p class="section-label" style="margin-top:1.1rem;">Observations</p>', unsafe_allow_html=True)
            st.markdown(render_evidence_list(pred["visual_evidence"]), unsafe_allow_html=True)

            st.markdown('<p class="section-label" style="margin-top:1.1rem;">Justification</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="justification-quote">{pred["justification"]}</div>', unsafe_allow_html=True)

            st.markdown('<p class="section-label" style="margin-top:1.1rem;">Limites</p>', unsafe_allow_html=True)
            st.markdown(render_limitation_chips(pred["limitations"]), unsafe_allow_html=True)

            st.markdown(
                f'<p class="meta-caption">Modèle : {pred["model_name"]} · '
                f'Prompt : {pred["prompt_version"]} · Latence : {pred["latency_ms"]} ms</p>',
                unsafe_allow_html=True,
            )

            with st.expander("Détails techniques (JSON)"):
                st.json(pred)
else:
    st.info("Déposez une image, ou utilisez les images synthétiques du dossier `data/sample_images` pour tester le flux.")