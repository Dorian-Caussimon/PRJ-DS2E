from __future__ import annotations

import json
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.inference import vlm_predict
from src.guardrails import apply_safety_guardrails
from src.database import insert_run, connect, init_db

DB_PATH = project_root / "data" / "medical_ai_evidence.sqlite"

st.set_page_config(
    page_title="Assistant radiologue virtuel — prototype pédagogique",
    page_icon="🩻",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS
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

[data-testid="stVerticalBlockBorderWrapper"] > div{
  background:var(--card-bg);
  border:1px solid var(--border);
  border-radius:14px;
  padding:1.4rem 1.6rem;
  box-shadow:0 1px 3px rgba(31,42,46,0.06);
  animation:fadeInUp .35s ease both;
}
@keyframes fadeInUp{ from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:translateY(0);} }

.eyebrow{ text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; color:var(--muted); font-weight:600; margin:0; }
.hero-title{ font-family:'Spectral', serif; font-weight:600; font-size:2.05rem; color:var(--ink); margin:.15rem 0 0 0; line-height:1.15; }
.hero-sub{ color:var(--muted); font-size:.95rem; margin:.5rem 0 0 0; max-width:46rem; }

.notice-banner{
  display:flex; gap:.65rem; align-items:flex-start;
  background:var(--accent-soft); border:1px solid var(--accent);
  border-radius:10px; padding:.8rem 1rem; margin:1.1rem 0 1.6rem 0;
  font-size:.86rem; line-height:1.45;
}
.notice-banner .dot{ flex-shrink:0; width:8px; height:8px; border-radius:50%; background:var(--accent); margin-top:.35rem; }

.section-label{ font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; color:var(--muted); font-weight:600; margin:0 0 .5rem 0; }

.badge{ display:inline-flex; align-items:center; gap:.45rem; padding:.4rem .9rem; border-radius:999px; font-weight:600; font-size:.92rem; }
.badge .dot{ width:8px; height:8px; border-radius:50%; }
.badge-normal{ background:var(--normal-soft); color:var(--normal); }
.badge-normal .dot{ background:var(--normal); }
.badge-suspected_opacity{ background:var(--opacity-soft); color:var(--opacity); }
.badge-suspected_opacity .dot{ background:var(--opacity); }
.badge-uncertain{ background:var(--uncertain-soft); color:var(--uncertain); }
.badge-uncertain .dot{ background:var(--uncertain); }

.tag{ display:inline-flex; align-items:center; gap:.35rem; padding:.2rem .6rem; border-radius:7px; font-size:.78rem; font-weight:500; }
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

.limitation-chip{ display:inline-block; background:var(--uncertain-soft); color:var(--muted); font-size:.78rem; padding:.28rem .7rem; border-radius:8px; margin:.2rem .35rem .2rem 0; }
.meta-caption{ color:var(--muted); font-size:.78rem; margin-top:1rem; }

/* Tableau historique */
.hist-table{ width:100%; border-collapse:collapse; font-size:.88rem; }
.hist-table th{ background:var(--accent-soft); color:var(--accent); font-weight:600; text-align:left; padding:.55rem .75rem; border-bottom:2px solid var(--accent); font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; }
.hist-table td{ padding:.55rem .75rem; border-bottom:1px solid var(--border); vertical-align:middle; }
.hist-table tr:hover td{ background:#F0FAF9; }
.hist-count{ font-family:'Spectral', serif; font-size:2.2rem; font-weight:600; color:var(--accent); line-height:1; }
.hist-sub{ color:var(--muted); font-size:.8rem; margin:.15rem 0 0 0; }
.stat-card{ background:var(--card-bg); border:1px solid var(--border); border-radius:12px; padding:1rem 1.2rem; text-align:center; }
"""

st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers d'affichage communs
# ---------------------------------------------------------------------------
CLASS_LABELS = {"normal": "Normal", "suspected_opacity": "Suspicion d'opacité", "uncertain": "Incertain"}
CLASS_ICONS  = {"normal": "🟢", "suspected_opacity": "🟠", "uncertain": "⚫"}
CLASS_COLOR_VARS = {"normal": "var(--normal)", "suspected_opacity": "var(--opacity)", "uncertain": "var(--uncertain)"}

def render_class_badge(predicted_class: str) -> str:
    label = CLASS_LABELS.get(predicted_class, predicted_class)
    return f'<span class="badge badge-{predicted_class}"><span class="dot"></span>{label}</span>'

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
        f'</div><div class="gauge-value" style="color:{color};">{pct}%</div></div>'
    )

def render_evidence_list(items: list[str]) -> str:
    rows = "".join(f"<li>{item}</li>" for item in items)
    return f'<ul class="evidence-list">{rows}</ul>'

def render_limitation_chips(items: list[str]) -> str:
    return "".join(f'<span class="limitation-chip">{item}</span>' for item in items)


# ---------------------------------------------------------------------------
# DB : lecture de l'historique
# ---------------------------------------------------------------------------
def fetch_history(limit: int = 100) -> list[dict]:
    try:
        init_db(DB_PATH)
        conn = connect(DB_PATH)
        rows = conn.execute(
            "SELECT id, case_id, image_path, predicted_class, confidence, model_name, latency_ms, created_at "
            "FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []

def fetch_run_detail(run_id: int) -> dict | None:
    try:
        conn = connect(DB_PATH)
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        conn.close()
        if row:
            r = dict(row)
            r["prediction_json"] = json.loads(r["prediction_json"])
            return r
    except Exception:
        return None

def count_by_class() -> dict[str, int]:
    try:
        conn = connect(DB_PATH)
        rows = conn.execute(
            "SELECT predicted_class, COUNT(*) as n FROM runs GROUP BY predicted_class"
        ).fetchall()
        conn.close()
        return {r["predicted_class"]: r["n"] for r in rows}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# En-tête (commun aux deux pages)
# ---------------------------------------------------------------------------
HEADER_MARK = (
    '<svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="18" cy="18" r="15.5" stroke="#0E7C86" stroke-width="2" stroke-dasharray="68 28" '
    'stroke-linecap="round" transform="rotate(-90 18 18)"/>'
    '<circle cx="18" cy="18" r="4" fill="#0E7C86"/>'
    "</svg>"
)

st.markdown(
    '<div style="display:flex; align-items:center; gap:.8rem;">'
    + HEADER_MARK
    + '<div><p class="eyebrow">Module de validation pédagogique — EFREI</p>'
    + '<h1 class="hero-title">Assistant radiologue virtuel</h1></div></div>'
    + '<p class="hero-sub">Pipeline de prédiction VLM sur radiographies thoraciques synthétiques — '
    + "destiné à valider l'enchaînement du code, pas à produire un avis médical. "
    + "Position non clinique : ce dépôt n'est pas un dispositif médical.</p>"
    + '<div class="notice-banner"><span class="dot"></span>'
    + '<div><strong>Prototype pédagogique.</strong> Non destiné au diagnostic. '
    + "Validation par un professionnel qualifié requise.</div></div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar : navigation + paramètres
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        label="Page",
        options=["🩻 Analyse", "📋 Historique"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Paramètres")
    mode = st.selectbox("Mode de prédiction", ["baseline", "improved"])
    st.caption(
        "**baseline** : prompt simple, sortie JSON directe.\n\n"
        "**improved** : raisonnement structuré en 3 étapes avec grille de calibration de la confiance."
    )


    with st.expander("Comment tester ?"):
        st.markdown(
            "Utilisez les images synthétiques de `data/sample_images`. "
            "Le nom de fichier encode la classe : `..._normal.png`, "
            "`..._suspected_opacity.png`, `..._uncertain.png`."
        )

    st.markdown("---")
    st.caption("Code pédagogique MIT. Datasets/modèles externes : licences propres.")


# ===========================================================================
# PAGE 1 — Analyse
# ===========================================================================
if page == "🩻 Analyse":
    uploaded = st.file_uploader(
        "Déposer une radiographie thoracique frontale",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded:
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_path = tmp_dir / uploaded.name
        tmp_path.write_bytes(uploaded.read())

        # Toujours via VLM — le mode détermine quel prompt est utilisé
        # baseline → prompts/prompt_baseline_v1.txt
        # improved → prompts/prompt_improved_v1.txt
        pred = apply_safety_guardrails(vlm_predict(tmp_path, mode=mode))

        # Persistance en base
        case_id = uuid.uuid4().hex[:8]
        try:
            insert_run(DB_PATH, case_id, str(tmp_path), pred)
        except Exception:
            pass  # ne pas bloquer l'UI si la DB n'est pas disponible

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            with st.container(border=True):
                st.markdown('<p class="section-label">Image source</p>', unsafe_allow_html=True)
                st.image(Image.open(tmp_path), use_container_width=True)
                st.markdown(render_quality_tag(pred["image_quality"]), unsafe_allow_html=True)
                st.caption(f"{uploaded.name} · ID run : {case_id}")

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
        st.info("Déposez une image, ou utilisez les images synthétiques de `data/sample_images`.")


# ===========================================================================
# PAGE 2 — Historique
# ===========================================================================
else:
    st.markdown("## 📋 Historique des prédictions")

    history = fetch_history()
    counts  = count_by_class()
    total   = sum(counts.values())

    # ---- Statistiques rapides ----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="stat-card"><div class="hist-count">{total}</div>'
            '<p class="hist-sub">Analyses totales</p></div>',
            unsafe_allow_html=True,
        )
    for col, cls in zip([c2, c3, c4], ["normal", "suspected_opacity", "uncertain"]):
        with col:
            n = counts.get(cls, 0)
            icon = CLASS_ICONS[cls]
            label = CLASS_LABELS[cls]
            st.markdown(
                f'<div class="stat-card"><div class="hist-count">{n}</div>'
                f'<p class="hist-sub">{icon} {label}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")  # espace

    if not history:
        st.info("Aucune prédiction enregistrée pour l'instant. Déposez une image dans la page Analyse.")
    else:
        # ---- Filtres ----
        fcol1, fcol2 = st.columns([1, 3])
        with fcol1:
            filter_class = st.selectbox(
                "Filtrer par classe",
                ["Toutes", "normal", "suspected_opacity", "uncertain"],
            )

        rows = history if filter_class == "Toutes" else [r for r in history if r["predicted_class"] == filter_class]

        # ---- Tableau ----
        with st.container(border=True):
            header = (
                '<table class="hist-table"><thead><tr>'
                "<th>#</th><th>Date</th><th>Fichier</th>"
                "<th>Classe</th><th>Confiance</th><th>Modèle</th><th>Latence</th>"
                "</tr></thead><tbody>"
            )
            body = ""
            for r in rows:
                cls   = r["predicted_class"]
                label = CLASS_LABELS.get(cls, cls)
                icon  = CLASS_ICONS.get(cls, "")
                conf  = f"{r['confidence']:.0%}"
                fname = Path(r["image_path"]).name
                date  = r["created_at"][:16].replace("T", " ")
                lat   = f"{r['latency_ms']} ms"
                body += (
                    f"<tr>"
                    f"<td style='color:var(--muted);font-size:.75rem;'>{r['id']}</td>"
                    f"<td>{date}</td>"
                    f"<td style='max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;' title='{fname}'>{fname}</td>"
                    f"<td><span class='badge badge-{cls}' style='font-size:.78rem;padding:.25rem .65rem;'>"
                    f"<span class='dot'></span>{icon} {label}</span></td>"
                    f"<td style='font-weight:600;'>{conf}</td>"
                    f"<td style='color:var(--muted);'>{r['model_name']}</td>"
                    f"<td style='color:var(--muted);'>{lat}</td>"
                    f"</tr>"
                )
            st.markdown(header + body + "</tbody></table>", unsafe_allow_html=True)

        st.caption(f"{len(rows)} entrée(s) affichée(s).")

        # ---- Détail d'un run ----
        st.markdown("---")
        st.markdown("**Inspecter un run en détail**")
        run_id = st.number_input("ID du run", min_value=1, step=1, value=rows[0]["id"] if rows else 1)
        if st.button("Charger le détail"):
            detail = fetch_run_detail(int(run_id))
            if detail:
                st.markdown(render_class_badge(detail["predicted_class"]), unsafe_allow_html=True)
                st.markdown(render_confidence_gauge(detail["confidence"], detail["predicted_class"]), unsafe_allow_html=True)
                st.json(detail["prediction_json"])
            else:
                st.warning(f"Aucun run trouvé avec l'ID {run_id}.")