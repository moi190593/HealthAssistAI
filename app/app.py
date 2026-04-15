"""
Interfície Streamlit per al sistema de suport al triatge d'atenció primària (CAP).
Pestanyes:
  1. Predicció   — introdueix dades clíniques i obtiés la prioritat de visita
  2. Comparació  — taula i gràfic amb la comparació de tots els models entrenats
Executar amb: python -m streamlit run app/app.py
"""
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH       = os.path.join(ROOT, "model", "pkl", "triage_model.pkl")
LABELS_PATH      = os.path.join(ROOT, "model", "pkl", "label_names.pkl")
META_PATH        = os.path.join(ROOT, "model", "pkl", "ui_meta.pkl")
COMPARISON_PATH  = os.path.join(ROOT, "model", "pkl", "comparison_results.pkl")
JUSTIF_PATH      = os.path.join(ROOT, "model", "pkl", "model_justification.pkl")
BEST_NAME_PATH   = os.path.join(ROOT, "model", "pkl", "best_model_name.pkl")
BEST_PARAMS_PATH = os.path.join(ROOT, "model", "pkl", "best_params.pkl")

# --- Configuració de pàgina ---
st.set_page_config(
    page_title="HealthAssist AI — Triatge",
    page_icon="🏥",
    layout="centered",
)

# --- Configuració visual per nivell ---
LEVEL_CONFIG = {
    1: {"color": "#c0392b", "bg": "#fdecea", "icon": "🔴", "wait": "Atenció immediata (< 15 min)"},
    2: {"color": "#e67e22", "bg": "#fef5e7", "icon": "🟠", "wait": "Atenció urgent (< 30 min)"},
    3: {"color": "#2980b9", "bg": "#eaf4fb", "icon": "🔵", "wait": "Atenció preferent (< 60 min)"},
    4: {"color": "#27ae60", "bg": "#eafaf1", "icon": "🟢", "wait": "Atenció no urgent (> 60 min)"},
}

RECOMMENDATIONS = {
    1: "Assignar a box de crítics. Derivar a atenció hospitalària via SEM. Monitoratge continu.",
    2: "Assignar box d'exploració. Avaluació mèdica en menys de 30 minuts.",
    3: "Ubicar a sala d'espera assistida. Reavaluar si canvi clínic.",
    4: "Orientar a consulta d'atenció primària o zona d'espera general.",
}

# --- Carregar model ---
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model       = joblib.load(MODEL_PATH)
    label_names = joblib.load(LABELS_PATH)
    meta        = joblib.load(META_PATH)
    return model, label_names, meta

model, label_names, meta = load_artifacts()

# --- Capçalera ---
st.title("🏥 HealthAssist AI")
st.subheader("Sistema de suport al triatge d'atenció primària en adults (CAP)")
st.divider()

if model is None:
    st.error(
        "⚠️ Model no trobat. Executa primer l'script d'entrenament:\n\n"
        "```bash\ncd HealthAssistAI\npython model/train_model.py\n```"
    )
    st.stop()

# PESTANYES
tab_pred, tab_cmp = st.tabs(["🔍 Predicció", "📊 Comparació de models"])


# PESTANYA 1 — PREDICCIÓ
with tab_pred:
    st.markdown("Introdueix les dades clíniques del pacient per obtenir el **nivell de triatge**.")

    st.markdown("#### Dades del pacient")
    col1, col2, col3 = st.columns(3)
    with col1:
        edat = st.number_input("Edat (anys)", min_value=15, max_value=120, value=35)
    with col2:
        genere = st.selectbox("Gènere", options=meta["genere_options"])
    with col3:
        simptomes_ordenats = sorted(
            meta["simptomes_options"],
            key=lambda s: meta.get("simptoma_a_nivell", {}).get(s, 99)
        )
        simptoma = st.selectbox("Símptoma principal", options=simptomes_ordenats)

    st.markdown("#### Constants vitals")
    col7, col8, col9, col10 = st.columns(4)
    with col7:
        ta_sis = st.number_input("TA sistòlica (mmHg)", min_value=50, max_value=250, value=120)
    with col8:
        ta_dia = st.number_input("TA diastòlica (mmHg)", min_value=30, max_value=150, value=80)
    with col9:
        fc = st.number_input("Freqüència cardíaca (bpm)", min_value=20, max_value=250, value=75)
    with col10:
        temp = st.number_input("Temperatura (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)

    col11, col12 = st.columns(2)
    with col11:
        spo2 = st.number_input("Saturació d'oxigen SpO2 (%)", min_value=50, max_value=100, value=98)
    with col12:
        fr = st.number_input("Freqüència respiratòria (rpm)", min_value=4, max_value=60, value=16)

    submitted = st.button("🔍 Predir nivell de triatge", use_container_width=True, type="primary")

    if submitted:
        input_df = pd.DataFrame([{
            "Edat": edat,
            "Gènere": genere,
            "Simptomes principals": simptoma,
            "TA_sistolica": float(ta_sis),
            "TA_diastolica": float(ta_dia),
            "Freqüència cardíaca": fc,
            "Temperatura": temp,
            "Saturació_oxigen": spo2,
            "Freqüència_respiratoria": fr,
        }])

        proba = model.predict_proba(input_df)[0]
        classes = model.classes_
        predicted_level = int(classes[np.argmax(proba)])
        confidence = proba.max() * 100

        cfg = LEVEL_CONFIG[predicted_level]

        st.divider()
        st.markdown("### Resultat de la classificació")

        st.markdown(
            f"""
            <div style="
                background-color: {cfg['bg']};
                border-left: 8px solid {cfg['color']};
                border-radius: 8px;
                padding: 20px 24px;
                margin-bottom: 16px;
            ">
                <h2 style="color: {cfg['color']}; margin: 0;">
                    {cfg['icon']} {label_names[predicted_level]}
                </h2>
                <p style="font-size: 1.1rem; margin: 8px 0 4px 0;">
                    <b>Temps d'espera màxim:</b> {cfg['wait']}
                </p>
                <p style="margin: 0;">
                    <b>Recomanació:</b> {RECOMMENDATIONS[predicted_level]}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric("Confiança del model", f"{confidence:.1f}%")

        with st.expander("📊 Distribució de probabilitats per nivell"):
            prob_df = pd.DataFrame({
                "Nivell": [f"{LEVEL_CONFIG[int(c)]['icon']} {label_names[int(c)]}" for c in classes],
                "Probabilitat (%)": [round(p * 100, 2) for p in proba],
            }).sort_values("Probabilitat (%)", ascending=False)
            st.bar_chart(prob_df.set_index("Nivell"))
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

        st.info(
            "⚠️ **Avís:** Aquesta eina és un prototip educatiu. "
            "No substitueix el criteri clínic d'un professional sanitari."
        )

    st.divider()
    with st.expander("ℹ️ Prioritats del triatge d'atenció primària"):
        for level, cfg in LEVEL_CONFIG.items():
            st.markdown(
                f"{cfg['icon']} **{label_names[level]}** — Temps màxim: {cfg['wait']}. "
                f"{RECOMMENDATIONS[level]}"
            )

#
# PESTANYA 2 — COMPARACIÓ DE MODELS
#
with tab_cmp:
    st.markdown("### Comparació de models de classificació")
    st.markdown(
        "S'han entrenat **5 models** amb validació creuada de 3 particions (3-fold CV). "
        "El criteri de selecció principal és el **F1 macro**, que penalitza els errors "
        "en classes minoritàries (nivells crítics)."
    )

    if not os.path.exists(COMPARISON_PATH):
        st.warning("Resultats de comparació no trobats. Executa `python model/train_model.py`.")
    else:
        cmp_df    = joblib.load(COMPARISON_PATH)
        justif    = joblib.load(JUSTIF_PATH)
        best_name = joblib.load(BEST_NAME_PATH)

        # Hiperparàmetres òptims (opcional, pot no existir en installations antigues)
        best_params = joblib.load(BEST_PARAMS_PATH) if os.path.exists(BEST_PARAMS_PATH) else {}

        # --- Taula de resultats ---
        st.markdown("#### Resultats RandomizedSearchCV (CV 3-fold)")
        display_df = cmp_df[["Model", "Accuracy CV", "Accuracy std", "F1 macro CV", "F1 std", "Temps (s)"]].copy()
        display_df = display_df.set_index("Model")

        def highlight_best(s):
            is_best = s == s.max()
            return ["background-color: #d4edda; font-weight: bold" if v else "" for v in is_best]

        styled = display_df.style.apply(highlight_best, subset=["F1 macro CV"])
        st.dataframe(styled, use_container_width=True)

        # --- Gràfic de barres comparatiu ---
        st.markdown("#### F1 macro per model")
        chart_df = cmp_df.set_index("Model")[["F1 macro CV"]].sort_values("F1 macro CV")
        st.bar_chart(chart_df)

        # --- Guanyador ---
        best_row = cmp_df[cmp_df["Model"] == best_name].iloc[0]
        st.success(
            f"**Model seleccionat: {best_name}** — "
            f"F1 macro: {best_row['F1 macro CV']:.4f} · "
            f"Accuracy: {best_row['Accuracy CV']:.4f} · "
            f"Temps: {best_row['Temps (s)']}s"
        )

        # --- Justificació individual de cada model ---
        st.markdown("#### Per què cada model?")
        for model_name, text in justif.items():
            icon = "🏆 " if model_name == best_name else ""
            with st.expander(f"{icon}{model_name}"):
                st.write(text)
                row = cmp_df[cmp_df["Model"] == model_name].iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("F1 macro CV",  f"{row['F1 macro CV']:.4f}")
                c2.metric("Accuracy CV",  f"{row['Accuracy CV']:.4f}")
                c3.metric("Temps (s)",    f"{row['Temps (s)']}s")
                if model_name in best_params:
                    st.markdown("**Hiperparàmetres òptims:**")
                    st.code(best_params[model_name], language=None)
