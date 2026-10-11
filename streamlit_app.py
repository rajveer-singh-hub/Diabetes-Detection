import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report

import plots
from model import (ARTIFACT_DIR, GENDERS, NUMERIC, SMOKING, TARGET, load_clean, make_input,
                   risk_band)

st.set_page_config(page_title="Diabetes Risk Screening", layout="wide",
                   initial_sidebar_state="expanded")
plots.set_style()

NAVY, TEAL, SLATE, BG = "#0F2A43", "#0E7C86", "#5B6B7B", "#F4F6F9"
BAND_COLORS = {"Low": "#2E7D5B", "Moderate": "#C27C0E", "High": "#B3382C"}
SMOKING_LABELS = {"never": "Never smoked", "No Info": "No information", "former": "Former smoker",
                  "current": "Current smoker", "not current": "Not current", "ever": "Ever smoked"}

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background-color: {BG}; }}
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.5rem; max-width: 1250px; }}
    .app-header {{ background: {NAVY}; padding: 1.6rem 2rem; border-radius: 10px; margin-bottom: 1.5rem; }}
    .app-header h1 {{ font-size: 1.6rem; font-weight: 600; margin: 0; color: #fff; }}
    .app-header p  {{ margin: .35rem 0 0; color: #B8C7D6; font-size: .95rem; }}
    .card {{ background: #fff; border: 1px solid #E2E7ED; border-radius: 10px;
             padding: 1.4rem 1.6rem; margin-bottom: 1rem; }}
    .card h3 {{ font-size: .8rem; font-weight: 600; letter-spacing: .06em;
                text-transform: uppercase; color: {SLATE}; margin: 0 0 .8rem; }}
    .big-number {{ font-size: 2.8rem; font-weight: 700; color: {NAVY}; line-height: 1; }}
    .badge {{ display: inline-block; padding: .25rem .8rem; border-radius: 999px; color: #fff;
              font-size: .8rem; font-weight: 600; margin-left: .8rem; vertical-align: middle; }}
    .bar-bg {{ background: #E8ECF1; height: 10px; border-radius: 999px; margin-top: 1rem; }}
    .bar-fg {{ height: 10px; border-radius: 999px; }}
    .muted {{ color: {SLATE}; font-size: .88rem; }}
    section[data-testid="stSidebar"] {{ background: #fff; border-right: 1px solid #E2E7ED; }}
    section[data-testid="stSidebar"] h2 {{ font-size: 1rem; color: {NAVY}; font-weight: 600; }}
    .stFormSubmitButton > button {{ background: {TEAL}; color: #fff; border: 0; border-radius: 8px;
        font-weight: 600; padding: .6rem 1rem; width: 100%; }}
    .stFormSubmitButton > button:hover {{ background: {NAVY}; color: #fff; }}
    div[data-testid="stMetric"] {{ background: #fff; border: 1px solid #E2E7ED;
        border-radius: 10px; padding: .8rem 1rem; }}
    button[data-baseweb="tab"] {{ font-weight: 500; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    best = joblib.load(ARTIFACT_DIR / "best_model.joblib")
    summary = joblib.load(ARTIFACT_DIR / "summary.joblib")
    return best, summary


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    return load_clean()


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


try:
    best, S = load_artifacts()
except Exception as exc:  # missing or incompatible artifacts
    st.error("Model artifacts could not be loaded. Run `python train.py` once, then restart the app.")
    st.caption(str(exc))
    st.stop()

df = load_data()
model, best_name = best["model"], best["name"]
results, info = S["results"], S["info"]

st.markdown(
    """
    <div class="app-header">
        <h1>Diabetes Risk Screening</h1>
        <p>Eight machine-learning models compared on 96,000 patient records; the best performer powers the estimate.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Patient details")
    with st.form("patient_form"):
        gender = st.selectbox("Gender", GENDERS)
        age = st.number_input("Age (years)", 1.0, 100.0, 45.0, step=1.0)
        hyper = st.selectbox("Hypertension", ["No", "Yes"])
        heart = st.selectbox("Heart disease", ["No", "Yes"])
        smoking = st.selectbox("Smoking history", SMOKING, format_func=SMOKING_LABELS.get)
        bmi = st.number_input("BMI", 10.0, 70.0, 27.0, step=0.1, format="%.1f")
        hba1c = st.number_input("HbA1c level (%)", 3.5, 9.0, 5.5, step=0.1, format="%.1f")
        glucose = st.number_input("Blood glucose (mg/dL)", 50, 300, 120)
        submitted = st.form_submit_button("Run assessment")

tabs = st.tabs(["Assessment", "Data explorer", "Model comparison", "Best model analysis", "About"])

# ---------- Assessment ----------
with tabs[0]:
    if not submitted:
        st.markdown(
            '<div class="card"><h3>Getting started</h3><p class="muted" style="margin:0">'
            "Enter the patient details in the sidebar and select <b>Run assessment</b>."
            "</p></div>", unsafe_allow_html=True)
    else:
        X = make_input(gender, age, hyper == "Yes", heart == "Yes", smoking, bmi, hba1c, glucose)
        p = float(model.predict_proba(X)[0][1])
        band = risk_band(p)
        color = BAND_COLORS[band]
        left, right = st.columns([3, 2])
        with left:
            st.markdown(
                f"""
                <div class="card">
                    <h3>Estimated diabetes risk</h3>
                    <span class="big-number">{p*100:.1f}%</span>
                    <span class="badge" style="background:{color}">{band} risk</span>
                    <div class="bar-bg"><div class="bar-fg" style="width:{min(p*100, 100):.1f}%; background:{color}"></div></div>
                    <p class="muted" style="margin-top:.9rem">Predicted by <b>{best_name}</b>
                    (test accuracy {results.loc[best_name, 'Accuracy']*100:.2f}%). Only about
                    {df[TARGET].mean()*100:.0f}% of the training population is diabetic.</p>
                </div>""", unsafe_allow_html=True)
        with right:
            st.markdown("##### Patient vs. dataset medians")
            med = df.groupby(TARGET)[NUMERIC].median().T.rename(
                columns={0: "Non-diabetic", 1: "Diabetic"})
            med.insert(0, "Patient", [age, bmi, hba1c, glucose])
            med.index = [plots.NICE[i] for i in med.index]
            st.dataframe(med.round(2), width="stretch")
        show(plots.patient_position(df, {"HbA1c_level": hba1c, "blood_glucose_level": glucose}))
        st.caption("Educational tool, not a medical diagnosis. Consult a qualified healthcare professional.")

# ---------- Data explorer ----------
with tabs[1]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records (de-duplicated)", f"{len(df):,}")
    c2.metric("Features", 8)
    c3.metric("Diabetic share", f"{df[TARGET].mean()*100:.1f}%")
    c4.metric("Majority-class baseline", f"{info['baseline']*100:.1f}%")
    st.caption("3,854 exact duplicate rows were removed from the raw file to avoid train/test leakage. "
               "The classes are imbalanced, so accuracy should be read together with diabetic recall and F1.")
    a, b = st.columns([2, 3])
    with a:
        show(plots.class_distribution(df))
    with b:
        show(plots.correlation_heatmap(df))
    show(plots.numeric_distributions(df))
    show(plots.boxplots_by_outcome(df))
    show(plots.categorical_rates(df))
    with st.expander("View sample of the dataset"):
        st.dataframe(df.head(500), width="stretch")

# ---------- Model comparison ----------
with tabs[2]:
    st.markdown(f"**Best model: {best_name}** (highest test accuracy; ties broken by weighted F1)")
    styled = (results.style.format("{:.4f}")
              .apply(lambda r: ["background-color:#DCEFF1; font-weight:600" if r.name == best_name else ""
                                for _ in r], axis=1))
    st.dataframe(styled, width="stretch")
    st.caption(f"Trained on {info['train']:,} records, evaluated on {info['test']:,} held-out records. "
               "Precision, Recall and F1 are class-weighted; Diabetic Recall/F1 are for the positive class.")
    a, b = st.columns(2)
    with a:
        show(plots.accuracy_bar(results, info["baseline"]))
    with b:
        show(plots.metrics_comparison(results))
    a, b = st.columns(2)
    with a:
        show(plots.roc_curves(S["scores"], S["y_test"]))
    with b:
        show(plots.pr_curves(S["scores"], S["y_test"]))

# ---------- Best model analysis ----------
with tabs[3]:
    st.markdown(f"#### {best_name}")
    a, b = st.columns(2)
    with a:
        show(plots.confusion_matrix_fig(S["y_test"], S["best_pred"], f"{best_name}: confusion matrix"))
    with b:
        show(plots.importance_fig(S["importances"]))
    rep = pd.DataFrame(classification_report(
        S["y_test"], S["best_pred"], target_names=["Non-diabetic", "Diabetic"], output_dict=True)).T
    st.markdown("##### Classification report")
    st.dataframe(rep.round(3), width="stretch")

# ---------- About ----------
with tabs[4]:
    st.markdown(
        """
        **Dataset:** Diabetes Prediction Dataset (Kaggle), 100,000 records, 8 features
        (gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, blood glucose level)
        and binary target `diabetes`.

        **Pipeline:** duplicate removal, one-hot encoding of categorical features, standard scaling of
        numeric features, stratified 70/30 split, then eight models (Decision Tree, Logistic Regression,
        Random Forest, KNN, SVC, XGBoost, soft Voting, Stacking). The model with the highest test
        accuracy is used for predictions.

        **Reading the accuracy:** about 91% of patients are non-diabetic, so a model that always answers
        "non-diabetic" already scores about 91%. HbA1c and blood glucose are the clinical measures that
        define diabetes, which is why accuracy reaches about 97%. The weaker point is recall on the
        diabetic class, shown in the model comparison table.

        **Limitations:** educational project; not suitable for clinical decisions.
        """)
