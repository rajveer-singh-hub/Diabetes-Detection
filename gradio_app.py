import joblib
import gradio as gr

from model import ARTIFACT_DIR, GENDERS, SMOKING, make_input, risk_band

bundle = joblib.load(ARTIFACT_DIR / "best_model.joblib")
model, best_name = bundle["model"], bundle["name"]


def assess(gender, age, hypertension, heart_disease, smoking, bmi, hba1c, glucose):
    X = make_input(gender, age, hypertension == "Yes", heart_disease == "Yes", smoking, bmi, hba1c, glucose)
    p = float(model.predict_proba(X)[0][1])
    return ({"Diabetic": p, "Non-diabetic": 1 - p},
            f"Model: {best_name}  |  Probability: {p*100:.1f}%  |  Risk band: {risk_band(p)}")


theme = gr.themes.Base(primary_hue="teal", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "sans-serif"])

with gr.Blocks(title="Diabetes Risk Screening") as demo:
    gr.Markdown(f"# Diabetes Risk Screening\nPowered by the best of eight compared models: **{best_name}**.")
    with gr.Row():
        with gr.Column():
            gender = gr.Dropdown(GENDERS, value="Female", label="Gender")
            age = gr.Number(label="Age (years)", value=45)
            hypertension = gr.Radio(["No", "Yes"], value="No", label="Hypertension")
            heart_disease = gr.Radio(["No", "Yes"], value="No", label="Heart disease")
        with gr.Column():
            smoking = gr.Dropdown(SMOKING, value="never", label="Smoking history")
            bmi = gr.Number(label="BMI", value=27.0)
            hba1c = gr.Number(label="HbA1c level (%)", value=5.5)
            glucose = gr.Number(label="Blood glucose (mg/dL)", value=120)
    btn = gr.Button("Run assessment", variant="primary")
    out_label = gr.Label(label="Result", num_top_classes=2)
    out_text = gr.Textbox(label="Summary", interactive=False)
    btn.click(assess, [gender, age, hypertension, heart_disease, smoking, bmi, hba1c, glucose],
              [out_label, out_text])
    gr.Markdown("*For educational use only. Not a medical diagnosis.*")

if __name__ == "__main__":
    demo.launch(theme=theme)
