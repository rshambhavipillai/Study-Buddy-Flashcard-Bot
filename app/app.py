"""Study Buddy — Gradio app.

Three tabs:
  1. Ask a Question  — Gemma answers study questions (via Ollama)
  2. Flashcard Generator — Gemma builds Q/A flashcard sets (via Ollama)
  3. Risk Dashboard  — OULAD-based dropout risk prediction (sklearn)

Requirements:
  brew install ollama
  ollama pull gemma:2b
  ollama serve          ← run in a separate terminal before starting the app

Run from project root:
    python app/app.py
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import gradio as gr
import joblib
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
LR_MODEL    = ROOT / "models" / "logistic_regression.pkl"
DT_MODEL    = ROOT / "models" / "decision_tree.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"


# ── Ollama inference ───────────────────────────────────────────────────────────

def generate(instruction: str, max_new_tokens: int = 400) -> str:
    prompt = (
        f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_new_tokens, "temperature": 0.7},
        }, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Ollama is not running.\n\n"
            "Start it with:\n```\nollama serve\n```\n"
            "Then make sure `gemma:2b` is pulled:\n```\nollama pull gemma:2b\n```"
        )
    except Exception as e:
        return f"⚠️ Error: {e}"


# ── load sklearn risk models ───────────────────────────────────────────────────
lr_model = joblib.load(LR_MODEL)    if LR_MODEL.exists()    else None
dt_model = joblib.load(DT_MODEL)    if DT_MODEL.exists()    else None
scaler   = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None


CATEGORICAL_MAPS = {
    "gender":            ["M", "F"],
    "highest_education": [
        "Lower Than A Level", "A Level or Equivalent",
        "HE Qualification", "Post Graduate Qualification", "No Formal quals",
    ],
    "imd_band": [
        "0-10%","10-20","20-30%","30-40%","40-50%",
        "50-60%","60-70%","70-80%","80-90%","90-100%","Unknown",
    ],
    "age_band":  ["0-35", "35-55", "55<="],
    "disability": ["N", "Y"],
    "region": [
        "East Anglian Region", "East Midlands Region", "Ireland",
        "London Region", "North Region", "North Western Region",
        "Scotland", "South East Region", "South Region",
        "South West Region", "Wales", "West Midlands Region",
        "Yorkshire Region",
    ],
}

# Exact column order the sklearn models were trained on
FEATURE_COLS = [
    "gender", "region", "highest_education", "imd_band", "age_band",
    "num_of_prev_attempts", "studied_credits", "disability", "date_registration",
    "avg_score", "max_score", "std_score", "num_submissions", "avg_days_early",
    "weighted_avg_score", "avg_tma_score", "avg_cma_score", "avg_exam_score",
    "total_clicks", "active_days",
    "clicks_forumng", "clicks_oucontent", "clicks_subpage", "clicks_homepage",
    "clicks_quiz", "clicks_resource", "clicks_url", "clicks_ouwiki",
]

def encode_features(gender, region, highest_education, imd_band, age_band,
                    disability, num_prev_attempts, studied_credits, date_registration,
                    avg_score, max_score, std_score, num_submissions, avg_days_early,
                    weighted_avg_score, avg_tma_score, avg_cma_score, avg_exam_score,
                    total_clicks, active_days):
    row = {
        "gender": gender, "region": region,
        "highest_education": highest_education, "imd_band": imd_band,
        "age_band": age_band, "num_of_prev_attempts": num_prev_attempts,
        "studied_credits": studied_credits, "disability": disability,
        "date_registration": date_registration, "avg_score": avg_score,
        "max_score": max_score, "std_score": std_score,
        "num_submissions": num_submissions, "avg_days_early": avg_days_early,
        "weighted_avg_score": weighted_avg_score, "avg_tma_score": avg_tma_score,
        "avg_cma_score": avg_cma_score, "avg_exam_score": avg_exam_score,
        "total_clicks": total_clicks, "active_days": active_days,
        # default activity-type clicks to 0 — not exposed in UI
        "clicks_forumng": 0, "clicks_oucontent": 0, "clicks_subpage": 0,
        "clicks_homepage": 0, "clicks_quiz": 0, "clicks_resource": 0,
        "clicks_url": 0, "clicks_ouwiki": 0,
    }
    df = pd.DataFrame([row])[FEATURE_COLS]
    for col in CATEGORICAL_MAPS:
        le = LabelEncoder()
        le.fit(CATEGORICAL_MAPS[col])
        df[col] = le.transform(df[col].astype(str))
    return df.values


def predict_risk(gender, region, highest_education, imd_band, age_band,
                 disability, num_prev_attempts, studied_credits, date_registration,
                 avg_score, max_score, std_score, num_submissions, avg_days_early,
                 weighted_avg_score, avg_tma_score, avg_cma_score, avg_exam_score,
                 total_clicks, active_days):
    if lr_model is None or dt_model is None or scaler is None:
        return "⚠️ Models not found. Run `python src/train_models.py` first."

    X = scaler.transform(encode_features(
        gender, region, highest_education, imd_band, age_band,
        disability, num_prev_attempts, studied_credits, date_registration,
        avg_score, max_score, std_score, num_submissions, avg_days_early,
        weighted_avg_score, avg_tma_score, avg_cma_score, avg_exam_score,
        total_clicks, active_days,
    ))
    lr_prob  = lr_model.predict_proba(X)[0][1]
    dt_prob  = dt_model.predict_proba(X)[0][1]
    avg_prob = (lr_prob + dt_prob) / 2

    if avg_prob >= 0.7:
        level = "🔴 HIGH RISK"
    elif avg_prob >= 0.4:
        level = "🟡 MODERATE RISK"
    else:
        level = "🟢 LOW RISK"

    return (
        f"## {level}\n\n"
        f"| Model | Dropout Probability |\n|---|---|\n"
        f"| Logistic Regression | {lr_prob:.1%} |\n"
        f"| Decision Tree | {dt_prob:.1%} |\n"
        f"| **Ensemble average** | **{avg_prob:.1%}** |\n\n"
        f"*Higher probability = higher risk of withdrawal.*"
    )


# ── Gradio UI ──────────────────────────────────────────────────────────────────

def ask_question(question):
    if not question.strip():
        return "Please enter a question."
    return generate(question)


def make_flashcards(topic, n_cards):
    if not topic.strip():
        return "Please enter a topic."
    instruction = f"Create {int(n_cards)} flashcards on the topic of {topic}."
    return generate(instruction, max_new_tokens=600)


with gr.Blocks(title="Study Buddy", theme=gr.themes.Default()) as demo:

    gr.Markdown("# Study Buddy\nAI-powered study assistant backed by Gemma 2B + OULAD risk models.")

    with gr.Tabs():

        # ── Tab 1: Ask a question ──────────────────────────────────────────────
        with gr.Tab("Ask a Question"):
            gr.Markdown("Ask any study question — concepts, explanations, definitions.")
            gr.Markdown(
                "**Example questions:** What is a p-value? · Explain recursion · "
                "What is natural selection? · How does gradient descent work?"
            )
            q_input  = gr.Textbox(label="Your question", placeholder="e.g. Explain what a p-value is", lines=2)
            q_button = gr.Button("Ask", variant="primary")
            q_output = gr.Markdown(label="Answer")
            q_button.click(fn=ask_question, inputs=q_input, outputs=q_output)

        # ── Tab 2: Flashcard generator ─────────────────────────────────────────
        with gr.Tab("Flashcard Generator"):
            gr.Markdown("Enter a topic and get instant revision flashcards.")
            gr.Markdown(
                "**Example topics:** sorting algorithms · SQL joins · "
                "Newton's laws · Git commands · supply and demand"
            )
            f_topic  = gr.Textbox(label="Topic", placeholder="e.g. SQL joins, Newton's laws, photosynthesis")
            f_n      = gr.Slider(minimum=3, maximum=10, value=5, step=1, label="Number of flashcards")
            f_button = gr.Button("Generate Flashcards", variant="primary")
            f_output = gr.Markdown(label="Flashcards")
            f_button.click(fn=make_flashcards, inputs=[f_topic, f_n], outputs=f_output)

        # ── Tab 3: Risk dashboard ──────────────────────────────────────────────
        with gr.Tab("Dropout Risk Assessment"):
            gr.Markdown("Enter a student profile to predict dropout risk using OULAD-trained models.")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Demographics**")
                    gender      = gr.Dropdown(["M", "F"], value="M", label="Gender")
                    age_band    = gr.Dropdown(["0-35", "35-55", "55<="], value="0-35", label="Age band")
                    region      = gr.Dropdown(CATEGORICAL_MAPS["region"], value="South East Region", label="Region")
                    disability  = gr.Dropdown(["N", "Y"], value="N", label="Disability")
                    imd_band    = gr.Dropdown(CATEGORICAL_MAPS["imd_band"], value="50-60%", label="IMD deprivation band")
                    highest_edu = gr.Dropdown(CATEGORICAL_MAPS["highest_education"], value="A Level or Equivalent", label="Highest education")
                    num_prev    = gr.Slider(0, 6, value=0, step=1, label="Previous attempts")
                    credits     = gr.Slider(30, 600, value=60, step=30, label="Studied credits")

                with gr.Column():
                    gr.Markdown("**Assessment performance**")
                    date_reg    = gr.Slider(-200, 0, value=-30, step=1, label="Days registered before course start")
                    avg_score   = gr.Slider(0, 100, value=65, label="Average score")
                    max_score   = gr.Slider(0, 100, value=80, label="Max score")
                    std_score   = gr.Slider(0, 50,  value=10, label="Score std dev")
                    n_subs      = gr.Slider(0, 20,  value=5,  step=1, label="Submissions count")
                    avg_early   = gr.Slider(-30, 30, value=2, label="Avg days submitted early")
                    w_avg       = gr.Slider(0, 100, value=65, label="Weighted avg score")
                    tma_score   = gr.Slider(0, 100, value=65, label="Avg TMA score")
                    cma_score   = gr.Slider(0, 100, value=65, label="Avg CMA score")
                    exam_score  = gr.Slider(0, 100, value=60, label="Avg exam score")

                with gr.Column():
                    gr.Markdown("**VLE engagement**")
                    tot_clicks  = gr.Slider(0, 50000, value=3000, step=100, label="Total VLE clicks")
                    active_days = gr.Slider(0, 300, value=60, step=1, label="Active study days")
                    gr.Markdown("---")
                    risk_output = gr.Markdown(value="*Fill in the profile and click Predict.*")
                    risk_button = gr.Button("Predict Risk", variant="primary")

            risk_button.click(
                fn=predict_risk,
                inputs=[
                    gender, region, highest_edu, imd_band, age_band, disability,
                    num_prev, credits, date_reg,
                    avg_score, max_score, std_score, n_subs,
                    avg_early, w_avg, tma_score, cma_score, exam_score,
                    tot_clicks, active_days,
                ],
                outputs=risk_output,
            )


if __name__ == "__main__":
    demo.launch(share=False)
