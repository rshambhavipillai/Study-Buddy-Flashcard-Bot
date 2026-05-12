"""Study Buddy — Gradio app.

Three tabs:
  1. Ask a Question  — Gemma answers study questions
  2. Flashcard Generator — Gemma builds Q/A flashcard sets
  3. Risk Dashboard  — OULAD-based dropout risk prediction (sklearn)

Run from project root:
    python app/app.py
"""

import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "models" / "gemma-study-buddy"
LR_MODEL    = ROOT / "models" / "logistic_regression.pkl"
DT_MODEL    = ROOT / "models" / "decision_tree.pkl"

# ── device detection ───────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE      = "cuda"
    USE_4BIT    = True
elif torch.backends.mps.is_available():
    DEVICE      = "mps"
    USE_4BIT    = False          # bitsandbytes requires CUDA
else:
    DEVICE      = "cpu"
    USE_4BIT    = False

print(f"Device: {DEVICE}  |  4-bit quantisation: {USE_4BIT}")

# ── load Gemma + adapter ───────────────────────────────────────────────────────
model      = None
tokenizer  = None
GEMMA_READY = False

def load_gemma():
    global model, tokenizer, GEMMA_READY
    if not ADAPTER_DIR.exists():
        print(f"[WARN] Adapter not found at {ADAPTER_DIR}. "
              "Download it from Kaggle and place it in models/gemma-study-buddy/")
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    BASE_ID = "google/gemma-2b-it"
    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model …")
    if USE_4BIT:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_ID, quantization_config=bnb,
            device_map={"": 0}, torch_dtype=torch.bfloat16,
        )
    else:
        # MPS has a 4 GB per-buffer limit that Gemma 2B exceeds in float16.
        # Load on CPU in float32 — slower but works on any Mac.
        base = AutoModelForCausalLM.from_pretrained(
            BASE_ID, dtype=torch.float32, device_map="cpu",
        )

    print("Attaching LoRA adapter …")
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model.eval()
    GEMMA_READY = True
    print("Gemma ready.")


def generate(instruction: str, max_new_tokens: int = 400) -> str:
    if not GEMMA_READY:
        return "⚠️ Gemma model not loaded. Place the adapter in models/gemma-study-buddy/ and restart."
    prompt = (
        f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    new = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


# ── load sklearn risk models ───────────────────────────────────────────────────
lr_model = joblib.load(LR_MODEL) if LR_MODEL.exists() else None
dt_model = joblib.load(DT_MODEL) if DT_MODEL.exists() else None


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

def encode_features(
    gender, region, highest_education, imd_band, age_band, disability,
    num_prev_attempts, studied_credits, date_registration,
    avg_score, max_score, std_score, num_submissions,
    avg_days_early, weighted_avg_score,
    avg_tma_score, avg_cma_score, avg_exam_score,
    total_clicks, active_days,
):
    row = {
        "gender":            gender,
        "region":            region,
        "highest_education": highest_education,
        "imd_band":          imd_band,
        "age_band":          age_band,
        "disability":        disability,
        "num_of_prev_attempts": num_prev_attempts,
        "studied_credits":   studied_credits,
        "date_registration": date_registration,
        "avg_score":         avg_score,
        "max_score":         max_score,
        "std_score":         std_score,
        "num_submissions":   num_submissions,
        "avg_days_early":    avg_days_early,
        "weighted_avg_score": weighted_avg_score,
        "avg_tma_score":     avg_tma_score,
        "avg_cma_score":     avg_cma_score,
        "avg_exam_score":    avg_exam_score,
        "total_clicks":      total_clicks,
        "active_days":       active_days,
    }
    df = pd.DataFrame([row])
    for col, _ in CATEGORICAL_MAPS.items():
        le = LabelEncoder()
        le.fit(CATEGORICAL_MAPS[col])
        df[col] = le.transform(df[col].astype(str))
    return df.values


def predict_risk(
    gender, region, highest_education, imd_band, age_band, disability,
    num_prev_attempts, studied_credits, date_registration,
    avg_score, max_score, std_score, num_submissions,
    avg_days_early, weighted_avg_score,
    avg_tma_score, avg_cma_score, avg_exam_score,
    total_clicks, active_days,
):
    if lr_model is None or dt_model is None:
        return "Models not found. Run src/train_models.py first."

    X = encode_features(
        gender, region, highest_education, imd_band, age_band, disability,
        num_prev_attempts, studied_credits, date_registration,
        avg_score, max_score, std_score, num_submissions,
        avg_days_early, weighted_avg_score,
        avg_tma_score, avg_cma_score, avg_exam_score,
        total_clicks, active_days,
    )
    lr_prob = lr_model.predict_proba(X)[0][1]
    dt_prob = dt_model.predict_proba(X)[0][1]
    avg_prob = (lr_prob + dt_prob) / 2

    if avg_prob >= 0.7:
        level, colour = "🔴 HIGH RISK", "red"
    elif avg_prob >= 0.4:
        level, colour = "🟡 MODERATE RISK", "orange"
    else:
        level, colour = "🟢 LOW RISK", "green"

    return (
        f"## {level}\n\n"
        f"| Model | Dropout Probability |\n"
        f"|---|---|\n"
        f"| Logistic Regression | {lr_prob:.1%} |\n"
        f"| Decision Tree       | {dt_prob:.1%} |\n"
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


with gr.Blocks(title="Study Buddy", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 📚 Study Buddy\nAI-powered study assistant backed by Gemma 2B + OULAD risk models.")

    with gr.Tabs():

        # ── Tab 1: Ask a question ──────────────────────────────────────────────
        with gr.Tab("💬 Ask a Question"):
            gr.Markdown("Ask any study question — concepts, explanations, definitions.")
            q_input  = gr.Textbox(label="Your question", placeholder="e.g. Explain what a p-value is", lines=2)
            q_button = gr.Button("Ask", variant="primary")
            q_output = gr.Markdown(label="Answer")
            q_button.click(fn=ask_question, inputs=q_input, outputs=q_output)
            gr.Examples(
                examples=[
                    ["What is the difference between mean, median, and mode?"],
                    ["Explain recursion in programming with an example."],
                    ["What is natural selection?"],
                    ["How does gradient descent work?"],
                ],
                inputs=q_input,
            )

        # ── Tab 2: Flashcard generator ─────────────────────────────────────────
        with gr.Tab("🃏 Flashcard Generator"):
            gr.Markdown("Enter a topic and get instant revision flashcards.")
            f_topic  = gr.Textbox(label="Topic", placeholder="e.g. SQL joins, Newton's laws, photosynthesis")
            f_n      = gr.Slider(minimum=3, maximum=10, value=5, step=1, label="Number of flashcards")
            f_button = gr.Button("Generate Flashcards", variant="primary")
            f_output = gr.Markdown(label="Flashcards")
            f_button.click(fn=make_flashcards, inputs=[f_topic, f_n], outputs=f_output)
            gr.Examples(
                examples=["sorting algorithms", "the human digestive system", "Git commands", "supply and demand"],
                inputs=f_topic,
            )

        # ── Tab 3: Risk dashboard ──────────────────────────────────────────────
        with gr.Tab("📊 Dropout Risk Assessment"):
            gr.Markdown(
                "Enter a student's profile to predict their dropout risk "
                "using models trained on the OULAD dataset."
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Demographics**")
                    gender      = gr.Dropdown(["M", "F"], value="M", label="Gender")
                    age_band    = gr.Dropdown(["0-35", "35-55", "55<="], value="0-35", label="Age band")
                    region      = gr.Dropdown(CATEGORICAL_MAPS["region"], value="South East Region", label="Region")
                    disability  = gr.Dropdown(["N", "Y"], value="N", label="Disability")
                    imd_band    = gr.Dropdown(CATEGORICAL_MAPS["imd_band"], value="50-60%", label="IMD deprivation band")
                    highest_edu = gr.Dropdown(CATEGORICAL_MAPS["highest_education"], value="A Level or Equivalent", label="Highest education")

                with gr.Column():
                    gr.Markdown("**Enrolment**")
                    num_prev    = gr.Slider(0, 6, value=0, step=1, label="Previous attempts")
                    credits     = gr.Slider(30, 600, value=60, step=30, label="Studied credits")
                    date_reg    = gr.Slider(-200, 0, value=-30, step=1, label="Days registered before course start")

                    gr.Markdown("**Assessment performance**")
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
                    active_days = gr.Slider(0, 300,   value=60,   step=1,   label="Active study days")

                    gr.Markdown("**Result**")
                    risk_output = gr.Markdown()
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
    load_gemma()
    demo.launch(share=False)
