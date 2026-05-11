# Study Buddy: Student Performance Prediction & AI Tutor

We use the Open University Learning Analytics Dataset (OULAD) to predict student performance/risk and integrate a fine-tuned Gemma-based Study Buddy that answers questions and generates flashcards.

OULAD contains 32,593 students across 22 course presentations, with assessment results and more than 10.6 million VLE click records — making it a rich foundation for both classical ML and LLM-based educational tools.

---

## Target Variable

**Primary:** `final_result` — 4-class prediction (Pass / Distinction / Fail / Withdrawn)
**Secondary:** Binary dropout/risk flag — Withdrawn vs. all others (Pass + Distinction + Fail)

The binary risk framing is more actionable for early intervention and will be the main evaluation target.

---

## Project Structure

```
study-buddy-project/
├── data/
│   ├── raw/          # Original OULAD CSVs (never modified)
│   └── processed/    # Merged, cleaned, feature-engineered tables
├── notebooks/        # Exploratory analysis and step-by-step experiments
├── src/              # Reusable Python modules (features, training, eval)
├── models/           # Saved sklearn models and fine-tuned Gemma checkpoints
├── app/              # Gradio / Streamlit Study Buddy interface
└── reports/          # Figures, metrics, and final write-up artifacts
```

---

## Pipeline Overview

### Part 1 — ML Pipeline (OULAD)
1. **Ingest** — Load and inspect all 7 OULAD tables
2. **Merge** — Join on `id_student`, `code_module`, `code_presentation`
3. **Feature engineering** — Demographics, assessment scores, VLE engagement aggregates
4. **Modelling** — Logistic Regression → Random Forest → XGBoost baseline
5. **Evaluation** — Accuracy, F1, ROC-AUC, confusion matrix; early-week cut-off experiments

### Part 2 — Study Buddy Chatbot (Gemma)
1. **Dataset prep** — Curate Q&A + flashcard pairs from OULAD course metadata
2. **Fine-tune** — Gemma 2B via PEFT/LoRA with 4-bit quantization (bitsandbytes)
3. **Integrate** — Surface risk scores + chatbot in a single Gradio app

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download OULAD from https://analyse.kmi.open.ac.uk/open_dataset and place the CSV files in `data/raw/`.

---

## Key OULAD Tables

| File | Description | Key Join Fields |
|------|-------------|-----------------|
| `studentInfo.csv` | Demographics, final result | `id_student`, `code_module`, `code_presentation` |
| `studentAssessment.csv` | Scores per assessment | `id_student`, `id_assessment` |
| `studentRegistration.csv` | Enrolment / unregistration dates | `id_student`, `code_module`, `code_presentation` |
| `studentVle.csv` | Daily VLE click counts | `id_student`, `code_module`, `id_site` |
| `assessments.csv` | Assessment metadata (type, weight, date) | `id_assessment` |
| `courses.csv` | Module length in days | `code_module`, `code_presentation` |
| `vle.csv` | Activity type per site | `id_site`, `code_module` |
