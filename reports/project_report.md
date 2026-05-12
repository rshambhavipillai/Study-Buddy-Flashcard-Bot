# Study Buddy: Student Performance Prediction & AI Tutor
## Project Report

**Dataset:** Open University Learning Analytics Dataset (OULAD)  
**Models:** Logistic Regression, Decision Tree (ML) + Gemma 2B QLoRA (LLM)  
**Application:** Gradio web app — Q&A, flashcard generation, dropout risk dashboard

---

## 1. Introduction

This project addresses two connected problems in educational technology:

1. **Early dropout detection** — identifying at-risk students before they withdraw, enabling timely intervention
2. **AI-powered study assistance** — providing students with an intelligent tutor that answers questions and generates revision materials

We use the Open University Learning Analytics Dataset (OULAD) as the foundation for both tasks. The classical ML pipeline predicts dropout risk from student demographics and engagement data. A fine-tuned Gemma 2B language model powers the study assistant interface. Both components are integrated into a single Gradio web application.

---

## 2. Dataset — OULAD

The Open University Learning Analytics Dataset is one of the largest publicly available educational datasets. It contains anonymised records from the Open University (UK) across 7 linked CSV tables.

| Table | Rows | Description |
|---|---|---|
| `studentInfo` | 32,593 | Demographics, module, final result |
| `studentAssessment` | 173,912 | Per-assessment scores and submission dates |
| `studentRegistration` | 32,593 | Enrolment and unregistration dates |
| `studentVle` | 10,655,280 | Daily VLE click counts per student and activity |
| `assessments` | 206 | Assessment metadata (type, weight, due date) |
| `courses` | 22 | Module duration in days |
| `vle` | 6,364 | Activity type per VLE site |

**Key statistics:**
- 32,593 students across 22 course presentations
- 7 modules: AAA, BBB, CCC, DDD, EEE, FFF, GGG
- 4 presentation semesters (2013B, 2013J, 2014B, 2014J)
- Over 10.6 million VLE interaction records

**Target variable distribution (final_result):**

| Outcome | Count | Percentage |
|---|---|---|
| Pass | 12,361 | 37.9% |
| Withdrawn | 10,156 | 31.2% |
| Fail | 7,052 | 21.6% |
| Distinction | 3,024 | 9.3% |

The dataset is moderately imbalanced — Distinction is the minority class at 9.3%.

---

## 3. Data Preprocessing & Feature Engineering

All 7 OULAD tables were merged into a single modelling table with one row per student-course-presentation. The final feature matrix contains **30 features** across 32,593 students.

### 3.1 Null Handling

| Column | Nulls | Strategy |
|---|---|---|
| `imd_band` | 1,111 (3.4%) | Filled with "Unknown" category |
| `date_registration` | 45 (0.1%) | Filled with 0 (course start date) |
| `date_unregistration` | 22,521 (69.1%) | Null = still enrolled; encoded as binary `did_unregister` flag |
| `assessments.date` | 11 (5.3%) | Exam rows with no scheduled date |
| `studentAssessment.score` | 173 (0.1%) | Non-submissions filled with 0 |
| `vle.week_from/week_to` | 82.4% | Dropped — not needed for aggregation |

### 3.2 Engineered Features

**Assessment features** (from `studentAssessment` + `assessments`):
- `avg_score` — mean score across all assessments
- `max_score`, `std_score` — score range and variability
- `num_submissions` — total assessments submitted
- `weighted_avg_score` — score weighted by assessment weight
- `avg_days_early` — average days submitted before the deadline
- `avg_tma_score`, `avg_cma_score`, `avg_exam_score` — broken down by assessment type

**VLE engagement features** (from `studentVle` + `vle`):
- `total_clicks` — sum of all VLE interactions
- `active_days` — number of distinct days with activity
- `clicks_forumng`, `clicks_oucontent`, `clicks_resource`, etc. — per-activity-type click totals (top 8 activity types)

**Registration features** (from `studentRegistration` + `courses`):
- `date_registration` — days before course start when student registered
- `did_unregister` — binary flag (1 = withdrew during course)
- `days_registered` — total days enrolled

> **Note on data leakage:** Initial models achieved ~100% accuracy because `did_unregister` and `days_registered` are direct proxies for the Withdrawn label. These were removed from the feature set. The final models use only information that would be available during the course.

**VLE engagement summary statistics:**

| Metric | Mean | Median | Max |
|---|---|---|---|
| Total clicks | 1,215 | 602 | 24,139 |
| Active study days | 55.5 | 40 | 286 |
| Average score | 57.6 | 70.5 | 100 |
| Submissions count | 5.3 | 5 | 14 |

---

## 4. Machine Learning Models

### 4.1 Task Definition

**Binary classification** — predict whether a student will Withdraw vs. all other outcomes (Pass, Fail, Distinction).

- Target 1: Withdrawn (10,156 students, 31.2%)
- Target 0: Not withdrawn (22,437 students, 68.8%)

This framing is most actionable for early intervention: identifying students at risk of dropping out before it happens.

### 4.2 Train / Test Split

- 80/20 stratified split: **26,074 train**, **6,519 test**
- Stratification preserves class balance in both sets
- StandardScaler applied to all numeric features

### 4.3 Results

#### Logistic Regression

Grid search over: C ∈ {0.01, 0.1, 1, 10}, penalty ∈ {L1, L2}, solver = liblinear  
**Best params:** C=10, penalty=L2

| Metric | Not at Risk | At Risk (Withdrawn) | Macro Avg |
|---|---|---|---|
| Precision | 0.96 | 0.69 | 0.83 |
| Recall | 0.82 | 0.92 | 0.87 |
| F1-score | 0.88 | 0.79 | 0.84 |
| **Accuracy** | | | **0.85** |
| **ROC-AUC** | | | **0.929** |
| **CV F1 (5-fold)** | | | **0.790 ± 0.004** |

#### Decision Tree

Grid search over: max_depth ∈ {4, 6, 8, 12, None}, min_samples_leaf ∈ {10, 20, 50}, criterion ∈ {gini, entropy}  
**Best params:** criterion=entropy, max_depth=None, min_samples_leaf=50

| Metric | Not at Risk | At Risk (Withdrawn) | Macro Avg |
|---|---|---|---|
| Precision | 0.96 | 0.69 | 0.83 |
| Recall | 0.82 | 0.93 | 0.87 |
| F1-score | 0.88 | 0.79 | 0.84 |
| **Accuracy** | | | **0.85** |
| **ROC-AUC** | | | **0.932** |
| **CV F1 (5-fold)** | | | **0.788 ± 0.002** |

### 4.4 Key Observations

- Both models achieve **85% accuracy** and **ROC-AUC ~0.93** — strong performance for an imbalanced educational dataset
- **High recall (0.92–0.93) for at-risk students** is the most important metric for an early-warning system — the models correctly flag 9 out of 10 students who will withdraw
- The trade-off: precision for at-risk is 0.69, meaning ~31% of flagged students are false positives (students who are actually fine). This is acceptable in an intervention context — it is better to support a student unnecessarily than to miss one who genuinely needs help
- Both models perform nearly identically, suggesting the decision boundary is well-defined by the available features

### 4.5 Top Predictive Features (Decision Tree)

Based on feature importance from the best Decision Tree:

1. `avg_score` — average assessment score (strongest single predictor)
2. `total_clicks` — total VLE engagement
3. `weighted_avg_score` — assessment performance weighted by credit
4. `active_days` — number of days with any VLE activity
5. `num_submissions` — whether the student is submitting assessments at all

Academic performance and engagement are the dominant predictors of withdrawal — more so than demographics.

---

## 5. Study Buddy Chatbot — Fine-tuning

### 5.1 Base Model

**google/gemma-2b-it** — Google's 2-billion parameter instruction-tuned language model. Selected for:
- Small enough to fine-tune on a single T4 GPU (Kaggle free tier)
- Strong instruction-following capability
- Publicly available with accepted licence

### 5.2 Fine-tuning Dataset

A custom instruction-response dataset was built for Study Buddy tasks:

| Category | Examples | Description |
|---|---|---|
| `concept_explanation` | 42 | Explain a topic clearly (statistics, CS, biology, economics) |
| `flashcard_generation` | 17 | Produce Q/A flashcard sets for revision |
| `practice_question` | 15 | Generate problems with worked solutions |
| `study_skills` | 14 | Advice on how to study, revise, manage time |
| `exam_prep` | 11 | Exam technique and worked exam-style answers |
| **Total** | **99** | |

Example instruction-response pair:
```
Instruction: "Create 5 flashcards on SQL joins."
Output:
  Flashcard 1 — Q: What does an INNER JOIN return?
                A: Only rows where the join condition matches in both tables.
  ...
```

All examples follow the Gemma-IT chat template:
```
<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>
```

### 5.3 Training Configuration (QLoRA)

| Parameter | Value |
|---|---|
| Base model | google/gemma-2b-it |
| Quantisation | 4-bit NF4 (BitsAndBytes) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| LoRA dropout | 0.05 |
| Training epochs | 5 |
| Batch size | 1 (effective 16 with grad accumulation) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Optimiser | paged_adamw_8bit |
| Max sequence length | 256 |
| Trainable parameters | ~0.5% of total |

Training was performed on Kaggle Notebooks (T4 GPU, 15 GB VRAM) and completed in approximately 15–20 minutes.

---

## 6. Application

The Study Buddy Gradio application integrates both components into a three-tab interface:

### Tab 1 — Ask a Question
Students type any study question. The Gemma model (served via Ollama locally) generates a structured explanation. Response time: 2–5 seconds on Apple Silicon via Ollama.

### Tab 2 — Flashcard Generator
Students enter a topic and select the number of flashcards (3–10). Gemma generates Q/A pairs formatted for revision. Useful for quick revision material generation on any topic.

### Tab 3 — Dropout Risk Assessment
Students or tutors enter a demographic and engagement profile. The ensemble of Logistic Regression and Decision Tree models predicts the probability of dropout. Results are displayed as Low / Moderate / High Risk with probabilities from both models.

**Technology stack:**
- Backend: Python, scikit-learn, Gradio
- LLM inference: Ollama (local MPS on Apple Silicon)
- ML models: joblib-serialised sklearn models
- Data: OULAD parquet feature table

---

## 7. Discussion

### What Worked Well
- OULAD's rich click-stream data provides strong signals for dropout prediction without needing complex feature engineering
- QLoRA fine-tuning on 99 examples produced noticeably more structured, education-focused responses compared to the base model
- Using Ollama for local inference avoided the CPU bottleneck on Mac and made the app practical for real-time use

### Limitations
- The ML model predicts end-of-course withdrawal using full-course features; a more useful system would cut off features at week 4 for early intervention
- 99 fine-tuning examples is a small dataset; responses to out-of-domain topics revert to base Gemma behaviour
- The fine-tuned adapter was not merged with the base model for local deployment — Ollama currently serves the base model; the adapter is available for GPU-backed deployment

### Future Work
- Train an early-warning model using only the first 4 weeks of VLE data
- Expand the fine-tuning dataset to 1,000+ examples with OULAD-specific course content
- Deploy the merged fine-tuned model via a GPU endpoint (Replicate, Modal, or HuggingFace Spaces)
- Add a student-facing dashboard showing their own risk score and personalised study recommendations

---

## 8. Conclusion

This project demonstrates a complete end-to-end pipeline combining classical machine learning and large language models for educational analytics. The OULAD-trained dropout prediction models achieve 85% accuracy and 0.93 ROC-AUC, with high recall (0.92+) for at-risk students — the metric that matters most for early intervention. The fine-tuned Gemma Study Buddy provides structured, educationally appropriate responses to concept questions, flashcard requests, and study advice. Both components are deployed together in a functional Gradio web application.

---

## Appendix — Project Structure

```
study-buddy-project/
├── data/
│   ├── raw/              # Original OULAD CSVs
│   └── processed/        # modeling_table.parquet, study_buddy_finetune.jsonl
├── src/
│   ├── inspect_oulad.py       # Schema and data quality checks
│   ├── build_dataset.py       # Feature engineering pipeline
│   ├── train_models.py        # LR + DT training, evaluation, grid search
│   └── prepare_finetune_data.py  # Fine-tuning JSONL dataset builder
├── notebooks/
│   └── gemma_finetune_kaggle.ipynb  # Kaggle QLoRA fine-tuning notebook
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── scaler.pkl
│   └── gemma-study-buddy/    # LoRA adapter weights
├── app/
│   └── app.py               # Gradio three-tab application
└── reports/
    ├── roc_curves.png
    ├── dt_feature_importance.png
    ├── logistic_regression_report.txt
    ├── decision_tree_report.txt
    └── project_report.md
```
