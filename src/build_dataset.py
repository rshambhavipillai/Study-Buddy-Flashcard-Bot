"""Merge all OULAD tables and engineer features for modeling.

Output: data/processed/modeling_table.parquet

Run from project root:
    python src/build_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

KEY = ["id_student", "code_module", "code_presentation"]


def vle_features(student_vle: pd.DataFrame, vle: pd.DataFrame) -> pd.DataFrame:
    sv = student_vle.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")

    base = (
        sv.groupby(KEY)
        .agg(total_clicks=("sum_click", "sum"), active_days=("date", "nunique"))
        .reset_index()
    )

    # Top 8 activity types → per-student click totals
    top_types = sv["activity_type"].value_counts().head(8).index
    for atype in top_types:
        col = f"clicks_{atype}"
        sub = (
            sv[sv["activity_type"] == atype]
            .groupby(KEY)["sum_click"]
            .sum()
            .reset_index(name=col)
        )
        base = base.merge(sub, on=KEY, how="left")

    type_cols = [c for c in base.columns if c.startswith("clicks_")]
    base[type_cols] = base[type_cols].fillna(0)
    return base


def assessment_features(
    student_assess: pd.DataFrame, assessments: pd.DataFrame
) -> pd.DataFrame:
    sa = student_assess.merge(
        assessments[
            ["id_assessment", "code_module", "code_presentation",
             "assessment_type", "weight", "date"]
        ],
        on="id_assessment",
        how="left",
    )
    sa["score"] = sa["score"].fillna(0)  # non-submission → 0
    sa["days_early"] = sa["date"] - sa["date_submitted"]  # positive = submitted early

    base = (
        sa.groupby(KEY)
        .agg(
            avg_score=("score", "mean"),
            max_score=("score", "max"),
            std_score=("score", "std"),
            num_submissions=("id_assessment", "count"),
            avg_days_early=("days_early", "mean"),
        )
        .reset_index()
    )
    base["std_score"] = base["std_score"].fillna(0)

    # Weighted average score (by assessment weight)
    sa["weighted_score"] = sa["score"] * sa["weight"]
    wsum = sa.groupby(KEY)[["weighted_score", "weight"]].sum()
    wsum["weighted_avg_score"] = wsum["weighted_score"] / wsum["weight"].replace(0, np.nan)
    base = base.merge(wsum[["weighted_avg_score"]].reset_index(), on=KEY, how="left")

    # Average score broken down by assessment type
    for atype in ["TMA", "CMA", "Exam"]:
        sub = (
            sa[sa["assessment_type"] == atype]
            .groupby(KEY)["score"]
            .mean()
            .reset_index(name=f"avg_{atype.lower()}_score")
        )
        base = base.merge(sub, on=KEY, how="left")

    return base


def registration_features(
    student_reg: pd.DataFrame, courses: pd.DataFrame
) -> pd.DataFrame:
    sr = student_reg.merge(courses, on=["code_module", "code_presentation"], how="left")

    sr["did_unregister"] = sr["date_unregistration"].notna().astype(int)
    sr["date_registration"] = sr["date_registration"].fillna(0)

    # Days enrolled: use unregistration date, else full module length
    sr["effective_end"] = sr["date_unregistration"].fillna(
        sr["module_presentation_length"]
    )
    sr["days_registered"] = sr["effective_end"] - sr["date_registration"]

    return sr[KEY + ["date_registration", "did_unregister", "days_registered"]]


def build() -> pd.DataFrame:
    print("Loading tables …")
    courses        = pd.read_csv(RAW / "courses.csv")
    assessments    = pd.read_csv(RAW / "assessments.csv")
    vle            = pd.read_csv(RAW / "vle.csv")
    student_info   = pd.read_csv(RAW / "studentInfo.csv")
    student_reg    = pd.read_csv(RAW / "studentRegistration.csv")
    student_assess = pd.read_csv(RAW / "studentAssessment.csv")
    student_vle    = pd.read_csv(RAW / "studentVle.csv")

    print("Engineering VLE features …")
    vle_feats = vle_features(student_vle, vle)

    print("Engineering assessment features …")
    assess_feats = assessment_features(student_assess, assessments)

    print("Engineering registration features …")
    reg_feats = registration_features(student_reg, courses)

    print("Merging …")
    df = student_info.copy()
    df["imd_band"] = df["imd_band"].fillna("Unknown")

    df = df.merge(reg_feats,    on=KEY, how="left")
    df = df.merge(assess_feats, on=KEY, how="left")
    df = df.merge(vle_feats,    on=KEY, how="left")

    # Students with zero VLE activity
    vle_num_cols = [
        c for c in df.columns
        if c in ("total_clicks", "active_days") or c.startswith("clicks_")
    ]
    df[vle_num_cols] = df[vle_num_cols].fillna(0)

    # Students with no recorded assessments
    assess_num_cols = [
        "avg_score", "max_score", "std_score", "num_submissions",
        "avg_days_early", "weighted_avg_score",
        "avg_tma_score", "avg_cma_score", "avg_exam_score",
    ]
    for col in assess_num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Encode targets
    result_order = {"Withdrawn": 0, "Fail": 1, "Pass": 2, "Distinction": 3}
    df["target_multiclass"] = df["final_result"].map(result_order)
    df["target_binary"] = (df["final_result"] == "Withdrawn").astype(int)

    print(f"\nShape: {df.shape}")
    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]
    if remaining.empty:
        print("No nulls remaining.")
    else:
        print(f"Remaining nulls:\n{remaining}")

    print(f"\nTarget distribution:\n{df['final_result'].value_counts()}")

    out = PROCESSED / "modeling_table.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved → {out}")
    return df


if __name__ == "__main__":
    build()
