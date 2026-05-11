"""
Inspect all OULAD CSV files: print schema, row counts, sample rows, and
a basic null/dtype summary. Run from the project root:

    python src/inspect_oulad.py
"""

from pathlib import Path
import sys
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# All 7 OULAD tables in the order they are typically joined
TABLES = [
    "courses",
    "assessments",
    "vle",
    "studentInfo",
    "studentRegistration",
    "studentAssessment",
    "studentVle",
]


def inspect(name: str) -> pd.DataFrame | None:
    path = RAW / f"{name}.csv"
    if not path.exists():
        print(f"  [MISSING] {path.name} — place it in data/raw/ and re-run\n")
        return None

    df = pd.read_csv(path)
    sep = "=" * 60

    print(sep)
    print(f"  {name}.csv")
    print(sep)
    print(f"  Rows: {len(df):,}   Columns: {df.shape[1]}")
    print()

    # Column names, dtypes, null counts
    null_counts = df.isnull().sum()
    info = pd.DataFrame({
        "dtype":  df.dtypes,
        "nulls":  null_counts,
        "null_%": (null_counts / len(df) * 100).round(1),
    })
    print(info.to_string())
    print()

    # A few sample rows
    print("  Sample rows (head 3):")
    print(df.head(3).to_string(index=False))
    print()

    # Value counts for small-cardinality columns (good for target / categoricals)
    for col in df.columns:
        n_unique = df[col].nunique()
        if 1 < n_unique <= 10:
            print(f"  {col} value counts:")
            print(df[col].value_counts().to_string())
            print()

    return df


def main():
    if not RAW.exists():
        print(f"ERROR: Raw data folder not found at {RAW}")
        sys.exit(1)

    csv_files = list(RAW.glob("*.csv"))
    if not csv_files:
        print(
            f"No CSV files found in {RAW}\n"
            "Download OULAD from https://analyse.kmi.open.ac.uk/open_dataset\n"
            "and place the CSV files in data/raw/"
        )
        sys.exit(1)

    loaded = {}
    for name in TABLES:
        df = inspect(name)
        if df is not None:
            loaded[name] = df

    # Cross-table join key sanity check
    if "studentInfo" in loaded and "studentAssessment" in loaded:
        si = loaded["studentInfo"]
        sa = loaded["studentAssessment"]
        # id_student overlap
        overlap = sa["id_student"].isin(si["id_student"]).mean() * 100
        print(f"studentAssessment → studentInfo id_student match rate: {overlap:.1f}%")

    if "studentInfo" in loaded:
        print("\nTarget variable distribution (final_result):")
        print(loaded["studentInfo"]["final_result"].value_counts().to_string())

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
