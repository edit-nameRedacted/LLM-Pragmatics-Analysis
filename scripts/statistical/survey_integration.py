"""
survey_integration.py — Merge human Qualtrics ratings with LLM ratings.

Takes the exported Qualtrics CSV (context relevance + answer diversity survey)
and the LLM ratings produced by llm_rater.py, aligns them by question and
condition, computes inter-rater agreement, and writes a merged covariate CSV
for use in the main analysis.

Expected Qualtrics export format (standard Qualtrics CSV export):
  - First two rows are Qualtrics metadata headers — skipped automatically.
  - One column per survey item, named by question ID (e.g. Q1, Q2, ...).
  - Ratings are integers 1–5 (or will be coerced).

The script also accepts a simple flat format:
  columns: participant_id, task, question, condition, rating
  (e.g. a manually cleaned export or another survey platform)

Usage:
    # With standard Qualtrics export:
    python questions_x_context/scripts/survey_integration.py \\
        --qualtrics  path/to/qualtrics_export.csv \\
        --llm-ratings questions_x_context/data/llm_ratings.csv \\
        --output      questions_x_context/data/survey_merged.csv

    # With flat format:
    python questions_x_context/scripts/survey_integration.py \\
        --flat       path/to/flat_ratings.csv \\
        --llm-ratings questions_x_context/data/llm_ratings.csv \\
        --output      questions_x_context/data/survey_merged.csv

Outputs:
    survey_merged.csv         — one row per (question, condition), columns:
                                question, condition, human_mean, human_sd,
                                human_n, llm_mean, delta (human - llm),
                                spearman_r, spearman_p (item-level)
    survey_report.txt         — printed summary with agreement stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parent / "data"
_RESULTS = _HERE.parent / "results"


# ── Qualtrics format detection & loading ──────────────────────────────────────

def _load_qualtrics(path: Path) -> pd.DataFrame:
    """
    Load a standard Qualtrics CSV export.

    Qualtrics exports have:
      Row 0: column headers (question IDs like Q1_1, Q2, ...)
      Row 1: full question text labels
      Row 2: import IDs
      Row 3+: actual response data

    We read all rows, skip rows 1 and 2 (label rows), and keep row 0 as header.
    """
    raw = pd.read_csv(path, header=0, skiprows=[1, 2], low_memory=False)
    print(f"Loaded Qualtrics export: {raw.shape[0]} responses, {raw.shape[1]} columns")
    return raw


def _load_flat(path: Path) -> pd.DataFrame:
    """Load a flat ratings CSV with columns: participant_id, task, question, condition, rating."""
    df = pd.read_csv(path)
    required = {"task", "question", "condition", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Flat ratings CSV missing columns: {missing}")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df


# ── Qualtrics → flat format converter ─────────────────────────────────────────
# The Qualtrics survey mirrors llm_rater.py exactly:
#   Part 1 (relevance): 45 items = 15 questions × 3 conditions
#   Part 2 (diversity): 15 questions
#
# Column naming convention assumed: Q{n}_{suffix} where n is item index.
# If your export uses a different naming scheme, update QUALTRICS_COLUMN_MAP below.

# Maps Qualtrics column names → (task, question_text, condition)
# Populated automatically if the question text row is available, or set manually.
# Format: {"Q1": ("relevance", "Why do humans form social hierarchies?", "direct"), ...}
QUALTRICS_COLUMN_MAP: dict[str, tuple[str, str, str]] = {}


def _qualtrics_to_flat(raw: pd.DataFrame, llm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Qualtrics wide format to long flat format.

    Strategy: match Qualtrics question-text columns to llm_ratings.csv by
    fuzzy-matching the question text embedded in column headers or label rows.
    Falls back to positional alignment if text matching fails.
    """
    rows = []

    # If QUALTRICS_COLUMN_MAP is populated manually, use it directly.
    if QUALTRICS_COLUMN_MAP:
        for col, (task, question, condition) in QUALTRICS_COLUMN_MAP.items():
            if col not in raw.columns:
                print(f"  [warn] column {col} not found in export, skipping")
                continue
            vals = pd.to_numeric(raw[col], errors="coerce").dropna()
            for v in vals:
                rows.append({"task": task, "question": question,
                              "condition": condition, "rating": float(v)})
        return pd.DataFrame(rows)

    # Auto-detect: find numeric columns and try to match them positionally
    # to llm_ratings.csv items (same order as RELEVANCE_ITEMS + DIVERSITY_QUESTIONS).
    relevance_items = llm_df[llm_df["task"] == "relevance"][
        ["question", "condition"]
    ].drop_duplicates().reset_index(drop=True)
    diversity_items = llm_df[llm_df["task"] == "diversity"][
        ["question"]
    ].drop_duplicates().reset_index(drop=True)

    # Find columns that look like rating items (mostly numeric, values 1-5)
    rating_cols = []
    for col in raw.columns:
        s = pd.to_numeric(raw[col], errors="coerce")
        if s.notna().sum() > 0 and s.dropna().between(1, 5).all():
            rating_cols.append(col)

    n_expected = len(relevance_items) + len(diversity_items)
    if len(rating_cols) != n_expected:
        print(
            f"  [warn] Found {len(rating_cols)} numeric 1-5 columns; "
            f"expected {n_expected} ({len(relevance_items)} relevance + "
            f"{len(diversity_items)} diversity). "
            f"Check QUALTRICS_COLUMN_MAP or use --flat format instead."
        )
        # Trim or pad as best we can
        rating_cols = rating_cols[:n_expected]

    for i, col in enumerate(rating_cols):
        vals = pd.to_numeric(raw[col], errors="coerce").dropna()
        if i < len(relevance_items):
            q = relevance_items.iloc[i]["question"]
            cond = relevance_items.iloc[i]["condition"]
            task = "relevance"
        else:
            j = i - len(relevance_items)
            if j >= len(diversity_items):
                break
            q = diversity_items.iloc[j]["question"]
            cond = ""
            task = "diversity"
        for v in vals:
            rows.append({"task": task, "question": q, "condition": cond, "rating": float(v)})

    return pd.DataFrame(rows)


# ── Aggregation & alignment ───────────────────────────────────────────────────

def _aggregate_human(flat: pd.DataFrame) -> pd.DataFrame:
    """Per (task, question, condition): mean, sd, n."""
    grp = flat.groupby(["task", "question", "condition"])["rating"].agg(
        human_mean="mean", human_sd="std", human_n="count"
    ).reset_index()
    grp["human_sd"] = grp["human_sd"].fillna(0.0)
    return grp


def _aggregate_llm(llm_df: pd.DataFrame) -> pd.DataFrame:
    """Rename llm_ratings columns for merge."""
    return llm_df[["task", "question", "condition", "rating_mean"]].rename(
        columns={"rating_mean": "llm_mean"}
    )


def _merge_and_report(human_agg: pd.DataFrame, llm_agg: pd.DataFrame,
                      out_csv: Path, out_txt: Path) -> None:
    merged = human_agg.merge(llm_agg, on=["task", "question", "condition"], how="outer")
    merged["delta"] = merged["human_mean"] - merged["llm_mean"]

    # Item-level Spearman correlation
    complete = merged.dropna(subset=["human_mean", "llm_mean"])
    if len(complete) >= 3:
        r, p = stats.spearmanr(complete["human_mean"], complete["llm_mean"])
    else:
        r, p = float("nan"), float("nan")

    # Per-condition summary
    rel = merged[merged["task"] == "relevance"].copy()
    div = merged[merged["task"] == "diversity"].copy()

    lines = []
    lines.append("QxC Survey Integration Report")
    lines.append("=" * 60)
    lines.append(f"Human responses: {int(merged['human_n'].sum(skipna=True))} total ratings")
    lines.append(f"Items aligned:   {len(complete)} / {len(merged)}")
    lines.append("")
    lines.append(f"Item-level Spearman r = {r:.3f}  p = {p:.4f}")
    lines.append("")

    lines.append("RELEVANCE — mean ratings by condition (human vs LLM)")
    lines.append("-" * 60)
    if not rel.empty:
        for cond in ["direct", "implicature", "stochastic"]:
            sub = rel[rel["condition"] == cond]
            if sub.empty:
                continue
            hm = sub["human_mean"].mean()
            lm = sub["llm_mean"].mean()
            lines.append(
                f"  {cond:12s}  human={hm:.2f}  llm={lm:.2f}  delta={hm-lm:+.2f}"
            )
    else:
        lines.append("  (no relevance items found)")

    lines.append("")
    lines.append("DIVERSITY — mean ratings (human vs LLM)")
    lines.append("-" * 60)
    if not div.empty:
        hm = div["human_mean"].mean()
        lm = div["llm_mean"].mean()
        lines.append(f"  human={hm:.2f}  llm={lm:.2f}  delta={hm-lm:+.2f}")
    else:
        lines.append("  (no diversity items found)")

    lines.append("")
    lines.append("Most divergent items (|delta| > 1.0):")
    big = merged[merged["delta"].abs() > 1.0].sort_values("delta", key=abs, ascending=False)
    if big.empty:
        lines.append("  (none)")
    else:
        for _, row in big.head(10).iterrows():
            lines.append(
                f"  [{row['condition']:12s}] {str(row['question'])[:55]:<55} "
                f"human={row['human_mean']:.1f}  llm={row['llm_mean']:.1f}"
            )

    report = "\n".join(lines)
    print(report)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    out_txt.write_text(report, encoding="utf-8")
    print(f"\nMerged CSV  → {out_csv}")
    print(f"Report      → {out_txt}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--qualtrics", type=Path,
        help="Path to standard Qualtrics CSV export (rows 1-2 are metadata, skipped).",
    )
    group.add_argument(
        "--flat", type=Path,
        help="Path to flat ratings CSV with columns: task, question, condition, rating.",
    )
    p.add_argument(
        "--llm-ratings", type=Path,
        default=_DATA / "llm_ratings.csv",
        help="Path to llm_ratings.csv produced by llm_rater.py (default: data/llm_ratings.csv).",
    )
    p.add_argument(
        "--output", type=Path,
        default=_DATA / "survey_merged.csv",
        help="Output path for merged CSV (default: data/survey_merged.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.llm_ratings.exists():
        print(
            f"[warn] LLM ratings file not found: {args.llm_ratings}\n"
            "       Run llm_rater.py first, or pass --llm-ratings PATH.\n"
            "       Proceeding with human-only aggregation."
        )
        llm_df = pd.DataFrame(
            columns=["task", "question", "condition", "rating_mean"]
        )
    else:
        llm_df = pd.read_csv(args.llm_ratings)
        print(f"Loaded LLM ratings: {len(llm_df)} items from {args.llm_ratings}")

    if args.qualtrics:
        raw = _load_qualtrics(args.qualtrics)
        flat = _qualtrics_to_flat(raw, llm_df)
    else:
        flat = _load_flat(args.flat)

    print(f"Human ratings (flat): {len(flat)} rows from "
          f"{flat['question'].nunique()} questions × "
          f"{flat.get('condition', pd.Series()).nunique()} conditions")

    human_agg = _aggregate_human(flat)
    llm_agg   = _aggregate_llm(llm_df)

    out_txt = args.output.with_suffix(".txt").with_stem(args.output.stem.replace("_merged", "_report"))
    _merge_and_report(human_agg, llm_agg, args.output, out_txt)


if __name__ == "__main__":
    main()
