"""
cross_condition_contrasts.py — Discriminative cross-condition tests
====================================================================
Replaces three independent condition-vs-NC tests with two paired
discriminative tests over the three context conditions (DI, II, SI),
clustered by question:

  1. Contrast: II − (DI+SI)/2  per question, Wilcoxon signed-rank vs 0
     (tests whether II is set apart from the average of DI and SI)

  2. Pairwise paired Wilcoxon
     (DI vs II, II vs SI, DI vs SI), paired by question

Effect sizes
------------
  r_rb = matched-pairs rank-biserial correlation
       = (W_pos − W_neg) / (W_pos + W_neg),  range [-1, +1]
  Sign convention: for "a vs b", positive r_rb means a > b on average.

Multiple comparisons
--------------------
  Benjamini–Hochberg q-values are computed within each DV across all
  (model × test) cells. Per-DV is the looser correction; switch to
  global if you need to defend a stricter family.

Inputs
------
  analysis_base.csv with columns at minimum:
    model, question, condition, <DV columns>
  Conditions expected: direct_information, implicature_information,
  stochastic_information (NC rows are dropped).

Outputs
-------
  cross_condition_contrasts.csv  — one row per (model × DV × test)
  Printed summary table

Usage
-----
    python cross_condition_contrasts.py \\
        --data analysis_base.csv \\
        --dvs halflife_rel log_h_fb_quintile

Dependencies
------------
    pip install pandas numpy scipy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ── Defaults ──────────────────────────────────────────────────────────────────
DV_COLS = ["halflife_rel", "log_h_fb_quintile"]
MODELS = ["Qwen", "DeepSeek", "LLaMA", "Mistral", "DS-V2-Lite"]
CONTEXT_CONDITIONS = [
    "direct_information",
    "implicature_information",
    "stochastic_information",
]
COND_LABELS = {
    "direct_information": "DI",
    "implicature_information": "II",
    "stochastic_information": "SI",
}


# ── Effect size ───────────────────────────────────────────────────────────────
def rank_biserial(diffs: np.ndarray) -> float:
    """Matched-pairs rank-biserial r for one-sample Wilcoxon on `diffs`.

    Returns a sign-aware effect size in [-1, +1]. +1 means all paired
    differences are positive; -1 means all are negative; 0 means equal
    rank-mass on each side.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[diffs != 0]            # Wilcoxon drops zero diffs by default
    if len(diffs) == 0:
        return np.nan
    ranks = np.argsort(np.argsort(np.abs(diffs))) + 1
    w_pos = ranks[diffs > 0].sum()
    w_neg = ranks[diffs < 0].sum()
    denom = w_pos + w_neg
    return (w_pos - w_neg) / denom if denom > 0 else np.nan


# ── Data shaping ──────────────────────────────────────────────────────────────
def wide_per_question(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    """Pivot to rows=question, cols=condition, values=DV. Drops questions
    missing any of the three context conditions (Wilcoxon requires complete
    pairs)."""
    sub = df[df["condition"].isin(CONTEXT_CONDITIONS)][["question", "condition", dv]]
    wide = sub.pivot_table(
        index="question", columns="condition", values=dv, aggfunc="mean"
    )
    return wide.dropna(subset=CONTEXT_CONDITIONS)


# ── Tests ─────────────────────────────────────────────────────────────────────
def run_ii_vs_didsi(wide: pd.DataFrame) -> dict:
    """II − (DI+SI)/2 per question, two-sided Wilcoxon signed-rank vs 0."""
    contrast = (
        wide["implicature_information"]
        - 0.5 * (wide["direct_information"] + wide["stochastic_information"])
    ).dropna()
    n = len(contrast)
    out = {"test": "II - (DI+SI)/2", "n": n,
           "median": np.nan, "W": np.nan, "p": np.nan, "r_rb": np.nan}
    if n < 5:
        return out
    try:
        stat, p = wilcoxon(contrast.values, zero_method="wilcox",
                           alternative="two-sided")
        out.update(W=float(stat), p=float(p),
                   median=float(np.median(contrast)),
                   r_rb=float(rank_biserial(contrast.values)))
    except ValueError:
        pass
    return out


def run_pairwise(wide: pd.DataFrame, a: str, b: str) -> dict:
    """Paired two-sided Wilcoxon a vs b. diffs = a − b.
    Positive r_rb / median_diff means a > b on average."""
    diffs = (wide[a] - wide[b]).dropna()
    n = len(diffs)
    label = f"{COND_LABELS[a]} vs {COND_LABELS[b]}"
    out = {"test": label, "n": n,
           "median": np.nan, "W": np.nan, "p": np.nan, "r_rb": np.nan}
    if n < 5:
        return out
    try:
        stat, p = wilcoxon(diffs.values, zero_method="wilcox",
                           alternative="two-sided")
        out.update(W=float(stat), p=float(p),
                   median=float(np.median(diffs)),
                   r_rb=float(rank_biserial(diffs.values)))
    except ValueError:
        pass
    return out


def analyze_cell(df: pd.DataFrame, model: str, dv: str) -> list[dict]:
    """All four tests for one (model × DV) cell."""
    sub = df[df["model"] == model]
    if sub.empty or dv not in sub.columns:
        return []
    wide = wide_per_question(sub, dv)
    if wide.empty:
        return []
    results = [run_ii_vs_didsi(wide)]
    for a, b in [
        ("direct_information",      "implicature_information"),
        ("implicature_information", "stochastic_information"),
        ("direct_information",      "stochastic_information"),
    ]:
        results.append(run_pairwise(wide, a, b))
    return [{"model": model, "DV": dv, **r} for r in results]


# ── Multiple comparisons ──────────────────────────────────────────────────────
def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH q-values. NaN p-values pass through as NaN."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    out = np.full(n, np.nan)
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return out
    p_v = p[valid]
    order = np.argsort(p_v)
    q = p_v[order] * len(p_v) / (np.arange(len(p_v)) + 1)
    for i in range(len(q) - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    q_unsorted = np.empty_like(q)
    q_unsorted[order] = np.clip(q, 0, 1)
    out[valid] = q_unsorted
    return out


# ── Formatting ────────────────────────────────────────────────────────────────
def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "   n/a"
    if p < 0.001:
        return "  <.001"
    return f"  {p:.3f}"


def stars(q: float) -> str:
    if np.isnan(q):
        return "  "
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "* "
    if q < 0.10:
        return "† "
    return "  "


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="Path to analysis_base.csv")
    ap.add_argument("--out", type=Path,
                    default=Path("cross_condition_contrasts.csv"))
    ap.add_argument("--dvs", nargs="+", default=DV_COLS,
                    help="DV column names (default: halflife_rel log_h_fb_quintile)")
    ap.add_argument("--models", nargs="+", default=MODELS,
                    help="Model labels to include (must match the 'model' column)")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["condition"].isin(CONTEXT_CONDITIONS)].copy()

    # Sanity check — warn if any requested DV is missing or has no variance
    for dv in args.dvs:
        if dv not in df.columns:
            print(f"[WARN] DV '{dv}' not in {args.data.name}")
        elif df[dv].std() == 0:
            print(f"[WARN] DV '{dv}' has zero variance — tests will be uninformative")

    rows: list[dict] = []
    for m in args.models:
        if m not in df["model"].unique():
            print(f"[WARN] model '{m}' not found in data")
            continue
        for dv in args.dvs:
            rows.extend(analyze_cell(df, m, dv))
    res = pd.DataFrame(rows)

    if res.empty:
        print("\nNo results produced. Check --data, --dvs, --models.")
        return

    # BH within DV (per-family correction)
    res["q_BH"] = np.nan
    for dv, idx in res.groupby("DV").groups.items():
        res.loc[idx, "q_BH"] = benjamini_hochberg(res.loc[idx, "p"].values)

    res.to_csv(args.out, index=False)
    print(f"\nResults saved -> {args.out}\n")

    # Printed summary
    print("=" * 96)
    print(f"  {'Model':<11} {'Test':<18} {'n':>3}  "
          f"{'median':>9}  {'r_rb':>7}  {'p':>7}  {'q_BH':>7}  sig")
    print("=" * 96)
    for dv in args.dvs:
        sub = res[res["DV"] == dv]
        if sub.empty:
            continue
        print(f"\n  ── DV: {dv} " + "─" * max(0, 80 - len(dv)))
        for _, r in sub.iterrows():
            tag = stars(r["q_BH"]) if not np.isnan(r["q_BH"]) else stars(r["p"])
            print(f"  {r['model']:<11} {r['test']:<18} {int(r['n']):>3}  "
                  f"{r['median']:>+9.4f}  {r['r_rb']:>+7.3f}  "
                  f"{fmt_p(r['p'])}  {fmt_p(r['q_BH'])}  {tag}")

    print(
        "\nNotes:"
        "\n  • r_rb sign: for 'a vs b', positive = a > b on average (paired)."
        "\n  • For II−(DI+SI)/2: positive median = II > average of DI/SI."
        "\n  • |r_rb| guides: ~.10 negligible · ~.30 small · ~.50 moderate · >.70 large."
        "\n  • With n=15 paired observations, p<.05 typically requires |r_rb| ≳ .55,"
        "\n    so check effect size before getting excited about a significant p."
        "\n  • q_BH is per-DV BH correction. Switch to global if defending a stricter"
        "\n    family-wise error rate."
    )


if __name__ == "__main__":
    main()
