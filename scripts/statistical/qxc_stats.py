"""
qxc_stats.py — Statistical analysis for the Questions × Context experiment.

Runs three analyses on pilot_summary.csv (output of qxc_main.py):

  1. Levene's test — homogeneity of variance across all four conditions,
     then pairwise for no_context vs direct_information.

  2. One-way repeated measures ANOVA — DV: mean_token_entropy, within-factor:
     condition (4 levels), repeated unit: question. Followed by two planned
     pairwise comparisons with directional predictions:
       - no_context vs stochastic_information  (H: nc > si)
       - direct_information vs stochastic_information  (H: di < si)

  3. Mixed-effects regression — DV: mean_token_entropy, continuous predictor:
     context_question_similarity, fixed factor: condition, random intercept:
     question. Fits on rows with context only (no_context excluded; similarity
     is undefined for that condition). Tests the CRUX prediction directly:
     does I(Context;Question) predict entropy reduction, and in which direction?

Output: console report + results/stats_summary.txt

Usage (from llm_entropy_study/):
    python questions_x_context/qxc_stats.py --model qwen
    python questions_x_context/qxc_stats.py --model llama
    python questions_x_context/qxc_stats.py --input questions_x_context/results/pilot_summary_qwen.csv
"""

from __future__ import annotations

import argparse
import textwrap
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import levene, ttest_rel
from statsmodels.formula.api import mixedlm
from statsmodels.stats.anova import AnovaRM

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"
# Defaults — overridden by --model or --input/--output flags
DEFAULT_CSV = _RESULTS / "pilot_summary_qwen.csv"
DEFAULT_OUT = _RESULTS / "stats_summary_qwen.txt"

CONDITIONS = [
    "no_context",
    "stochastic_information",
    "implicature_information",
    "direct_information",
]

COND_SHORT = {
    "no_context": "NC",
    "stochastic_information": "SI",
    "implicature_information": "II",
    "direct_information": "DI",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _hr(char: str = "─", width: int = 60) -> str:
    return char * width


def _section(title: str) -> str:
    return f"\n{_hr('═')}\n{title}\n{_hr('═')}"


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "†"
    return "ns"


def _load_and_validate(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["condition"] = (
        df["condition"].astype(str).str.strip().str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    missing = [c for c in CONDITIONS if c not in df["condition"].unique()]
    if missing:
        raise ValueError(f"Conditions missing from CSV: {missing}")
    if "mean_token_entropy" not in df.columns:
        raise ValueError("Column 'mean_token_entropy' not found in CSV.")
    return df


def _condition_groups(df: pd.DataFrame, col: str = "mean_token_entropy") -> dict[str, np.ndarray]:
    return {
        cond: df.loc[df["condition"] == cond, col].dropna().to_numpy()
        for cond in CONDITIONS
    }


# ── Analysis 1: Levene's test ──────────────────────────────────────────────────
def run_levenes(df: pd.DataFrame, out: StringIO) -> None:
    out.write(_section("1. LEVENE'S TEST — Homogeneity of Variance") + "\n")
    out.write(
        "Tests whether within-condition variance is equal across conditions.\n"
        "Violation (p < .05) means conditions differ in spread, not just mean.\n\n"
    )

    groups = _condition_groups(df)

    # ── 1a. Omnibus across all four conditions ─────────────────────────────────
    stat, p = levene(*[groups[c] for c in CONDITIONS])
    out.write("1a. All four conditions\n")
    out.write(f"    F({len(CONDITIONS)-1}, {sum(len(g) for g in groups.values())-len(CONDITIONS)}) "
              f"= {stat:.4f},  p = {p:.4f}  {_stars(p)}\n")
    if p < 0.05:
        out.write("    -> Variances are heterogeneous. Interpret ANOVA F-ratio with caution.\n")
    else:
        out.write("    -> Variances are homogeneous. ANOVA assumption satisfied.\n")

    # ── 1b. Pairwise: no_context vs direct_information ─────────────────────────
    out.write("\n1b. Pairwise: no_context vs direct_information\n")
    nc = groups["no_context"]
    di = groups["direct_information"]
    stat2, p2 = levene(nc, di)
    n_total = len(nc) + len(di)
    out.write(f"    F(1, {n_total-2}) = {stat2:.4f},  p = {p2:.4f}  {_stars(p2)}\n")
    out.write(
        f"    NC  n={len(nc)}  var={np.var(nc, ddof=1):.6f}\n"
        f"    DI  n={len(di)}  var={np.var(di, ddof=1):.6f}\n"
    )
    if p2 < 0.05:
        out.write(
            "    -> These two conditions have significantly different variance.\n"
            "    Context does not merely shift the mean — it also changes response spread.\n"
        )
    else:
        out.write("    -> Variance is comparable between NC and DI.\n")


# ── Analysis 2: Repeated measures ANOVA + planned comparisons ─────────────────
def run_rm_anova(df: pd.DataFrame, out: StringIO) -> None:
    out.write(_section("2. REPEATED MEASURES ANOVA + PLANNED COMPARISONS") + "\n")
    out.write(
        "DV: mean_token_entropy\n"
        "Within factor: condition (4 levels)\n"
        "Repeated unit: question\n\n"
    )

    # ── Build balanced long-format subset ─────────────────────────────────────
    # Keep only questions that appear in all four conditions.
    q_counts = df.groupby("question")["condition"].nunique()
    complete_qs = q_counts[q_counts == len(CONDITIONS)].index.tolist()

    if len(complete_qs) == 0:
        out.write(
            "[WARNING] No question appears in all four conditions. "
            "Repeated measures ANOVA requires a balanced design.\n"
            "Check that your prompts CSV covers all four conditions per question.\n"
        )
        return

    dropped = df["question"].nunique() - len(complete_qs)
    if dropped > 0:
        out.write(
            f"[NOTE] {dropped} question(s) excluded: not present in all 4 conditions.\n"
            f"       {len(complete_qs)} question(s) retained for RM ANOVA.\n\n"
        )

    rm_df = df[df["question"].isin(complete_qs)][
        ["question", "condition", "mean_token_entropy"]
    ].dropna(subset=["mean_token_entropy"]).copy()

    # ── Descriptives ──────────────────────────────────────────────────────────
    out.write("Descriptives (mean_token_entropy):\n")
    desc = (
        rm_df.groupby("condition")["mean_token_entropy"]
        .agg(n="count", mean="mean", sd="std")
        .reindex(CONDITIONS)
    )
    for cond, row in desc.iterrows():
        short = COND_SHORT.get(cond, cond)
        out.write(f"  {cond:<28s} ({short}):  n={int(row['n'])}  M={row['mean']:.4f}  SD={row['sd']:.4f}\n")
    out.write("\n")

    # ── RM ANOVA ──────────────────────────────────────────────────────────────
    try:
        rm = AnovaRM(
            data=rm_df,
            depvar="mean_token_entropy",
            subject="question",
            within=["condition"],
        ).fit()
        table = rm.anova_table
        F = float(table["F Value"].iloc[0])
        df_num = float(table["Num DF"].iloc[0])
        df_den = float(table["Den DF"].iloc[0])
        p_rm = float(table["Pr > F"].iloc[0])
        out.write("RM ANOVA result:\n")
        out.write(
            f"  F({int(df_num)}, {int(df_den)}) = {F:.4f},  p = {p_rm:.4f}  {_stars(p_rm)}\n"
        )
        if p_rm < 0.05:
            out.write("  -> Significant main effect of condition on mean_token_entropy.\n")
        else:
            out.write(
                "  -> No significant main effect detected. Note: pilot N may lack power.\n"
            )
    except Exception as e:
        out.write(f"[ERROR] AnovaRM failed: {e}\n")
        out.write("  Falling back to descriptive comparison only.\n")

    # ── Planned pairwise comparisons ──────────────────────────────────────────
    out.write("\nPlanned pairwise comparisons (paired t-tests, Bonferroni k=2):\n")
    out.write("  Bonferroni-adjusted α = 0.025 for 2 planned comparisons.\n\n")

    alpha_corrected = 0.05 / 2

    def _paired_comparison(
        label: str, cond_a: str, cond_b: str, direction: str
    ) -> None:
        """direction: 'a > b' or 'a < b' — the directional prediction."""
        # Align on question
        wide = rm_df.pivot(index="question", columns="condition", values="mean_token_entropy")
        if cond_a not in wide.columns or cond_b not in wide.columns:
            out.write(f"  {label}: insufficient data.\n")
            return
        paired = wide[[cond_a, cond_b]].dropna()
        if len(paired) < 2:
            out.write(f"  {label}: n < 2 complete pairs — cannot compute.\n")
            return
        a_vals = paired[cond_a].to_numpy()
        b_vals = paired[cond_b].to_numpy()
        # Two-tailed t-test (report one-tailed p for directional prediction)
        t, p_two = ttest_rel(a_vals, b_vals)
        p_one = p_two / 2
        # Check direction matches prediction
        direction_confirmed = (
            (direction == "a > b" and np.mean(a_vals) > np.mean(b_vals))
            or (direction == "a < b" and np.mean(a_vals) < np.mean(b_vals))
        )
        sig_one = p_one < alpha_corrected and direction_confirmed
        out.write(f"  {label}\n")
        out.write(
            f"    Prediction: {cond_a} {direction.replace('a', COND_SHORT[cond_a]).replace('b', COND_SHORT[cond_b])}\n"
        )
        out.write(
            f"    M_a={np.mean(a_vals):.4f}  M_b={np.mean(b_vals):.4f}  "
            f"diff={np.mean(a_vals)-np.mean(b_vals):+.4f}\n"
        )
        out.write(
            f"    t({len(paired)-1}) = {t:.4f},  p(two-tailed) = {p_two:.4f},  "
            f"p(one-tailed) = {p_one:.4f}  {_stars(p_one)}\n"
        )
        if sig_one:
            out.write(
                f"    -> Significant in predicted direction (Bonferroni α={alpha_corrected}).\n"
            )
        elif direction_confirmed:
            out.write(
                f"    -> Trend in predicted direction but not significant at α={alpha_corrected}.\n"
            )
        else:
            out.write(
                f"    -> Effect in OPPOSITE direction to prediction.\n"
            )
        out.write("\n")

    _paired_comparison(
        "H1: no_context > stochastic_information",
        cond_a="no_context",
        cond_b="stochastic_information",
        direction="a > b",
    )
    _paired_comparison(
        "H2: direct_information < stochastic_information",
        cond_a="direct_information",
        cond_b="stochastic_information",
        direction="a < b",
    )


# ── Analysis 3: Mixed-effects regression ──────────────────────────────────────
def run_mixed_effects(df: pd.DataFrame, out: StringIO) -> None:
    out.write(_section("3. MIXED-EFFECTS REGRESSION") + "\n")
    out.write(
        "DV: mean_token_entropy\n"
        "Fixed effects: context_question_similarity (continuous) + condition (factor)\n"
        "Random effect: question (random intercept)\n"
        "Sample: rows WITH context only (no_context excluded — similarity undefined).\n\n"
    )

    if "context_question_similarity" not in df.columns:
        out.write(
            "[ERROR] 'context_question_similarity' column not found.\n"
            "Run sbert_similarity.py first to generate prompts_v5_sim.csv.\n"
        )
        return

    # ── Subset: context rows only ─────────────────────────────────────────────
    ctx_df = df[df["condition"] != "no_context"].copy()
    ctx_df = ctx_df.dropna(subset=["context_question_similarity", "mean_token_entropy"])

    if len(ctx_df) < 4:
        out.write(
            f"[WARNING] Only {len(ctx_df)} usable rows after filtering. "
            "Model may not converge — collect more data.\n"
        )

    # Reference level: stochastic_information (lowest theoretical I(C;Q))
    ctx_df["condition"] = pd.Categorical(
        ctx_df["condition"],
        categories=["stochastic_information", "implicature_information", "direct_information"],
    )

    # Descriptives on similarity by condition
    out.write("context_question_similarity by condition:\n")
    sim_desc = (
        ctx_df.groupby("condition", observed=True)["context_question_similarity"]
        .agg(n="count", mean="mean", sd="std")
        .reindex(["stochastic_information", "implicature_information", "direct_information"])
    )
    for cond, row in sim_desc.iterrows():
        short = COND_SHORT.get(cond, cond)
        out.write(f"  {cond:<28s} ({short}):  n={int(row['n'])}  M={row['mean']:.4f}  SD={row['sd']:.4f}\n")
    out.write("\n")

    # ── Fit model ─────────────────────────────────────────────────────────────
    formula = "mean_token_entropy ~ context_question_similarity + C(condition)"
    try:
        model = mixedlm(formula, data=ctx_df, groups=ctx_df["question"])
        result = model.fit(reml=True, method="lbfgs")
        out.write("Model: " + formula + "\n")
        out.write(f"Groups (random intercept): question  (n={ctx_df['question'].nunique()})\n")
        out.write(f"Observations: {len(ctx_df)}\n\n")

        # ── Fixed-effects table ────────────────────────────────────────────────
        out.write("Fixed effects:\n")
        fe = result.summary().tables[1]
        out.write(textwrap.indent(str(fe), "  ") + "\n\n")

        # ── Key coefficient: context_question_similarity ──────────────────────
        sim_key = "context_question_similarity"
        if sim_key in result.params.index:
            b = result.params[sim_key]
            se = result.bse[sim_key]
            z = result.tvalues[sim_key]
            p = result.pvalues[sim_key]
            out.write(f"KEY PREDICTOR — {sim_key}:\n")
            out.write(f"  β = {b:+.4f}  SE = {se:.4f}  z = {z:.4f}  p = {p:.4f}  {_stars(p)}\n")
            if p < 0.05:
                direction = "negative" if b < 0 else "positive"
                if b < 0:
                    out.write(
                        f"  -> Significant {direction} effect: higher I(Context;Question) similarity\n"
                        f"     is associated with LOWER mean token entropy.\n"
                        f"     Supports the Gricean side-information hypothesis.\n"
                    )
                else:
                    out.write(
                        f"  -> Significant {direction} effect: higher similarity is associated\n"
                        f"     with HIGHER entropy. Contradicts the Gricean prediction.\n"
                        f"     Consider whether context is increasing rather than constraining\n"
                        f"     the response space (e.g. introducing new topics).\n"
                    )
            else:
                out.write(
                    f"  -> Not significant (p = {p:.4f}). Similarity does not reliably\n"
                    f"     predict entropy reduction in this sample.\n"
                    f"     May reflect low pilot N or collinearity with condition.\n"
                )

        # ── Random-effects variance ────────────────────────────────────────────
        out.write(f"\nRandom effects:\n")
        out.write(f"  Question intercept variance: {result.cov_re.iloc[0, 0]:.6f}\n")
        out.write(f"  Residual variance:           {result.scale:.6f}\n")
        icc = result.cov_re.iloc[0, 0] / (result.cov_re.iloc[0, 0] + result.scale)
        out.write(f"  ICC (question):              {icc:.4f}\n")
        if icc > 0.1:
            out.write(
                "  -> Non-trivial between-question variance. "
                "Random intercept is well-motivated.\n"
            )
        else:
            out.write(
                "  -> Low ICC: question-level clustering is minimal in this sample.\n"
            )

    except Exception as e:
        out.write(f"[ERROR] Mixed model failed: {e}\n")
        out.write(
            "  Common causes: near-perfect collinearity between similarity and condition,\n"
            "  or too few groups for the random intercept to be estimated.\n"
            "  Try running with more prompts or collapsing to two conditions.\n"
        )


# ── Entry point ────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Statistical analysis for QxC experiment")
    p.add_argument(
        "--model",
        choices=["qwen", "llama", "mistral", "deepseek"],
        default="qwen",
        help=(
            "Which model's output to analyse. Sets default --input and --output paths "
            "to pilot_summary_{model}.csv / stats_summary_{model}.txt. "
            "Ignored if --input is supplied explicitly."
        ),
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Explicit path to pilot_summary CSV (overrides --model default).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit path for text output (overrides --model default).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve paths — explicit flags win; otherwise derive from --model
    input_csv = args.input or (_RESULTS / f"pilot_summary_{args.model}.csv")
    output_txt = args.output or (_RESULTS / f"stats_summary_{args.model}.txt")

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}\n"
            f"Run qxc_main.py --{args.model} first to generate it."
        )
    args.input = input_csv
    args.output = output_txt

    print(f"Input : {args.input}")
    print(f"Output: {args.output}")

    df = _load_and_validate(args.input)
    print(f"Loaded {len(df)} rows | conditions: {sorted(df['condition'].unique())}\n")

    buf = StringIO()
    buf.write(f"QxC Statistical Analysis\n")
    buf.write(f"Input: {args.input}\n")
    buf.write(f"Rows:  {len(df)}\n")

    run_levenes(df, buf)
    run_rm_anova(df, buf)
    run_mixed_effects(df, buf)

    buf.write("\n" + _hr("═") + "\n")
    buf.write("End of report.\n")

    report = buf.getvalue()
    print(report)

    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Report saved -> {args.output}")


if __name__ == "__main__":
    main()
