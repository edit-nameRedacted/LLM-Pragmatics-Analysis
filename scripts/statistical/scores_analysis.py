"""
Entropy Analysis: Does Context Type Narrow or Expand Answer Space?
==================================================================
Research question: Do direct / implicature / stochastic contexts
systematically narrow or expand a model's answer distribution relative
to no-context (NC) baseline, and does this track rater context-quality
scores in a continuous information-theoretic framework?

Analyses
--------
  B  Polynomial contrast  — tests DI < NC < II < SI ordering
  C  LME Δentropy ~ context_score + (1|question) — continuous IT test
  D  JSD(condition || NC) ~ context_score + (1|question)
  E  Non-parametric MI(context_score ; Δentropy)

Dependencies
------------
    pip install pandas numpy scipy statsmodels scikit-learn

Inputs (set paths in CONFIG)
------------------------------
    extended_metrics_{model}.csv     per-prompt entropy/uncertainty metrics
    QuestionContext_Scores.csv       human rater scores (AV column)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
import statsmodels.formula.api as smf
from sklearn.feature_selection import mutual_info_regression

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent.parent
MODELS       = ["qwen", "mistral", "deepseek", "llama"]
FOCAL_MODEL  = "deepseek"
ALPHA        = 0.05
N_PERM       = 5000
RANDOM_SEED  = 42

# Entropy metrics: label -> column name
ENTROPY_COLS = {
    "EAS":  "eas_mean",           # token-level entropy, averaged over sequence
    "Beam": "beam_score_entropy", # entropy over beam scores (output diversity)
    "CAA":  "caa_mean_cosine",    # cosine similarity of CAA displacement
                                  # (closer to 1 = less displaced from NC)
}

# 10 stochastic samples per prompt (for JSD)
SEQ_COLS = [f"token_entropy_sequence_sample_{i}" for i in range(10)]

# Polynomial contrast weights for DI < NC < II < SI
CONTRAST_WEIGHTS = {
    "direct_information":      -3,
    "no_context":              -1,
    "implicature_information": +1,
    "stochastic_information":  +3,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _model_folder(model: str) -> str:
    return "deepseek" if model.startswith("deepseek") else model


def load_model(model: str) -> pd.DataFrame:
    path = ROOT / "data" / "model" / _model_folder(model) / f"extended_metrics_{model}.csv"
    df = pd.read_csv(path)
    df["model"] = model
    # Prompts come in blocks of 4 (NC, DI, II, SI) - derive question ID
    df["question"] = (df["prompt_id"] - 1) // 4 + 1
    return df


def load_context_scores() -> pd.DataFrame:
    ctx = pd.read_csv(ROOT / "data" / "human" / "QuestionContext_Scores.csv")
    ctx = (ctx[["Num", "AV"]]
           .dropna(subset=["Num", "AV"])
           .rename(columns={"Num": "prompt_id", "AV": "context_score"}))
    ctx["prompt_id"] = ctx["prompt_id"].astype(int)
    return ctx


def parse_seq(s) -> float:
    """Semicolon-separated token entropy string -> row mean."""
    try:
        vals = [float(x) for x in str(s).split(";") if x.strip()]
        return float(np.mean(vals)) if vals else np.nan
    except Exception:
        return np.nan


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    for col in SEQ_COLS:
        if col in df.columns:
            df[col] = df[col].apply(parse_seq)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DELTA ENTROPY
# ─────────────────────────────────────────────────────────────────────────────
def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract NC baseline from each conditioned row, per question.
    Returns non-NC rows with delta_{NAME} columns appended.
    """
    present_cols = {col: f"nc_{name.lower()}"
                    for name, col in ENTROPY_COLS.items()
                    if col in df.columns}

    nc = (df[df["condition"] == "no_context"]
          [["question"] + list(present_cols.keys())]
          .rename(columns=present_cols))

    out = df[df["condition"] != "no_context"].merge(nc, on="question", how="inner")

    for name, col in ENTROPY_COLS.items():
        nc_col = f"nc_{name.lower()}"
        if col in out.columns and nc_col in out.columns:
            out[f"delta_{name}"] = out[col] - out[nc_col]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# JSD
# ─────────────────────────────────────────────────────────────────────────────
def compute_jsd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Jensen-Shannon Divergence between each condition's token-entropy
    sample distribution and the NC baseline for the same question.
    Uses the 10 stochastic samples as a discrete distribution (softmax-normed).
    Returns squared JSD (= JS divergence, bounded [0,1]).
    """
    seq_present = [c for c in SEQ_COLS if c in df.columns]
    if not seq_present:
        return pd.DataFrame()

    def softmax(x):
        x = np.nan_to_num(np.array(x, dtype=float))
        e = np.exp(x - x.max())
        return e / (e.sum() + 1e-12)

    records = []
    for q, grp in df.groupby("question"):
        nc_row = grp[grp["condition"] == "no_context"]
        if nc_row.empty:
            continue
        nc_dist = softmax(nc_row[seq_present].values[0])
        for _, row in grp[grp["condition"] != "no_context"].iterrows():
            cond_dist = softmax(row[seq_present].values)
            jsd_val   = float(jensenshannon(nc_dist, cond_dist) ** 2)
            records.append({
                "question":  q,
                "prompt_id": int(row["prompt_id"]),
                "condition": row["condition"],
                "model":     row["model"],
                "jsd":       jsd_val,
            })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS B - POLYNOMIAL CONTRAST
# ─────────────────────────────────────────────────────────────────────────────
def polynomial_contrast(df: pd.DataFrame, metric: str, label: str):
    """
    Linear polynomial contrast over ordered conditions.
    Permutation test with N_PERM shuffles.
    """
    grp_data  = {}
    grp_means = {}
    for cond in CONTRAST_WEIGHTS:
        vals = df[df["condition"] == cond][metric].dropna().values
        if len(vals) == 0:
            print(f"  {label:<10}: missing condition '{cond}' - skipping")
            return
        grp_data[cond]  = vals
        grp_means[cond] = vals.mean()

    observed = sum(CONTRAST_WEIGHTS[c] * grp_means[c] for c in CONTRAST_WEIGHTS)

    all_vals  = np.concatenate(list(grp_data.values()))
    cond_list = list(CONTRAST_WEIGHTS.keys())
    sizes     = [len(grp_data[c]) for c in cond_list]
    rng       = np.random.default_rng(RANDOM_SEED)

    perm_contrasts = []
    for _ in range(N_PERM):
        shuf = rng.permutation(all_vals)
        idx, pm = 0, {}
        for cond, sz in zip(cond_list, sizes):
            pm[cond] = shuf[idx:idx+sz].mean()
            idx += sz
        perm_contrasts.append(sum(CONTRAST_WEIGHTS[c] * pm[c] for c in CONTRAST_WEIGHTS))

    p_perm = (np.abs(perm_contrasts) >= np.abs(observed)).mean()
    sig    = "*" if p_perm < ALPHA else ""

    means_str = "  ".join(
        f"{c[:2].upper()}={grp_means[c]:.4f}" for c in cond_list)
    print(f"  {label:<10}  contrast={observed:+.4f}  p={p_perm:.4f}{sig}")
    print(f"             {means_str}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS C - LINEAR MIXED-EFFECTS MODEL
# ─────────────────────────────────────────────────────────────────────────────
def run_lme(data: pd.DataFrame, outcome: str, label: str):
    """Outcome ~ context_score + (1 | question)."""
    tmp = data[[outcome, "context_score", "question"]].dropna()
    if len(tmp) < 10:
        print(f"  {label:<14}: n={len(tmp)} - skipping")
        return None
    try:
        res = smf.mixedlm(
            f"{outcome} ~ context_score",
            data=tmp, groups=tmp["question"]
        ).fit(reml=True, method="lbfgs")
        b   = res.params["context_score"]
        se  = res.bse["context_score"]
        z   = res.tvalues["context_score"]
        p   = res.pvalues["context_score"]
        sig = "*" if p < ALPHA else ""
        n   = len(tmp)
        print(f"  {label:<14}  beta={b:+.4f}  SE={se:.4f}  z={z:+.3f}  p={p:.4f}{sig}  n={n}")
        return res
    except Exception as e:
        print(f"  {label:<14}: LME failed - {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS E - MUTUAL INFORMATION
# ─────────────────────────────────────────────────────────────────────────────
def mi_test(data: pd.DataFrame, outcome: str, label: str):
    """k-NN MI estimate (k=5) between context_score and outcome."""
    tmp = data[[outcome, "context_score"]].dropna()
    if len(tmp) < 10:
        return
    X  = tmp["context_score"].values.reshape(-1, 1)
    y  = tmp[outcome].values
    mi = mutual_info_regression(X, y, n_neighbors=5, random_state=RANDOM_SEED)[0]
    print(f"  {label:<14}  MI={mi:.4f} nats")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PER-MODEL RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_model(model: str, ctx: pd.DataFrame):
    sep = "=" * 62
    print(f"\n{sep}\n MODEL: {model.upper()}\n{sep}")

    df     = prepare(load_model(model))
    delta  = compute_deltas(df)
    merged = delta.merge(ctx, on="prompt_id", how="inner")

    # -- B. Polynomial contrast ----------------------------------------
    print("\n[B] Polynomial Contrast  (DI=-3, NC=-1, II=+1, SI=+3)")
    print("    (+) = SI expands most / DI narrows most;  (-) = reversed")
    for name, col in ENTROPY_COLS.items():
        if col in df.columns:
            polynomial_contrast(df, col, name)

    # -- C. LME: delta_entropy ~ context_score -------------------------
    print("\n[C] LME: delta_entropy ~ context_score + (1|question)")
    print("    beta < 0: higher-quality context -> more narrowing (negative delta)")
    for name in ENTROPY_COLS:
        dcol = f"delta_{name}"
        if dcol in merged.columns:
            run_lme(merged, dcol, name)

    # -- D. JSD --------------------------------------------------------
    jsd_df = compute_jsd(df)
    jsd_merged = pd.DataFrame()
    if not jsd_df.empty:
        jsd_merged = jsd_df.merge(ctx, on="prompt_id", how="inner")
        print("\n[D] LME: JSD(condition||NC) ~ context_score + (1|question)")
        print("    beta < 0: higher-quality context -> smaller distributional shift")
        run_lme(jsd_merged, "jsd", "JSD")
        tmp = jsd_merged[["jsd", "context_score"]].dropna()
        if len(tmp) > 2:
            r,  p  = pearsonr(tmp["jsd"], tmp["context_score"])
            rs, ps = spearmanr(tmp["jsd"], tmp["context_score"])
            print(f"  Pearson r={r:+.3f} p={p:.4f}  |  Spearman rho={rs:+.3f} p={ps:.4f}")

    # -- E. Mutual Information -----------------------------------------
    print("\n[E] Non-parametric MI (k-NN, k=5)")
    for name in ENTROPY_COLS:
        dcol = f"delta_{name}"
        if dcol in merged.columns:
            mi_test(merged, dcol, name)
    if not jsd_merged.empty:
        mi_test(jsd_merged, "jsd", "JSD")


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-MODEL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def multi_model_summary():
    sep = "=" * 62
    print(f"\n{sep}\n MULTI-MODEL SUMMARY - Polynomial Contrasts\n{sep}")
    for model in MODELS:
        print(f"\n  {model.upper()}")
        try:
            df = prepare(load_model(model))
            for name, col in ENTROPY_COLS.items():
                if col in df.columns:
                    polynomial_contrast(df, col, name)
        except FileNotFoundError:
            print(f"    File not found - skipping")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    ctx = load_context_scores()
    print(f"Context scores loaded: {ctx.shape[0]} rows")

    run_model(FOCAL_MODEL, ctx)
    multi_model_summary()