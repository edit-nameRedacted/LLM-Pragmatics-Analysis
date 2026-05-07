"""
run_mlm_analysis.py
===================

End-to-end multilevel-model analysis of the questions × context experiment.

Pipeline
--------
1. Load extended_metrics_<model>.csv for each of the 5 models.
2. Compute per-prompt response-shape measures from the 10 stored
   token_entropy_sequence samples (h_fb_quintile, halflife, exp decay fit, etc.).
3. Project quintile-binned trajectories onto per-model shape PCs.
4. Merge with prompt-level context ratings (pragmatic_relevance, ctx_q_cosine)
   and question-level rater data (answer_multiplicity, answer_multiplicity_sd).
5. Save analysis_base.csv (long format, one row per prompt × model).
6. Fit scalar MLMs (8 DVs × 5 models = up to 40 cells) using cluster-robust OLS
   with question as the cluster. Save mlm_scalar_extended.csv.
7. Fit the same model at each transformer layer for the three layer-wise DVs
   (CAA L2, CAA cosine, beam pair-cosine). Save mlm_per_layer.csv.

Predictor specification
-----------------------
   pragmatic_within  = z-score of (pragmatic_relevance - question mean)
   sim_within        = z-score of (ctx_q_cosine - question mean)
   mult_z            = z-score of question-level answer_multiplicity
   mult_sd_z         = z-score of question-level answer_multiplicity_sd

   The within-Q centering separates "which context within this question" from
   "what kind of question is this." Mult and mult_sd are constant within a
   question, so they only carry between-question variance.

   Cluster-robust SEs use question as the cluster (15 clusters per model).

Notes on choices
----------------
- Each row's shape measures are computed per-sample (one shape value per
  sequence) and then averaged. The alternative — averaging the 10 sequences
  pointwise and computing one shape — confounds with sample-length termination.
- HalfLife is computed on each sequence's own length, not a truncated common
  length. This avoids the Qwen-specific bias where the averaged-then-HL method
  amplifies condition differences that are partially driven by termination.
- All shape measures are z-scored within model before regression.
- statsmodels MLM with random intercept on question often fails to converge at
  n=15 clusters with REML. Cluster-robust OLS gives identical point estimates
  for this balanced design and always converges.

Usage
-----
   python run_mlm_analysis.py [--data-dir DATA_DIR] [--out-dir OUT_DIR]

Outputs
-------
   <out-dir>/analysis_base.csv
   <out-dir>/mlm_scalar_extended.csv
   <out-dir>/mlm_per_layer.csv

Dependencies
------------
   pandas numpy scipy scikit-learn statsmodels
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Map display name → file basename. Files are read as f"extended_metrics_{key}.csv".
MODELS = {
    "Qwen":       "qwen",
    "DeepSeek":   "deepseek",
    "LLaMA":      "llama",
    "Mistral":    "mistral",
    "DS-V2-Lite": "deepseek_v2_lite",
}

# Question-level rater data: 15 questions × answer multiplicity rating
# (1 = single answer, 5 = many answers possible). SD across raters.
ANSWER_MULTIPLICITY = [4.22, 3.56, 2.78, 1.67, 3.78, 3.78, 4.22, 3.67, 4.33,
                      3.56, 3.22, 2.33, 3.56, 1.89, 3.11]
ANSWER_MULTIPLICITY_SD = [0.67, 1.01, 1.39, 0.71, 0.83, 0.83, 0.83, 0.71, 1.12,
                         1.24, 1.64, 1.12, 1.24, 1.27, 1.54]

# Scalar DVs for the per-row regression. Family is for output organization.
SCALAR_DVs = [
    ("caa_mean_l2",         "representation"),
    ("caa_mean_cosine",     "representation"),
    ("shape_PC2",           "output_shape"),
    ("exp_k",               "output_shape"),
    ("halflife_rel",        "output_shape"),
    ("log_h_fb_quintile",   "output_shape"),
    ("beam_score_entropy",  "output_convergence"),
    ("beam_sbert_cosine",   "output_convergence"),
]

# Layer-wise DVs (semicolon-delimited per-layer values in the source CSVs).
LAYER_DVs = ["caa_per_layer_l2", "caa_per_layer_cosine", "beam_per_layer_cosine"]

# Shape-measure column keys produced from each entropy sequence.
SHAPE_KEYS = [
    "h_fb_thirds", "log_h_fb_thirds", "h_fb_quintile", "log_h_fb_quintile",
    "halflife_rel", "peak_pos_rel", "peak_rate_pos", "slope", "mean_H", "seq_len",
    "median_pos_rel", "p25_pos_rel", "p75_pos_rel",
    "exp_H0", "exp_k", "exp_Hasymp", "exp_rmse",
    "Qb1", "Qb2", "Qb3", "Qb4", "Qb5",
]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def parse_seq(s) -> np.ndarray | None:
    """Parse a semicolon-delimited string of floats."""
    if pd.isna(s) or not isinstance(s, str):
        return None
    try:
        vals = [float(x) for x in s.split(";") if x.strip()]
        return np.asarray(vals) if vals else None
    except ValueError:
        return None


def parse_layer_matrix(df: pd.DataFrame, col: str) -> np.ndarray | None:
    """Stack a layer-wise column into an (n_rows, n_layers) array, NaN-padding short rows."""
    seqs = [parse_seq(s) for s in df[col]]
    valid = [s for s in seqs if s is not None]
    if not valid:
        return None
    n_layers = len(valid[0])
    out = np.full((len(seqs), n_layers), np.nan)
    for i, s in enumerate(seqs):
        if s is not None and len(s) == n_layers:
            out[i] = s
    return out


def zscore(x: np.ndarray | pd.Series) -> np.ndarray:
    """Standardize, returning NaN-filled array if SD is degenerate."""
    arr = np.asarray(x, dtype=float)
    sd = np.nanstd(arr)
    if sd < 1e-10:
        return np.full_like(arr, np.nan)
    return (arr - np.nanmean(arr)) / sd


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE-SHAPE MEASURES
# ─────────────────────────────────────────────────────────────────────────────

def fit_exp_decay(seq: np.ndarray) -> tuple[float, float, float, float]:
    """
    Fit H(t) = H_asymp + (H_0 - H_asymp) * exp(-k * t_normalized) to one entropy sequence.

    Returns (H_0, k, H_asymp, RMSE) — all NaN on failure.
    Time is normalized to [0, 1] so k is comparable across response lengths.
    """
    if seq is None or len(seq) < 12:
        return np.nan, np.nan, np.nan, np.nan
    t = np.linspace(0.0, 1.0, len(seq))

    def model(t, H0, k, Ha):
        return Ha + (H0 - Ha) * np.exp(-k * t)

    try:
        p0 = [seq[0], 2.0, np.mean(seq[-len(seq) // 4:])]
        popt, _ = curve_fit(model, t, seq, p0=p0, maxfev=2000,
                            bounds=([0, -10, 0], [20, 50, 20]))
        H0, k, Ha = popt
        rmse = float(np.sqrt(np.mean((seq - model(t, *popt)) ** 2)))
        return float(H0), float(k), float(Ha), rmse
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan, np.nan


def compute_shape_measures(seq: np.ndarray | None) -> dict[str, float]:
    """Compute the full shape-measure suite for one entropy sequence."""
    if seq is None or len(seq) < 12:
        return {}

    n = len(seq)
    total = float(np.sum(seq))
    out: dict[str, float] = {}

    # Front-vs-back ratios at thirds and quintiles
    t = n // 3
    q = n // 5
    first_t, last_t = float(np.mean(seq[:t])), float(np.mean(seq[-t:]))
    first_q, last_q = float(np.mean(seq[:q])), float(np.mean(seq[-q:]))

    out["h_fb_thirds"] = last_t / first_t if first_t > 1e-6 else np.nan
    out["log_h_fb_thirds"] = (np.log(last_t / first_t)
                              if first_t > 1e-6 and last_t > 1e-6 else np.nan)
    out["h_fb_quintile"] = last_q / first_q if first_q > 1e-6 else np.nan
    out["log_h_fb_quintile"] = (np.log(last_q / first_q)
                                if first_q > 1e-6 and last_q > 1e-6 else np.nan)

    # Cumulative-percentile positions (normalized to [0, 1])
    if total > 1e-6:
        cum = np.cumsum(seq)
        out["halflife_rel"] = (np.searchsorted(cum, 0.5 * total) + 1) / n
        out["p25_pos_rel"] = (np.searchsorted(cum, 0.25 * total) + 1) / n
        out["p75_pos_rel"] = (np.searchsorted(cum, 0.75 * total) + 1) / n
    else:
        out["halflife_rel"] = np.nan
        out["p25_pos_rel"] = np.nan
        out["p75_pos_rel"] = np.nan

    # Peak-uncertainty positions
    out["peak_pos_rel"] = float(np.argmax(seq) / n)
    median_above = np.where(seq >= np.median(seq))[0]
    out["median_pos_rel"] = float(np.median(median_above) / n) if len(median_above) else np.nan
    out["peak_rate_pos"] = float(np.argmax(np.abs(np.diff(seq))) / max(n - 1, 1))

    # Aggregate shape stats
    out["slope"] = float(np.polyfit(np.arange(n), seq, 1)[0])
    out["mean_H"] = float(np.mean(seq))
    out["seq_len"] = float(n)

    # Quintile bin means (used as PCA features)
    for i, chunk in enumerate(np.array_split(seq, 5)):
        out[f"Qb{i + 1}"] = float(np.mean(chunk))

    # Exponential decay fit
    H0, k, Ha, rmse = fit_exp_decay(seq)
    out["exp_H0"] = H0
    out["exp_k"] = k
    out["exp_Hasymp"] = Ha
    out["exp_rmse"] = rmse

    return out


def aggregate_shape_per_row(row: pd.Series) -> dict[str, float]:
    """
    Compute shape measures for each of the 10 entropy samples in this row,
    then return the mean of each measure across samples.

    Per-sample-then-average is preferred over average-then-per-sequence because
    sample lengths can vary by EOS termination, and pointwise averaging
    (which requires truncating to the shortest length) confounds shape with
    termination patterns.
    """
    accumulated = {k: [] for k in SHAPE_KEYS}
    for i in range(10):
        col = f"token_entropy_sequence_sample_{i}"
        if col not in row.index:
            continue
        measures = compute_shape_measures(parse_seq(row.get(col)))
        for k, v in measures.items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                accumulated[k].append(v)
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in accumulated.items()}


# ─────────────────────────────────────────────────────────────────────────────
# DATA ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def load_question_metadata() -> pd.DataFrame:
    """Build the 15-row question-level metadata table."""
    return pd.DataFrame({
        "question": range(1, 16),
        "answer_multiplicity": ANSWER_MULTIPLICITY,
        "answer_multiplicity_sd": ANSWER_MULTIPLICITY_SD,
    })


def load_prompt_metadata(data_dir: Path) -> pd.DataFrame:
    """Load prompt-level pragmatic relevance and ctx-q similarity ratings."""
    df = pd.read_csv(data_dir / "data_compiled_sim.csv")
    df = df[["prompt_id", "rater_scores", "context_question_similarity"]]
    return df.rename(columns={
        "rater_scores": "pragmatic_relevance",
        "context_question_similarity": "ctx_q_cosine",
    })


def build_analysis_base(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    """
    Load all 5 models, compute shape measures per prompt, attach metadata,
    fit per-model PCA on quintile bins, and save analysis_base.csv.
    """
    question_meta = load_question_metadata()
    prompt_meta = load_prompt_metadata(data_dir)

    columns_to_keep = [
        "prompt_id", "model", "question", "condition", "domain", "pairwise",
        "pragmatic_relevance", "ctx_q_cosine",
        "answer_multiplicity", "answer_multiplicity_sd",
        "eas_mean", "eas_sd", "eas_slope", "eas_skew", "eas_sparsity",
        "eas_early", "eas_mid", "eas_late", "eas_final_quarter",
        "beam_score_entropy", "beam_score_raw_std", "beam_score_raw_range",
        "beam_first_divergence_position", "beam_length_mean", "beam_length_sd",
        "beam_sbert_cosine",
        "caa_mean_l2", "caa_mean_cosine", "displacement_cosine_vs_direct",
        "caa_per_layer_l2", "caa_per_layer_cosine", "beam_per_layer_cosine",
    ] + SHAPE_KEYS

    per_model_frames = []
    for display_name, file_key in MODELS.items():
        path = data_dir / f"extended_metrics_{file_key}.csv"
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        print(f"  Processing {display_name}...")
        ext = pd.read_csv(path)

        # Reconstruct question_id from prompt_id (1-indexed, 4 prompts per question)
        ext["question"] = (ext["prompt_id"] - 1) // 4 + 1
        ext = ext.merge(prompt_meta, on="prompt_id", how="left")
        ext = ext.merge(question_meta, on="question", how="left")
        ext["model"] = display_name

        shape_rows = pd.DataFrame([aggregate_shape_per_row(r) for _, r in ext.iterrows()])
        for k in SHAPE_KEYS:
            ext[k] = shape_rows[k].values

        per_model_frames.append(ext[[c for c in columns_to_keep if c in ext.columns]].copy())

    base = pd.concat(per_model_frames, ignore_index=True)

    # Per-model PCA on quintile bins of the entropy trajectory.
    # Fit on non-NC rows (the conditions where context exists) and project
    # all rows including NC. Standardization is per-model so PC1 doesn't
    # become a "which model" axis.
    qbin_cols = ["Qb1", "Qb2", "Qb3", "Qb4", "Qb5"]
    pc_frames = []
    for display_name in MODELS.keys():
        df_m = base[base["model"] == display_name].copy()
        if df_m.empty:
            continue
        train = df_m[df_m["condition"] != "no_context"][qbin_cols].dropna()
        if len(train) < 5:
            df_m["shape_PC1"] = np.nan
            df_m["shape_PC2"] = np.nan
            df_m["shape_PC3"] = np.nan
            pc_frames.append(df_m)
            continue
        mu, sd = train.mean(), train.std()
        train_z = (train - mu) / sd
        pca = PCA(n_components=3).fit(train_z)
        all_z = (df_m[qbin_cols] - mu) / sd
        valid = all_z.notna().all(axis=1)
        scores = np.full((len(df_m), 3), np.nan)
        scores[valid] = pca.transform(all_z[valid])
        df_m["shape_PC1"] = scores[:, 0]
        df_m["shape_PC2"] = scores[:, 1]
        df_m["shape_PC3"] = scores[:, 2]
        print(f"    {display_name} PCA explained variance: "
              f"PC1={pca.explained_variance_ratio_[0]:.0%} "
              f"PC2={pca.explained_variance_ratio_[1]:.0%} "
              f"PC3={pca.explained_variance_ratio_[2]:.0%}")
        pc_frames.append(df_m)

    base = pd.concat(pc_frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    base.to_csv(out_dir / "analysis_base.csv", index=False)
    print(f"  Saved: {out_dir / 'analysis_base.csv'} ({len(base)} rows)")
    return base


# ─────────────────────────────────────────────────────────────────────────────
# MULTILEVEL MODELS
# ─────────────────────────────────────────────────────────────────────────────

def prepare_predictors(sub: pd.DataFrame) -> pd.DataFrame:
    """Add the four standardized predictor columns required by the regression."""
    sub = sub.copy()
    for c in ["pragmatic_relevance", "ctx_q_cosine"]:
        q_mean = sub.groupby("question")[c].transform("mean")
        sub[c + "_within"] = zscore(sub[c] - q_mean)
    sub["mult_z"] = zscore(sub["answer_multiplicity"])
    sub["mult_sd_z"] = zscore(sub["answer_multiplicity_sd"])
    return sub


def fit_one_cell(data: pd.DataFrame, dv_col: str) -> dict | None:
    """
    Fit cluster-robust OLS for a single (model, DV) cell.

    Returns coefficient dict, or None if not enough data.
    """
    cols = ["pragmatic_relevance_within", "ctx_q_cosine_within", "mult_z", "mult_sd_z"]
    work = data.dropna(subset=[dv_col] + cols).copy()
    if len(work) < 15:
        return None
    work["dv_z"] = zscore(work[dv_col])
    if np.nanstd(work["dv_z"]) < 1e-10:
        return None
    X = sm.add_constant(work[cols])
    try:
        fit = sm.OLS(work["dv_z"], X).fit(
            cov_type="cluster", cov_kwds={"groups": work["question"]})
    except Exception:
        return None
    return {
        "b_prag":    fit.params["pragmatic_relevance_within"],
        "p_prag":    fit.pvalues["pragmatic_relevance_within"],
        "b_sim":     fit.params["ctx_q_cosine_within"],
        "p_sim":     fit.pvalues["ctx_q_cosine_within"],
        "b_mult":    fit.params["mult_z"],
        "p_mult":    fit.pvalues["mult_z"],
        "b_mult_sd": fit.params["mult_sd_z"],
        "p_mult_sd": fit.pvalues["mult_sd_z"],
        "R2":        fit.rsquared,
        "n":         len(work),
    }


def run_scalar_mlm(base: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Fit MLM for each (model × scalar DV). Save to mlm_scalar_extended.csv."""
    non_nc = base[base["condition"] != "no_context"].copy()
    rows = []
    for display_name in MODELS.keys():
        sub = non_nc[non_nc["model"] == display_name]
        if sub.empty:
            continue
        sub = prepare_predictors(sub)
        for dv_col, family in SCALAR_DVs:
            if dv_col not in sub.columns:
                continue
            fit_result = fit_one_cell(sub, dv_col)
            if fit_result is None:
                continue
            rows.append({
                "model": display_name, "family": family, "DV": dv_col, **fit_result})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "mlm_scalar_extended.csv", index=False)
    print(f"  Saved: {out_dir / 'mlm_scalar_extended.csv'} ({len(df)} cells)")
    return df


def run_per_layer_mlm(base: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Fit MLM at every transformer layer for each (model × layer-wise DV).

    For layer L of DV, the per-row predictor uses that row's value at layer L.
    Save trajectory to mlm_per_layer.csv.
    """
    non_nc = base[base["condition"] != "no_context"].copy()
    rows = []
    for display_name in MODELS.keys():
        sub = non_nc[non_nc["model"] == display_name].reset_index(drop=True)
        if sub.empty:
            continue
        sub = prepare_predictors(sub)
        for layer_col in LAYER_DVs:
            if layer_col not in sub.columns:
                continue
            mat = parse_layer_matrix(sub, layer_col)
            if mat is None:
                continue
            n_layers = mat.shape[1]
            for L in range(n_layers):
                # Build a per-layer DV column on the fly
                layer_data = sub.copy()
                layer_data["layer_dv"] = mat[:, L]
                fit_result = fit_one_cell(layer_data, "layer_dv")
                if fit_result is None:
                    continue
                rows.append({
                    "model": display_name, "DV": layer_col, "layer": L, **fit_result})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "mlm_per_layer.csv", index=False)
    print(f"  Saved: {out_dir / 'mlm_per_layer.csv'} ({len(df)} layer-cells)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def format_beta(b: float, p: float) -> str:
    if pd.isna(b):
        return "    —   "
    star = "★★" if p < 0.01 else "★ " if p < 0.05 else "† " if p < 0.10 else "  "
    return f"{b:+.2f}{star}"


def print_scalar_summary(scalar_results: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("SCALAR MLM RESULTS")
    print("Cell: β (★★ p<.01, ★ p<.05, † p<.10) — cluster-robust SE on question")
    print("=" * 100)
    header = (f"{'DV':<22} {'Model':<12} {'prag_w':>9} {'sim_w':>9} "
              f"{'mult':>9} {'mult_sd':>9}    R²")
    for family in ["representation", "output_shape", "output_convergence"]:
        print(f"\n── {family.upper()} ──")
        print(header)
        sub = scalar_results[scalar_results["family"] == family]
        for _, r in sub.iterrows():
            print(f"{r['DV']:<22} {r['model']:<12} "
                  f"{format_beta(r['b_prag'], r['p_prag']):>9} "
                  f"{format_beta(r['b_sim'], r['p_sim']):>9} "
                  f"{format_beta(r['b_mult'], r['p_mult']):>9} "
                  f"{format_beta(r['b_mult_sd'], r['p_mult_sd']):>9}    {r['R2']:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/mnt/user-data/uploads"),
                        help="Directory containing extended_metrics_*.csv and "
                             "data_compiled_sim.csv")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/mnt/user-data/outputs"),
                        help="Where to write analysis_base.csv and MLM result CSVs")
    args = parser.parse_args()

    print("─" * 60)
    print("STEP 1/3: Building analysis_base.csv")
    print("─" * 60)
    base = build_analysis_base(args.data_dir, args.out_dir)

    print("\n" + "─" * 60)
    print("STEP 2/3: Scalar MLM")
    print("─" * 60)
    scalar_results = run_scalar_mlm(base, args.out_dir)
    print_scalar_summary(scalar_results)

    print("\n" + "─" * 60)
    print("STEP 3/3: Per-layer MLM")
    print("─" * 60)
    run_per_layer_mlm(base, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
