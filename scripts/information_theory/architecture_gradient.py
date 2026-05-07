"""
Architecture Gradient Analysis: From Lexical to Pragmatic Encoding
===================================================================
Tests whether KV compression architecture (MLA vs GQA) and multi-token
prediction training (MTP, k>1) predict whether a model tracks pragmatic
relevance vs lexical similarity in its output entropy.

Theoretical framework
---------------------
Context conditions manipulate first-encoder quality I(C; W):
  DI  ~  I(C; W) high   — near-lossless encoding of communicative intent
  II  ~  I(C; W) medium — partial encoding, requires inference
  SI  ~  I(C; W) ~0     — near-zero encoding, noise in the channel
  NC  ~  I(C; W) = 0    — no first encoding

Two accuracy operationalisations:
  acc_rater = 6 − context_score  (pragmatic relevance; 1=relevant → 5=irrelevant)
  acc_sim   = context_question_similarity  (lexical/semantic overlap)

Hypothesis
----------
Models with MLA (low-rank KV compression) retain contextually predictive
features, tracking pragmatic relevance. Models with GQA (head-sharing only)
retain surface statistics, tracking lexical similarity. MTP amplifies MLA.

Predicted MI gradient:
  k=1, no MLA  →  MI(acc_sim ; Δentropy) > MI(acc_rater ; Δentropy)
  k=1, MLA     →  MI(acc_rater) begins to dominate
  k=2, MLA     →  MI(acc_rater) >> MI(acc_sim)

Models (ordered by architectural forward horizon)
-------------------------------------------------
  Mistral 7B        k=1, GQA/SWA, no MoE  — minimal post-training
  Qwen 2.5          k=1, GQA, no MoE      — heavy preference RLHF
  LLaMA 3.1         k=1, GQA, no MoE      — moderate RLHF
  DeepSeek-V2-Lite  k=1, MLA, MoE         — CRITICAL TEST: MLA without MTP
  DeepSeek-V3       k=2, MLA, MoE         — MLA + MTP

Outputs
-------
  architecture_gradient.png   — 5-panel figure
  Printed stats tables

Dependencies
------------
    pip install pandas numpy scipy scikit-learn matplotlib

Inputs (DATA_DIR configurable)
------------------------------
    extended_metrics_{model}.csv
    pilot_summary_{model}.csv       (ctx_tokens; not needed for all models)
    QuestionContext_Scores.csv
    prompts_v5_sim.csv
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = ROOT / "data"
OUT_DIR     = ROOT / "results" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42
ALPHA       = 0.05

SEQ_COLS = [f"token_entropy_sequence_sample_{i}" for i in range(10)]

COND_ORDER  = ["direct_information", "implicature_information", "stochastic_information"]
COND_LABELS = {"direct_information": "DI",
               "implicature_information": "II",
               "stochastic_information": "SI"}
COND_COLORS = {"direct_information":      "#2196F3",
               "implicature_information": "#FF9800",
               "stochastic_information":  "#E91E63"}

# Model metadata: architecture properties
ARCH = {
    "mistral":          {"MLA": 0, "MTP": 0, "MoE": 0, "RLHF_scale": "Low",  "k": 1},
    "qwen":             {"MLA": 0, "MTP": 0, "MoE": 0, "RLHF_scale": "High", "k": 1},
    "llama":            {"MLA": 0, "MTP": 0, "MoE": 0, "RLHF_scale": "Mod",  "k": 1},
    "deepseek_v2_lite": {"MLA": 1, "MTP": 0, "MoE": 1, "RLHF_scale": "Mod",  "k": 1},
    "deepseek":         {"MLA": 1, "MTP": 1, "MoE": 1, "RLHF_scale": "Mod",  "k": 2},
}

MODEL_COLORS = {
    "mistral":          "#F57C00",
    "qwen":             "#1976D2",
    "llama":            "#7B1FA2",
    "deepseek_v2_lite": "#00897B",
    "deepseek":         "#388E3C",
}

MODEL_LABELS = {
    "mistral":          "Mistral\n(k=1, no MLA)",
    "qwen":             "Qwen\n(k=1, no MLA)",
    "llama":            "LLaMA\n(k=1, no MLA)",
    "deepseek_v2_lite": "DS-V2-Lite\n(k=1, MLA)",
    "deepseek":         "DeepSeek-V3\n(k=2, MLA)",
}

# Analysis order: ascending architectural complexity
ORDERED_MODELS = ["mistral", "qwen", "llama", "deepseek_v2_lite", "deepseek"]


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
def _parse_seq(s) -> float:
    try:
        vals = [float(x) for x in str(s).split(";") if x.strip()]
        return float(np.mean(vals)) if vals else np.nan
    except Exception:
        return np.nan


def _parse_seq_full(s) -> list:
    """Return the full per-token entropy trajectory (not collapsed to mean)."""
    try:
        return [float(x) for x in str(s).split(";") if x.strip()]
    except Exception:
        return []


def _dtw_distance(a: list, b: list) -> float:
    """
    Dynamic Time Warping distance between two entropy trajectories.

    Finds the optimal alignment between sequences of different lengths before
    measuring distance — appropriate when NC and condition responses differ in
    length. Returns the normalised DTW cost (divided by alignment path length
    n+m) so comparisons across sequence-length pairs are on the same scale.
    Pure numpy, no external dependencies.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.nan
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m]) / (n + m)


def _mea_distance(a: list, b: list) -> float:
    """
    Mean Euclidean Area between two entropy trajectories.

    Truncates both sequences to the shorter length, then returns the mean
    absolute pointwise deviation |H_t^condition - H_t^NC|. Simpler than DTW
    and directly interpretable as average per-token entropy displacement.
    """
    n = min(len(a), len(b))
    if n == 0:
        return np.nan
    return float(np.mean(np.abs(
        np.array(a[:n], dtype=float) - np.array(b[:n], dtype=float)
    )))


def _model_folder(model: str) -> str:
    return "deepseek" if model.startswith("deepseek") else model


def _ext_path(model: str) -> Path:
    """Resolve extended metrics filepath for a model."""
    return DATA_DIR / "model" / _model_folder(model) / f"extended_metrics_{model}.csv"


def load_reference_data():
    """Load rater context scores and semantic similarity scores."""
    ctx_raw = pd.read_csv(DATA_DIR / "human" / "QuestionContext_Scores.csv")
    ctx = (ctx_raw[["Num", "AV"]]
           .dropna(subset=["Num", "AV"])
           .rename(columns={"Num": "prompt_id", "AV": "context_score"}))
    ctx["prompt_id"] = ctx["prompt_id"].astype(int)

    sim = (pd.read_csv(DATA_DIR / "human" / "prompts+SBERTsim_scores.csv")
           [["prompt_id", "context_question_similarity"]].dropna())
    sim["prompt_id"] = sim["prompt_id"].astype(int)

    return ctx, sim


def load_model(model: str, ctx: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    """
    Load extended metrics for one model, compute NC deltas and IB variables.

    Returns non-NC rows with:
      delta_{col}  — change from NC baseline per question
      acc_rater    — 6 − context_score (higher = more pragmatically relevant)
      acc_sim      — context_question_similarity (higher = more lexically similar)
      dtw_mean     — mean DTW distance between condition and NC entropy trajectories
                     across the 10 generation samples
      mea_mean     — mean MEA (mean absolute pointwise deviation) between condition
                     and NC entropy trajectories across the 10 generation samples
    """
    ext = pd.read_csv(_ext_path(model))

    # question column needed for groupby — add before copying for DTW/MEA
    ext["question"] = (ext["prompt_id"] - 1) // 4 + 1

    # Read full entropy trajectories BEFORE mean-collapsing, for DTW/MEA
    seq_present = [c for c in SEQ_COLS if c in ext.columns]
    ext_full = ext.copy()   # preserves raw semicolon-separated strings

    for col in SEQ_COLS:
        if col in ext.columns:
            ext[col] = ext[col].apply(_parse_seq)

    # Merge ctx_tokens if pilot summary available
    try:
        pilot = pd.read_csv(DATA_DIR / "model" / _model_folder(model) / f"pilot_summary_{model}.csv")
        ext   = ext.merge(pilot[["prompt_id", "ctx_tokens"]], on="prompt_id", how="left")
    except FileNotFoundError:
        ext["ctx_tokens"] = np.nan

    # NC baseline per question
    nc_cols = [c for c in ["eas_mean", "beam_score_entropy", "eas_sparsity",
                             "beam_sbert_cosine", "eas_early", "eas_mid", "eas_late",
                             "caa_mean_cosine"]
               if c in ext.columns]
    nc = (ext[ext["condition"] == "no_context"]
          [["question"] + nc_cols]
          .rename(columns={c: f"nc_{c}" for c in nc_cols}))

    out = ext[ext["condition"] != "no_context"].merge(nc, on="question", how="inner")
    out = out.merge(ctx, on="prompt_id", how="left")
    out = out.merge(sim, on="prompt_id", how="left")

    for col in nc_cols:
        out[f"delta_{col}"] = out[col] - out[f"nc_{col}"]

    out["acc_rater"] = 6 - out["context_score"]
    out["acc_sim"]   = out["context_question_similarity"]
    out["h_ctx"]     = out["ctx_tokens"].replace(0, np.nan)
    out["eff_Beam"]  = -out["delta_beam_score_entropy"] / out["h_ctx"]
    out["model"]     = model

    # ── DTW and MEA from full per-token entropy trajectories ──────────────────
    # Operates on ext_full (raw strings), NOT the mean-collapsed ext.
    # For each prompt: parse all 10 sample trajectories for NC and the condition,
    # compute DTW and MEA for each NC-sample vs condition-sample pair,
    # then store the mean distance across samples.
    trajectory_recs = []
    if seq_present:
        for q, grp in ext_full.groupby("question"):
            nc_rows = grp[grp["condition"] == "no_context"]
            if nc_rows.empty:
                continue
            # Parse all NC sample trajectories
            nc_trajs = [_parse_seq_full(nc_rows.iloc[0][col])
                        for col in seq_present]

            for _, row in grp[grp["condition"] != "no_context"].iterrows():
                cond_trajs = [_parse_seq_full(row[col]) for col in seq_present]

                dtw_vals, mea_vals = [], []
                for nc_t, cond_t in zip(nc_trajs, cond_trajs):
                    if nc_t and cond_t:
                        dtw_vals.append(_dtw_distance(nc_t, cond_t))
                        mea_vals.append(_mea_distance(nc_t, cond_t))

                trajectory_recs.append({
                    "prompt_id": int(row["prompt_id"]),
                    "dtw_mean":  float(np.nanmean(dtw_vals)) if dtw_vals else np.nan,
                    "mea_mean":  float(np.nanmean(mea_vals)) if mea_vals else np.nan,
                })

    if trajectory_recs:
        out = out.merge(pd.DataFrame(trajectory_recs), on="prompt_id", how="left")

    return out


def get_nc_beam_mean(model: str) -> float:
    df = pd.read_csv(_ext_path(model))
    return df[df["condition"] == "no_context"]["beam_score_entropy"].mean()


# ─────────────────────────────────────────────────────────────────
# ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────────
def compute_mi_pair(df: pd.DataFrame, metric_col: str, acc_col: str) -> float:
    """k-NN MI estimate between acc_col and metric_col."""
    tmp = df[[acc_col, metric_col]].dropna()
    if len(tmp) < 8:
        return 0.0
    return float(mutual_info_regression(
        tmp[acc_col].values.reshape(-1, 1),
        tmp[metric_col].values,
        n_neighbors=5, random_state=RANDOM_SEED)[0])


def compute_beta(df: pd.DataFrame, acc_col: str) -> float:
    """β = slope of dAccuracy/dComplexity across condition means."""
    pts_x, pts_y = [], []
    for cond in COND_ORDER:
        sub = df[df["condition"] == cond][[" beam_score_entropy".strip(), acc_col]].dropna()
        if sub.empty:
            sub = df[df["condition"] == cond][["beam_score_entropy", acc_col]].dropna()
        pts_x.append(sub["beam_score_entropy"].mean())
        pts_y.append(sub[acc_col].mean())
    if np.std(pts_x) < 1e-8:
        return np.nan
    return float(np.polyfit(pts_x, pts_y, 1)[0])


# ─────────────────────────────────────────────────────────────────
# PRINTED ANALYSES
# ─────────────────────────────────────────────────────────────────
def print_summary_table(datasets: dict):
    sep = "=" * 74
    print(f"\n{sep}")
    print("FULL 5-MODEL COMPARISON")
    print("Architecture: MLA | MTP | MoE | k (forward prediction horizon)")
    print(f"{sep}")
    print(f"\n{'Model':<20} {'MLA':>4} {'MTP':>4} {'MoE':>4} {'k':>3} | "
          f"{'MI(EAS×Rater)':>14} {'MI(EAS×Sim)':>12} | "
          f"{'β(Rater)':>9} {'β(Sim)':>8}  Dominant")
    print("-" * 74)
    for model in ORDERED_MODELS:
        df   = datasets[model]
        arch = ARCH[model]
        mi_r = compute_mi_pair(df, "delta_eas_mean", "acc_rater")
        mi_s = compute_mi_pair(df, "delta_eas_mean", "acc_sim")
        b_r  = compute_beta(df, "acc_rater")
        b_s  = compute_beta(df, "acc_sim")
        dom  = "Rater" if mi_r > mi_s else "Sim  "
        print(f"  {model:<18} {arch['MLA']:>4} {arch['MTP']:>4} {arch['MoE']:>4} "
              f"{arch['k']:>3} | {mi_r:>14.4f} {mi_s:>12.4f} | "
              f"{b_r:>+9.1f} {b_s:>+8.1f}  {dom}")


def print_v2lite_detail(df: pd.DataFrame):
    sep = "=" * 74
    print(f"\n{sep}")
    print("DEEPSEEK V3 — Detailed Results")
    print("Critical test: MLA=YES  MTP=NO  MoE=YES  k=1")
    print(f"{sep}")

    nb = get_nc_beam_mean("deepseek")
    print(f"\n[A] Rate-Distortion  (NC beam baseline = {nb:.5f})")
    for cond in COND_ORDER:
        sub = df[df["condition"] == cond]
        cx  = sub["beam_score_entropy"].mean()
        ar  = sub["acc_rater"].mean()
        as_ = sub["acc_sim"].mean()
        print(f"  {COND_LABELS[cond]}  cmplx={cx:.5f} ({cx-nb:+.6f})  "
              f"acc_rater={ar:.3f}  acc_sim={as_:.3f}")

    print("\n[B] MI(accuracy ; Δentropy / trajectory distance)")
    for m_col, m_lbl in [("delta_eas_mean",          "EAS Δ  "),
                          ("delta_beam_score_entropy", "Beam Δ "),
                          ("dtw_mean",                 "DTW    "),
                          ("mea_mean",                 "MEA    ")]:
        if m_col not in df.columns:
            continue
        for a_col, a_lbl in [("acc_rater", "Rater"), ("acc_sim", "Sim  ")]:
            tmp = df[[a_col, m_col]].dropna()
            if len(tmp) < 8:
                continue
            mi  = mutual_info_regression(tmp[a_col].values.reshape(-1, 1),
                                         tmp[m_col].values,
                                         n_neighbors=5, random_state=RANDOM_SEED)[0]
            r, p = pearsonr(tmp[a_col], tmp[m_col])
            sig  = "*" if p < ALPHA else ""
            print(f"  {m_lbl} × {a_lbl}  MI={mi:.4f}  r={r:+.3f}  p={p:.4f}{sig}")

    print("\n[C] Temporal EAS trajectory (ΔEAS vs NC baseline)")
    for cond in COND_ORDER:
        sub = df[df["condition"] == cond]
        vals = {}
        for p in ["early", "mid", "late"]:
            if f"delta_eas_{p}" in sub.columns:
                vals[p] = sub[f"delta_eas_{p}"].mean()
            elif f"eas_{p}" in sub.columns and f"nc_eas_{p}" in sub.columns:
                vals[p] = (sub[f"eas_{p}"] - sub[f"nc_eas_{p}"]).mean()
            else:
                vals[p] = np.nan
        traj = (sub["eas_late"] - sub["eas_early"]).mean() \
               if all(c in sub.columns for c in ["eas_late", "eas_early"]) else np.nan
        print(f"  {COND_LABELS[cond]}  "
              f"Δearly={vals['early']:+.4f}  Δmid={vals['mid']:+.4f}  "
              f"Δlate={vals['late']:+.4f}  late-early={traj:+.4f}")

    if "delta_caa_mean_cosine" in df.columns:
        print("\n[D] CAA representational displacement")
        for cond in COND_ORDER:
            sub = df[df["condition"] == cond]
            print(f"  {COND_LABELS[cond]}  "
                  f"Δcaa_cosine={sub['delta_caa_mean_cosine'].mean():+.5f}")

    print("\n[E] Beam semantic convergence")
    for cond in COND_ORDER:
        sub = df[df["condition"] == cond]
        if "beam_sbert_cosine" in sub.columns:
            print(f"  {COND_LABELS[cond]}  "
                  f"beam_sbert={sub['beam_sbert_cosine'].mean():.4f}  "
                  f"Δbeam_sbert={sub['delta_beam_sbert_cosine'].mean():+.4f}"
                  if "delta_beam_sbert_cosine" in sub.columns else
                  f"  {COND_LABELS[cond]}  beam_sbert={sub['beam_sbert_cosine'].mean():.4f}")


def print_gradient_test(datasets: dict):
    sep = "=" * 74
    print(f"\n{sep}")
    print("CRITICAL TEST: MI(EAS×Rater) gradient by architectural complexity")
    print("Prediction: monotonically increases with k and MLA presence")
    print(f"{sep}")
    print(f"\n{'Model':<20} {'k':>3} {'MLA':>4} {'MI(×Rater)':>12} {'MI(×Sim)':>10}  Dominant")
    for model in ORDERED_MODELS:
        df   = datasets[model]
        arch = ARCH[model]
        mi_r = compute_mi_pair(df, "delta_eas_mean", "acc_rater")
        mi_s = compute_mi_pair(df, "delta_eas_mean", "acc_sim")
        dom  = "Rater" if mi_r > mi_s else "Sim  "
        print(f"  {model:<18} {arch['k']:>3} {arch['MLA']:>4} "
              f"{mi_r:>12.4f} {mi_s:>10.4f}  {dom}")


# ─────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────
def plot_architecture_gradient(datasets: dict, out_path: str):
    """
    5-panel figure:
      Top-left (wide): MI gradient bar chart — the key result
      Top-right:       Rate-distortion operating points, all 5 models
      Bottom row:      Temporal EAS per condition (DI / II / SI)
    """
    fig = plt.figure(figsize=(18, 16))
    fig.patch.set_facecolor("#F5F5F5")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    # ── Panel 1: MI gradient (wide) ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.set_facecolor("white")
    ax1.spines[["top", "right"]].set_visible(False)

    mi_rater, mi_sim = [], []
    for model in ORDERED_MODELS:
        df = datasets[model]
        mi_rater.append(compute_mi_pair(df, "delta_eas_mean", "acc_rater"))
        mi_sim.append(compute_mi_pair(df, "delta_eas_mean", "acc_sim"))

    x_pos = np.arange(len(ORDERED_MODELS))
    w     = 0.35
    bars_r = ax1.bar(x_pos - w / 2, mi_rater, w,
                     color="#FF7043", alpha=0.85, label="MI × Rater (pragmatic)", zorder=3)
    bars_s = ax1.bar(x_pos + w / 2, mi_sim,   w,
                     color="#42A5F5", alpha=0.85, label="MI × Similarity (lexical)", zorder=3)

    for bar, val in zip(list(bars_r) + list(bars_s), mi_rater + mi_sim):
        if val > 0.01:
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.004,
                     f"{val:.3f}", ha="center", va="bottom",
                     fontsize=8, fontweight="bold")

    ax1.axvline(2.5, color="#9E9E9E", lw=1.5, ls="--", alpha=0.7)
    ax1.text(2.55, 0.21, "← GQA | MLA →", fontsize=8, color="#555")
    ax1.axvline(3.5, color="#388E3C", lw=1.5, ls="--", alpha=0.7)
    ax1.text(3.55, 0.21, "← k=1 | k=2 →", fontsize=8, color="#388E3C")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([MODEL_LABELS[m] for m in ORDERED_MODELS], fontsize=8.5)
    ax1.set_ylabel("Mutual Information (nats)", fontsize=9)
    ax1.set_title("IB Complexity I(X;T): MI(accuracy ; ΔEAS entropy) by Model\n"
                  "Ordered by architectural forward horizon k — orange=pragmatic, blue=lexical",
                  fontsize=10, pad=5, fontweight="bold")
    ax1.legend(fontsize=8.5, loc="upper left", framealpha=0.7)
    ax1.set_ylim(0, 0.28)

    # ── Panel 2: Rate-distortion, all 5 models ────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("white")
    ax2.spines[["top", "right"]].set_visible(False)

    for model in ORDERED_MODELS:
        df  = datasets[model]
        nb  = get_nc_beam_mean(model)
        pts = [(df[df["condition"] == c]["beam_score_entropy"].mean(),
                df[df["condition"] == c]["acc_rater"].mean())
               for c in COND_ORDER]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax2.scatter(xs, ys, color=MODEL_COLORS[model], s=70, zorder=5, alpha=0.85)
        ax2.plot(xs, ys, color=MODEL_COLORS[model], lw=1.2, alpha=0.5, zorder=4)
        ax2.scatter(nb, np.mean(ys), color=MODEL_COLORS[model],
                    s=40, marker="D", alpha=0.5, zorder=4)
        for (x, y), cond in zip(pts, COND_ORDER):
            ax2.annotate(COND_LABELS[cond], (x, y),
                         xytext=(2, 2), textcoords="offset points",
                         fontsize=6.5, color=MODEL_COLORS[model])

    legend_els = [Line2D([0], [0], color=MODEL_COLORS[m], lw=2,
                         label=m.replace("deepseek_v2_lite", "DS-V2-Lite")
                               .replace("deepseek", "DS-V3"))
                  for m in ORDERED_MODELS]
    ax2.legend(handles=legend_els, fontsize=7, loc="upper left", framealpha=0.7)
    ax2.set_xlabel("Complexity (beam entropy)", fontsize=8.5)
    ax2.set_ylabel("Accuracy (6 − rater score)", fontsize=8.5)
    ax2.tick_params(labelsize=7.5)
    ax2.set_title("Rate-Distortion Operating Points\n(rater accuracy) — all 5 models",
                  fontsize=9.5, pad=5, fontweight="bold")

    # ── Panels 3-5: Temporal EAS per condition ────────────────────
    for col_i, cond in enumerate(COND_ORDER):
        ax = fig.add_subplot(gs[1, col_i])
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="#AAAAAA", lw=1, ls="--")

        phases = ["early", "mid", "late"]
        x_pos  = np.arange(3)

        for model in ORDERED_MODELS:
            df  = datasets[model]
            sub = df[df["condition"] == cond]
            means, sems = [], []
            for p in phases:
                d_col = f"delta_eas_{p}"
                if d_col in sub.columns:
                    means.append(sub[d_col].mean())
                    sems.append(sub[d_col].sem())
                elif f"eas_{p}" in sub.columns and f"nc_eas_{p}" in sub.columns:
                    diff = sub[f"eas_{p}"] - sub[f"nc_eas_{p}"]
                    means.append(diff.mean())
                    sems.append(diff.sem())
                else:
                    means.append(np.nan)
                    sems.append(0)

            lbl = model.replace("deepseek_v2_lite", "DS-V2-Lite") \
                       .replace("deepseek", "DS-V3")
            ax.plot(x_pos, means, color=MODEL_COLORS[model], lw=2,
                    marker="o", markersize=5, label=lbl, zorder=4)
            ax.fill_between(x_pos,
                            np.array(means) - np.array(sems),
                            np.array(means) + np.array(sems),
                            color=MODEL_COLORS[model], alpha=0.12)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Early", "Mid", "Late"], fontsize=8.5)
        ax.set_xlabel("Response phase", fontsize=8)
        ax.set_ylabel("ΔEAS vs NC baseline", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.set_title(f"{COND_LABELS[cond]} — {cond.replace('_', ' ').title()}\n"
                     "Temporal IB commitment speed",
                     fontsize=9.5, pad=5, fontweight="bold")
        if col_i == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

    fig.suptitle(
        "Five-Model Architecture Gradient: Lexical → Pragmatic Encoding\n"
        "k=1 GQA → k=1 MLA → k=2 MLA+MTP  |  "
        "Orange = Rater (pragmatic)  Blue = Similarity (lexical)",
        fontsize=11, fontweight="bold", y=1.012)

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nFigure saved to {out_path}")

def plot_temporal_deferral(datasets, out_path="temporal_deferral.png"):
    """
    Plots the cognitive 'Temporal Deferral' curve for the Implicature condition.
    Contrasts DeepSeek V2-Lite (k=1) vs DeepSeek V3 (k=2).
    """
    sns.set_theme(style="whitegrid", rc={"axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 5))
    
    phases = ["early", "mid", "late"]
    x_pos = np.arange(3)
    
    models_to_plot = {
        "deepseek_v2_lite": {"label": "DeepSeek V2-Lite (k=1, MLA)", "color": "#E64A19"}, # Deep Orange
        "deepseek":         {"label": "DeepSeek V3 (k=2, MLA+MTP)", "color": "#1976D2"}  # Deep Blue
    }
    
    for model_key, meta in models_to_plot.items():
        df = datasets[model_key]
        # Filter strictly to the Implicature condition
        sub = df[df["condition"] == "implicature_information"]
        
        means = []
        sems = []
        for p in phases:
            d_col = f"delta_eas_{p}"
            if d_col in sub.columns:
                means.append(sub[d_col].mean())
                sems.append(sub[d_col].sem())
            else:
                means.append(np.nan)
                sems.append(0)
                
        means = np.array(means)
        sems = np.array(sems)
        
        # Smooth interpolation for the line to make it look like a continuous cognitive curve
        from scipy.interpolate import make_interp_spline
        x_smooth = np.linspace(x_pos.min(), x_pos.max(), 300)
        spl = make_interp_spline(x_pos, means, k=2)
        y_smooth = spl(x_smooth)
        
        # Plot the curve and standard error shading
        ax.plot(x_smooth, y_smooth, color=meta["color"], lw=3, label=meta["label"])
        ax.scatter(x_pos, means, color=meta["color"], s=80, zorder=5)
        ax.fill_between(x_pos, means - sems, means + sems, color=meta["color"], alpha=0.15, zorder=1)

    ax.axhline(0, color="#9E9E9E", lw=1.5, ls="--", zorder=0)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Early Phase\n(Context Intake)", "Mid Phase", "Late Phase\n(Implicature Resolution)"], fontsize=10)
    ax.set_ylabel("Entropy Shift vs Baseline (ΔEAS)", fontsize=11, fontweight="bold")
    ax.set_title("Temporal Deferral of Pragmatic Resolution\nImplicature Condition Only", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10, loc="upper left", frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved Temporal Deferral curve to {out_path}")
    plt.close()


def plot_shape_vs_magnitude(datasets, out_path="shape_vs_magnitude.png"):
    """
    Plots the decoupling of Trajectory Shape (DTW) and Magnitude (EAS) for DeepSeek V3.
    """
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # We only care about DeepSeek V3 for this specific finding
    df = datasets["deepseek"].copy()
    
    # Ensure our metrics exist (requires the DTW modification we made earlier)
    if "dtw_mean" not in df.columns or "delta_eas_mean" not in df.columns:
        print("Required columns (dtw_mean, delta_eas_mean) not found. Skipping scatter plot.")
        return

    # Create the scatter plot
    # x = Shape (DTW), y = Magnitude (EAS), color = Pragmatics (Rater), size = Lexical (Sim)
    scatter = sns.scatterplot(
        data=df,
        x="dtw_mean",
        y="delta_eas_mean",
        hue="acc_rater",
        size="acc_sim",
        palette="viridis",  # Great for ordinal data like a 1-5 rating
        sizes=(20, 250),    # Makes the size differences very obvious
        alpha=0.8,
        edgecolor="w",
        linewidth=0.5,
        ax=ax
    )

    # Clean up the visual frame
    sns.despine(trim=True, offset=5)

    # Draw quadrants (centered on the medians)
    ax.axvline(df["dtw_mean"].median(), color="#E0E0E0", ls="--", zorder=0)
    ax.axhline(df["delta_eas_mean"].median(), color="#E0E0E0", ls="--", zorder=0)
    
    ax.set_xlabel("Trajectory Shape Shift vs Baseline (DTW Distance)\n← Tracks Lexical/Syntax →", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Uncertainty Magnitude (ΔEAS)\n← Tracks Pragmatic Intent →", fontsize=11, fontweight="bold")
    ax.set_title("Decoupling of Shape and Magnitude in MTP Architectures\nDeepSeek V3 (k=2)", fontsize=14, fontweight="bold", pad=15)
    
    # Customize the legend to explain the multi-variate mapping clearly
    handles, labels = ax.get_legend_handles_labels()
    # Find the split points in the seaborn auto-legend
    hue_idx = labels.index("acc_rater") if "acc_rater" in labels else 0
    size_idx = labels.index("acc_sim") if "acc_sim" in labels else len(labels)
    
    # Rebuild legend with better titles
    labels[hue_idx] = "Pragmatic Relevance\n(Color)"
    labels[size_idx] = "\nLexical Similarity\n(Marker Size)"
    
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved Shape vs Magnitude scatter to {out_path}")
    plt.close()

def plot_decoupled_mechanics(datasets, out_path="decoupled_mechanics.png"):
    """
    Side-by-side regression plots isolating the two structural claims for DeepSeek V3:
    Both target variables are continuous, allowing for a mirrored 1x2 comparison.
    1. Magnitude (EAS) tracks human pragmatics (Continuous -> Regplot)
    2. Shape (DTW) tracks lexical similarity (Continuous -> Regplot)
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    df = datasets["deepseek"].copy()

    # --- Panel A: Magnitude vs Pragmatics ---
    # Using a regression plot for the continuous average rater scores
    sns.regplot(
        data=df, 
        x="acc_rater", 
        y="delta_eas_mean", 
        ax=axes[0], 
        color="#E65100",  # Deep Orange
        scatter_kws={'alpha': 0.6, 's': 60, 'edgecolor': 'w'},
        line_kws={'color': '#BF360C', 'lw': 2}
    )
    
    axes[0].set_title("Total Uncertainty Tracks Pragmatic Intent", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Average Human Rater Score (Pragmatics)\n← Less Relevant ... Highly Relevant →", fontsize=11)
    axes[0].set_ylabel("Magnitude of Entropy Shift (ΔEAS)", fontsize=11)
    axes[0].axhline(0, color="#9E9E9E", ls="--", lw=1, zorder=0)

    # --- Panel B: Shape vs Lexical Similarity ---
    # Using a regression plot for continuous semantic similarity
    sns.regplot(
        data=df, 
        x="acc_sim", 
        y="dtw_mean", 
        ax=axes[1], 
        color="#1976D2",  # Deep Blue
        scatter_kws={'alpha': 0.6, 's': 60, 'edgecolor': 'w'},
        line_kws={'color': '#0D47A1', 'lw': 2}
    )
    
    axes[1].set_title("Trajectory Shape Tracks Lexical Syntax", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Context-Question Semantic Similarity (Lexical)\n← Less Similar ... Highly Similar →", fontsize=11)
    axes[1].set_ylabel("Trajectory Distance from Baseline (DTW)", fontsize=11)

    sns.despine(trim=True, offset=5)
    
    fig.suptitle(
        "The MTP Decoupling Effect (DeepSeek V3)\nHow the model separates what it says from how it plans", 
        fontsize=16, 
        fontweight="bold", 
        y=1.05
    )
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved continuous Decoupled Mechanics visual to {out_path}")
    plt.close()
def plot_nonlinear_mechanics(datasets, out_path="decoupled_mechanics_fixed.png"):
    """
    Side-by-side plots isolating the non-linear structural claims for DeepSeek V3.
    Uses LOESS smoothing and KDE density contours to visualize the Mutual Information signal
    that strict linear regression hides.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    df = datasets["deepseek"].copy()

    # --- Panel A: Magnitude vs Pragmatics (Non-Linear) ---
    # 1. Density contour to show the joint probability distribution (what MI actually measures)
    sns.kdeplot(
        data=df, x="acc_rater", y="delta_eas_mean", 
        ax=axes[0], fill=True, cmap="Oranges", alpha=0.4, levels=6, zorder=1
    )
    # 2. Scatter plot for the underlying data
    sns.scatterplot(
        data=df, x="acc_rater", y="delta_eas_mean", 
        ax=axes[0], color="#D84315", s=40, alpha=0.7, edgecolor="w", zorder=2
    )
    # 3. LOESS curve (flexible, non-linear trendline)
    sns.regplot(
        data=df, x="acc_rater", y="delta_eas_mean", 
        ax=axes[0], scatter=False, lowess=True, color="#3E2723", line_kws={"lw": 2.5, "ls": "--"}, zorder=3
    )
    
    axes[0].set_title("Total Uncertainty Tracks Pragmatic Intent", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Average Human Rater Score (Pragmatics)\n← Less Relevant ... Highly Relevant →", fontsize=11)
    axes[0].set_ylabel("Magnitude of Entropy Shift (ΔEAS)", fontsize=11)
    axes[0].axhline(0, color="#9E9E9E", ls=":", lw=1.5, zorder=0)

    # --- Panel B: Shape vs Lexical Syntax (Non-Linear) ---
    sns.kdeplot(
        data=df, x="acc_sim", y="entropy_dtw", 
        ax=axes[1], fill=True, cmap="Blues", alpha=0.4, levels=6, zorder=1
    )
    sns.scatterplot(
        data=df, x="acc_sim", y="entropy_dtw", 
        ax=axes[1], color="#1565C0", s=40, alpha=0.7, edgecolor="w", zorder=2
    )
    sns.regplot(
        data=df, x="acc_sim", y="entropy_dtw", 
        ax=axes[1], scatter=False, lowess=True, color="#0D47A1", line_kws={"lw": 2.5, "ls": "--"}, zorder=3
    )
    
    axes[1].set_title("Trajectory Shape Tracks Lexical Syntax", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Context-Question Semantic Similarity (Lexical)\n← Less Similar ... Highly Similar →", fontsize=11)
    axes[1].set_ylabel("Trajectory Distance from Baseline (DTW)", fontsize=11)

    sns.despine(trim=True, offset=5)
    
    fig.suptitle(
        "The MTP Decoupling Effect (DeepSeek V3)\nVisualizing Non-Linear Mutual Information (Density & LOESS Curves)", 
        fontsize=16, 
        fontweight="bold", 
        y=1.05
    )
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved non-linear Decoupled Mechanics visual to {out_path}")
    plt.close()
# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    ctx, sim = load_reference_data()
    print(f"Reference data loaded: {len(ctx)} rater scores, {len(sim)} similarity scores")

    datasets = {m: load_model(m, ctx, sim) for m in ORDERED_MODELS}
    print(f"Models loaded: {list(datasets.keys())}")

    print_summary_table(datasets)
    print_v2lite_detail(datasets["deepseek"])
    print_gradient_test(datasets)
    plot_temporal_deferral(
        datasets,
        out_path=str(OUT_DIR / "temporal_deferral.png")
    )

    plot_decoupled_mechanics(
        datasets,
        out_path=str(OUT_DIR / "decoupled_mechanics.png")
    )
    plot_shape_vs_magnitude(
        datasets,
        out_path=str(OUT_DIR / "shape_vs_magnitude.png")
    )
    plot_nonlinear_mechanics(
        datasets,
        out_path=str(OUT_DIR / "nonlinear_mechanics.png")
    )
    plot_architecture_gradient(
        datasets,
        out_path=str(OUT_DIR / "architecture_gradient.png")
    )