"""
entropy_trajectory_analysis.py — per-token entropy trajectory analysis.

Answers the question: is the information-theoretic "work" of answering a
question *front-loaded* (model commits fast, then fills in) or *back-loaded*
(model stays exploratory, then converges late)?

Operates on the `extended_metrics_*.csv` produced by the pilot pipeline.
For each prompt we have:

  - token_entropy_sequence_sample_0 … _9  (per-step logit entropy, 10 samples)
  - eas_cross_sample_variance_profile     (per-step cross-sample variance)

From these we derive, per prompt:

  * H_total        — total response entropy (Σ H_t)              [nats]
  * H_half_life    — earliest step at which cumulative entropy
                     exceeds 50 % of H_total                     [token index]
  * half_life_rel  — H_half_life normalised to [0,1] by the
                     actual (non-padded) generation length        [unitless]
  * front_auc      — AUC of first-third H(t)                      [nats]
  * back_auc       — AUC of last-third H(t)                       [nats]
  * fb_ratio       — back_auc / front_auc
                     > 1 → back-loaded, < 1 → front-loaded        [unitless]
  * eas_auc_front / eas_auc_back / eas_fb_ratio — same on the
                     cross-sample variance profile (representational
                     analogue of the logit-entropy measure)

Per condition we aggregate the per-prompt metrics, run Wilcoxon signed-rank
tests between conditions on the same questions (we have 15 questions × 4
conditions, so the pairings are exact), and produce the trajectory plots.

Usage:
    python questions_x_context/scripts/entropy_trajectory_analysis.py \
        --csv questions_x_context/data/extended_metrics_deepseek.csv \
        --out questions_x_context/data

The script is deliberately dependency-light (numpy, pandas, scipy, matplotlib
only — all already in requirements.txt).
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Data parsing ─────────────────────────────────────────────────────────────

def _parse_semicolon_sequence(s: object) -> np.ndarray:
    """Parse 'a;b;c;...' → float array.  Empty / NaN → length-0 array."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.array([], dtype=float)
    parts = [p for p in str(s).split(";") if p != ""]
    if not parts:
        return np.array([], dtype=float)
    try:
        return np.asarray(parts, dtype=float)
    except ValueError:
        # Tolerate malformed cells by coercing element-wise.
        out = []
        for p in parts:
            try:
                out.append(float(p))
            except ValueError:
                out.append(np.nan)
        return np.asarray(out, dtype=float)


def _mean_entropy_across_samples(row: pd.Series, n_samples: int = 10) -> np.ndarray:
    """Average the 10 per-sample token_entropy_sequence vectors to a single
    per-position curve.  Sequences are mean-pooled up to the shortest length
    so we never average against zero-padding."""
    seqs = [
        _parse_semicolon_sequence(row.get(f"token_entropy_sequence_sample_{j}"))
        for j in range(n_samples)
    ]
    seqs = [s for s in seqs if s.size > 0]
    if not seqs:
        return np.array([], dtype=float)
    min_len = min(s.size for s in seqs)
    stacked = np.stack([s[:min_len] for s in seqs], axis=0)
    return stacked.mean(axis=0)


# ── Per-prompt information-theoretic metrics ─────────────────────────────────

def _half_life(h: np.ndarray) -> tuple[int, float]:
    """Return (index, (index+1)/len) of earliest step at which cumulative entropy
    exceeds 50 % of total entropy.  len is the actual curve length."""
    if h.size == 0:
        return -1, float("nan")
    total = h.sum()
    if total <= 0:
        return -1, float("nan")
    cum = np.cumsum(h)
    idx = int(np.searchsorted(cum, total / 2.0))
    idx = min(idx, h.size - 1)
    return idx, (idx + 1) / h.size


def _third_aucs(h: np.ndarray) -> tuple[float, float, float]:
    """Mean of first-third, middle-third, last-third of the curve (AUC
    normalized by third-length so all three are on the same scale)."""
    if h.size < 3:
        return float("nan"), float("nan"), float("nan")
    n = h.size
    t1 = n // 3
    t2 = 2 * n // 3
    a = float(h[:t1].mean()) if t1 > 0 else float("nan")
    b = float(h[t1:t2].mean()) if t2 > t1 else float("nan")
    c = float(h[t2:].mean()) if n > t2 else float("nan")
    return a, b, c


def _prompt_metrics(row: pd.Series) -> dict:
    H = _mean_entropy_across_samples(row)
    EAS = _parse_semicolon_sequence(row.get("eas_cross_sample_variance_profile"))

    total_H = float(H.sum()) if H.size else float("nan")
    hl_idx, hl_rel = _half_life(H)
    fH, mH, bH = _third_aucs(H)
    fE, mE, bE = _third_aucs(EAS)

    def _ratio(back, front):
        if front is None or np.isnan(front) or front <= 1e-9:
            return float("nan")
        if back is None or np.isnan(back):
            return float("nan")
        return back / front

    return {
        "prompt_id":        int(row["prompt_id"]),
        "condition":        row["condition"],
        "domain":           row["domain"],
        "gen_len_H":        int(H.size),
        "gen_len_EAS":      int(EAS.size),
        "H_total":          total_H,
        "H_mean":           float(H.mean()) if H.size else float("nan"),
        "H_half_life":      hl_idx,
        "half_life_rel":    hl_rel,
        "H_front_auc":      fH,
        "H_mid_auc":        mH,
        "H_back_auc":       bH,
        "H_fb_ratio":       _ratio(bH, fH),
        "EAS_front_auc":    fE,
        "EAS_mid_auc":      mE,
        "EAS_back_auc":     bE,
        "EAS_fb_ratio":     _ratio(bE, fE),
    }


# ── Aggregation & stats ──────────────────────────────────────────────────────

def per_condition_summary(per_prompt: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "H_total", "H_mean", "half_life_rel",
        "H_front_auc", "H_mid_auc", "H_back_auc", "H_fb_ratio",
        "EAS_front_auc", "EAS_mid_auc", "EAS_back_auc", "EAS_fb_ratio",
    ]
    agg = per_prompt.groupby("condition")[metric_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index()


def wilcoxon_by_condition(per_prompt: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank tests between every pair of conditions,
    pairing by `question_id` (4 conditions × 15 questions = 60 rows in the
    extended_metrics schema; question_id = (prompt_id - 1) // 4)."""
    conditions = sorted(per_prompt["condition"].unique())
    df = per_prompt.copy()
    df["question_id"] = (df["prompt_id"] - 1) // 4
    pivot = df.pivot(index="question_id", columns="condition", values=metric)
    rows = []
    for a, b in combinations(conditions, 2):
        pair = pivot[[a, b]].dropna()
        if len(pair) < 3:
            rows.append({"metric": metric, "cond_a": a, "cond_b": b,
                         "n_pairs": len(pair), "statistic": np.nan,
                         "p_value": np.nan, "mean_a": np.nan, "mean_b": np.nan})
            continue
        # Avoid SciPy error when differences are all zero.
        diff = pair[a].to_numpy() - pair[b].to_numpy()
        if np.allclose(diff, 0):
            w, p = 0.0, 1.0
        else:
            w, p = stats.wilcoxon(pair[a], pair[b], zero_method="wilcox")
        rows.append({
            "metric":    metric,
            "cond_a":    a,
            "cond_b":    b,
            "n_pairs":   len(pair),
            "statistic": float(w),
            "p_value":   float(p),
            "mean_a":    float(pair[a].mean()),
            "mean_b":    float(pair[b].mean()),
        })
    return pd.DataFrame(rows)


# ── Trajectory plots ─────────────────────────────────────────────────────────

def _resample_to_grid(curve: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linearly resample `curve` (length L) onto the normalised grid ∈ [0,1]."""
    if curve.size < 2:
        return np.full_like(grid, np.nan, dtype=float)
    src_x = np.linspace(0.0, 1.0, curve.size)
    return np.interp(grid, src_x, curve)


def plot_trajectories(
    df: pd.DataFrame,
    col_fn,
    ylabel: str,
    title: str,
    out_path: Path,
    n_points: int = 100,
) -> None:
    grid = np.linspace(0.0, 1.0, n_points)
    conditions = sorted(df["condition"].unique())
    colors = {"no_context": "#2c3e50",
              "direct_information": "#27ae60",
              "implicature_information": "#8e44ad",
              "stochastic_information": "#c0392b"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for cond in conditions:
        sub = df[df["condition"] == cond]
        resampled = []
        for _, row in sub.iterrows():
            curve = col_fn(row)
            if curve.size >= 2:
                resampled.append(_resample_to_grid(curve, grid))
        if not resampled:
            continue
        arr = np.vstack(resampled)
        mean = np.nanmean(arr, axis=0)
        sem  = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
        c = colors.get(cond, "#555555")
        ax.plot(grid, mean, color=c, lw=2.0, label=f"{cond} (n={len(sub)})")
        ax.fill_between(grid, mean - sem, mean + sem, color=c, alpha=0.15, linewidth=0)

    ax.set_xlabel("Normalized generation position (0 = first token, 1 = last)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_fb_distributions(per_prompt: pd.DataFrame, out_path: Path) -> None:
    conditions = sorted(per_prompt["condition"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, metric, ttl in zip(
        axes,
        ["H_fb_ratio", "EAS_fb_ratio"],
        ["Logit-entropy back/front ratio", "Cross-sample variance back/front ratio"],
    ):
        data = [per_prompt.loc[per_prompt["condition"] == c, metric].dropna().to_numpy()
                for c in conditions]
        bp = ax.boxplot(data, labels=conditions, patch_artist=True)
        palette = ["#2c3e50", "#27ae60", "#8e44ad", "#c0392b"]
        for patch, color in zip(bp["boxes"], palette[:len(conditions)]):
            patch.set_facecolor(color); patch.set_alpha(0.5)
        ax.axhline(1.0, color="black", lw=1, ls="--", alpha=0.6,
                   label="Equal front/back (= 1)")
        ax.set_title(ttl)
        ax.set_ylabel("back_AUC / front_AUC")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Front-loaded vs back-loaded processing per condition")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ── Top-level driver ─────────────────────────────────────────────────────────

def run(csv_path: Path, out_dir: Path, model: str = "") -> None:
    df = pd.read_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{model}" if model else ""

    per_prompt_rows = [_prompt_metrics(r) for _, r in df.iterrows()]
    per_prompt = pd.DataFrame(per_prompt_rows)
    per_prompt.to_csv(out_dir / f"entropy_trajectory_per_prompt{suffix}.csv", index=False)

    summary = per_condition_summary(per_prompt)
    summary.to_csv(out_dir / f"entropy_trajectory_summary{suffix}.csv", index=False)

    # Paired Wilcoxon tests on the key metrics.
    wilcox_rows = []
    for metric in ["H_total", "half_life_rel", "H_fb_ratio", "EAS_fb_ratio"]:
        wilcox_rows.append(wilcoxon_by_condition(per_prompt, metric))
    wilcox = pd.concat(wilcox_rows, ignore_index=True)
    wilcox.to_csv(out_dir / f"entropy_trajectory_wilcoxon{suffix}.csv", index=False)

    # Trajectory plots.
    plot_trajectories(
        df,
        col_fn=lambda r: _mean_entropy_across_samples(r),
        ylabel="Mean token entropy [nats]",
        title="Logit entropy trajectory by condition (mean ± SEM)",
        out_path=out_dir / f"entropy_trajectory_logit{suffix}.png",
    )
    plot_trajectories(
        df,
        col_fn=lambda r: _parse_semicolon_sequence(r.get("eas_cross_sample_variance_profile")),
        ylabel="Cross-sample variance (EAS)",
        title="Representational entropy (EAS) trajectory by condition (mean ± SEM)",
        out_path=out_dir / f"entropy_trajectory_eas{suffix}.png",
    )
    plot_fb_distributions(per_prompt, out_dir / f"entropy_frontback_distributions{suffix}.png")

    # Console summary.
    print(f"Rows analysed:        {len(df)}")
    print(f"Conditions:           {sorted(df['condition'].unique())}")
    print(f"Prompts per condition {df['condition'].value_counts().to_dict()}")
    print()
    print("=== Per-condition means (key metrics) ===")
    keep = [c for c in summary.columns if c == "condition" or c.endswith(("_mean",))]
    print(summary[keep].to_string(index=False))
    print()
    print("=== Wilcoxon signed-rank (paired by prompt_id) ===")
    print(wilcox.to_string(index=False))
    print()
    print(f"Outputs written to: {out_dir}")


def parse_args() -> argparse.Namespace:
    _repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="Path to extended_metrics_*.csv")
    p.add_argument("--out", default=str(_repo / "results" / "plots"),
                   help="Output directory for CSVs and PNGs")
    p.add_argument("--model", default="",
                   help="Model label appended to output filenames (e.g. 'deepseek'). "
                        "Required when running multiple models to avoid overwriting.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(csv_path=Path(args.csv), out_dir=Path(args.out), model=args.model)
