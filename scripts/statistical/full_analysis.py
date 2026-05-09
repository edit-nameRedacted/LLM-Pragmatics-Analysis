"""
full_analysis.py — Complete QxC statistical analysis across all models.

Reads extended_metrics_*.csv files for deepseek, llama, mistral, and qwen,
computes all key metrics, runs paired Wilcoxon tests, and writes a full
report to results/full_analysis_report.txt and results/full_analysis_wilcoxon.csv.

For EAS trajectory: computes cross-sample variance profile from the 10
token_entropy_sequence_sample_* columns when eas_cross_sample_variance_profile
is absent (deepseek, llama, mistral).

Note on beam_score_entropy: this metric is ~log(10) = 2.3026 for all
conditions in deepseek/llama/mistral because the diverse-sampling approach
produces near-equal log-prob scores across the 10 runs. It is uninformative
and is flagged as such in the report.

Usage:
    python questions_x_context/scripts/full_analysis.py
"""

from __future__ import annotations

import json
from io import StringIO
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA = ROOT / "data"

def _model_path(model: str) -> Path:
    return DATA / "model" / model / f"extended_metrics_{model}.csv"

MODEL_PATHS = {m: _model_path(m) for m in ("deepseek", "llama", "mistral", "qwen")}

CONDITIONS = [
    "no_context",
    "direct_information",
    "implicature_information",
    "stochastic_information",
]

CONDITION_SHORT = {
    "no_context":               "NC",
    "direct_information":       "DI",
    "implicature_information":  "II",
    "stochastic_information":   "SI",
}

KEY_COMPARISONS = [
    ("direct_information",      "stochastic_information"),
    ("direct_information",      "no_context"),
    ("implicature_information", "stochastic_information"),
    ("no_context",              "stochastic_information"),
]

LOG10 = np.log(10)


# ── Semicolon-sequence parser ─────────────────────────────────────────────────

def _parse_seq(s) -> np.ndarray:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.array([], dtype=float)
    parts = [p for p in str(s).split(";") if p != ""]
    if not parts:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(parts, dtype=float)
    except ValueError:
        arr = np.array([float(p) if p else np.nan for p in parts], dtype=float)
    return np.where(arr < 0, 0.0, arr)  # clamp -0.0 artifacts


# ── Cross-sample variance profile ─────────────────────────────────────────────

def _compute_csv_profile(row: pd.Series, n_samples: int = 10) -> np.ndarray:
    """Compute per-position cross-sample variance from the 10 sample sequences."""
    seqs = [_parse_seq(row.get(f"token_entropy_sequence_sample_{j}"))
            for j in range(n_samples)]
    seqs = [s for s in seqs if s.size > 0]
    if not seqs:
        return np.array([], dtype=float)
    min_len = min(s.size for s in seqs)
    stacked = np.stack([s[:min_len] for s in seqs], axis=0)  # (n_samples, min_len)
    return stacked.var(axis=0)


def _mean_entropy_across_samples(row: pd.Series, n_samples: int = 10) -> np.ndarray:
    seqs = [_parse_seq(row.get(f"token_entropy_sequence_sample_{j}"))
            for j in range(n_samples)]
    seqs = [s for s in seqs if s.size > 0]
    if not seqs:
        return np.array([], dtype=float)
    min_len = min(s.size for s in seqs)
    return np.stack([s[:min_len] for s in seqs], axis=0).mean(axis=0)


# ── Per-token trajectory shape ────────────────────────────────────────────────

def _third_aucs(h: np.ndarray) -> tuple[float, float, float]:
    if h.size < 3:
        return np.nan, np.nan, np.nan
    n = h.size
    t1, t2 = n // 3, 2 * n // 3
    a = float(h[:t1].mean()) if t1 > 0 else np.nan
    b = float(h[t1:t2].mean()) if t2 > t1 else np.nan
    c = float(h[t2:].mean()) if n > t2 else np.nan
    return a, b, c


def _fb_ratio(h: np.ndarray) -> float:
    if h.size < 3:
        return np.nan
    front, _, back = _third_aucs(h)
    if np.isnan(front) or front <= 1e-9:
        return np.nan
    if np.isnan(back):
        return np.nan
    return back / front


def _half_life_rel(h: np.ndarray) -> float:
    if h.size == 0:
        return np.nan
    total = h.sum()
    if total <= 0:
        return np.nan
    cum = np.cumsum(h)
    idx = int(np.searchsorted(cum, total / 2.0))
    idx = min(idx, h.size - 1)
    return idx / max(h.size - 1, 1)


def _prompt_trajectory_metrics(row: pd.Series) -> dict:
    H = _mean_entropy_across_samples(row)

    # EAS: prefer stored profile, else compute from samples
    if "eas_cross_sample_variance_profile" in row.index:
        EAS = _parse_seq(row.get("eas_cross_sample_variance_profile"))
    else:
        EAS = _compute_csv_profile(row)

    return {
        "prompt_id":    int(row["prompt_id"]),
        "condition":    row["condition"],
        "H_fb_ratio":   _fb_ratio(H),
        "EAS_fb_ratio": _fb_ratio(EAS),
        "half_life_rel": _half_life_rel(H),
        "H_mean":       float(H.mean()) if H.size else np.nan,
        "gen_len":      int(H.size),
    }


# ── Statistics helpers ────────────────────────────────────────────────────────

def question_id(prompt_id: int) -> int:
    return (prompt_id - 1) // 4


def wilcoxon_pair(df: pd.DataFrame, metric: str, a: str, b: str):
    if metric not in df.columns:
        return None
    tmp = df.copy()
    tmp["qid"] = tmp["prompt_id"].apply(question_id)
    pivot = tmp.pivot(index="qid", columns="condition", values=metric)
    if a not in pivot.columns or b not in pivot.columns:
        return None
    pair = pivot[[a, b]].dropna()
    if len(pair) < 3:
        return None
    diff = pair[a].to_numpy() - pair[b].to_numpy()
    if np.allclose(diff, 0):
        return dict(w=0.0, p=1.0, mean_a=float(pair[a].mean()),
                    mean_b=float(pair[b].mean()), n=len(pair))
    w, p = stats.wilcoxon(pair[a], pair[b], zero_method="wilcox")
    return dict(w=float(w), p=float(p), mean_a=float(pair[a].mean()),
                mean_b=float(pair[b].mean()), n=len(pair))


def stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "†"
    return "ns"


# ── Trajectory plot (resampled to [0,1]) ─────────────────────────────────────

def _resample(curve: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if curve.size < 2:
        return np.full_like(grid, np.nan)
    return np.interp(grid, np.linspace(0.0, 1.0, curve.size), curve)


def plot_trajectory(df: pd.DataFrame, col_fn, ylabel: str, title: str,
                    out_path: Path, n: int = 100) -> None:
    grid = np.linspace(0.0, 1.0, n)
    colors = {
        "no_context":               "#2c3e50",
        "direct_information":       "#27ae60",
        "implicature_information":  "#8e44ad",
        "stochastic_information":   "#c0392b",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond in sorted(df["condition"].unique()):
        sub = df[df["condition"] == cond]
        resampled = [_resample(col_fn(row), grid)
                     for _, row in sub.iterrows()]
        valid = [r for r in resampled if not np.all(np.isnan(r))]
        if not valid:
            continue
        arr = np.vstack(valid)
        mean = np.nanmean(arr, axis=0)
        sem  = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0).clip(1))
        c = colors.get(cond, "#555555")
        ax.plot(grid, mean, color=c, lw=2, label=f"{CONDITION_SHORT.get(cond, cond)} (n={len(sub)})")
        ax.fill_between(grid, mean - sem, mean + sem, color=c, alpha=0.15, lw=0)
    ax.set_xlabel("Normalised generation position")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ── Main report ───────────────────────────────────────────────────────────────

def analyse_model(model: str, path: Path, out: StringIO) -> list[dict]:
    """Run full analysis for one model. Returns rows for the combined Wilcoxon CSV."""
    if not path.exists():
        out.write(f"\n[SKIP] {model}: file not found at {path}\n")
        return []

    df = pd.read_csv(path)
    df["condition"] = df["condition"].str.strip()

    hr = "─" * 64
    out.write(f"\n{'═'*64}\n  MODEL: {model.upper()}  ({len(df)} rows)\n{'═'*64}\n")

    # ── Trajectory metrics ────────────────────────────────────────────────────
    traj = pd.DataFrame([_prompt_trajectory_metrics(r) for _, r in df.iterrows()])
    # Merge back into df for Wilcoxon convenience
    traj_cols = ["prompt_id", "H_fb_ratio", "EAS_fb_ratio", "half_life_rel"]
    df = df.merge(traj[traj_cols], on="prompt_id", how="left")

    # ── Descriptives ──────────────────────────────────────────────────────────
    out.write("\n1. DESCRIPTIVES\n" + hr + "\n")

    scalar_metrics = {
        "eas_mean":                    "EAS mean (token entropy, per-sample mean)",
        "beam_sbert_cosine":           "beam_sbert_cosine (semantic agreement across beams)",
        "beam_score_entropy":          "beam_score_entropy [note: expect ~log(10)=2.303 — flat]",
        "beam_first_divergence_position": "beam_first_divergence_position",
        "beam_length_mean":            "beam_length_mean (words)",
        "H_fb_ratio":                  "H_fb_ratio (back/front logit entropy — <1 = front-loaded)",
        "EAS_fb_ratio":                "EAS_fb_ratio (back/front cross-sample variance)",
        "half_life_rel":               "half_life_rel (rel. position of 50% cumulative entropy)",
    }

    for col, label in scalar_metrics.items():
        if col not in df.columns:
            continue
        out.write(f"\n  {label}\n")
        grp = df.groupby("condition")[col].agg(["mean", "std", "count"])
        for cond in CONDITIONS:
            if cond not in grp.index:
                continue
            r = grp.loc[cond]
            short = CONDITION_SHORT.get(cond, cond)
            out.write(f"    {short}  M={r['mean']:.4f}  SD={r['std']:.4f}  n={int(r['count'])}\n")

        # Flag if beam_score_entropy is flat
        if col == "beam_score_entropy":
            spread = df[col].std()
            if spread < 0.01:
                out.write(f"    [!] SD={spread:.6f} — metric is flat (near-uniform sampling "
                          f"gives ~log(10) always). Excluded from inferential tests.\n")

    # ── Wilcoxon tests ────────────────────────────────────────────────────────
    out.write(f"\n2. PAIRED WILCOXON TESTS (by question_id = (prompt_id-1)//4)\n" + hr + "\n")

    test_metrics = [
        "eas_mean", "beam_sbert_cosine",
        "beam_first_divergence_position", "beam_length_mean",
        "H_fb_ratio", "EAS_fb_ratio", "half_life_rel",
    ]

    rows: list[dict] = []
    for col in test_metrics:
        if col not in df.columns:
            continue
        out.write(f"\n  {col}\n")
        for a, b in KEY_COMPARISONS:
            res = wilcoxon_pair(df, col, a, b)
            if res is None:
                continue
            sa = CONDITION_SHORT.get(a, a)
            sb = CONDITION_SHORT.get(b, b)
            sig = stars(res["p"])
            out.write(
                f"    {sa} vs {sb}:  M={res['mean_a']:.4f} vs {res['mean_b']:.4f}"
                f"  Δ={res['mean_a']-res['mean_b']:+.4f}"
                f"  W={res['w']:.0f}  p={res['p']:.4f}  {sig}"
                f"  (n={res['n']})\n"
            )
            rows.append({
                "model": model, "metric": col,
                "cond_a": a, "cond_b": b,
                "mean_a": res["mean_a"], "mean_b": res["mean_b"],
                "delta": res["mean_a"] - res["mean_b"],
                "W": res["w"], "p": res["p"],
                "sig": sig, "n": res["n"],
            })

    # ── Trajectory plots for this model ───────────────────────────────────────
    plots_dir = RESULTS / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_trajectory(
        df,
        col_fn=lambda r: _mean_entropy_across_samples(r),
        ylabel="Mean token entropy [nats]",
        title=f"{model} — Logit entropy trajectory (mean ± SEM)",
        out_path=plots_dir / f"entropy_trajectory_logit_{model}.png",
    )

    # EAS: use eas_cross_sample_variance_profile if present, else compute
    if "eas_cross_sample_variance_profile" in df.columns:
        eas_col_fn = lambda r: _parse_seq(r.get("eas_cross_sample_variance_profile"))
    else:
        eas_col_fn = lambda r: _compute_csv_profile(r)

    plot_trajectory(
        df,
        col_fn=eas_col_fn,
        ylabel="Cross-sample variance (EAS)",
        title=f"{model} — EAS trajectory (mean ± SEM)",
        out_path=plots_dir / f"entropy_trajectory_eas_{model}.png",
    )

    out.write(f"\n  Trajectory plots → results/plots/entropy_trajectory_{{logit,eas}}_{model}.png\n")

    return rows


def main() -> None:
    out = StringIO()
    out.write("QxC Full Analysis Report\n")
    out.write("=" * 64 + "\n")
    out.write("Models: deepseek, llama, mistral, qwen\n")
    out.write("Conditions: NC (no_context), DI (direct_information),\n"
              "            II (implicature_information), SI (stochastic_information)\n")
    out.write("Paired Wilcoxon: questions paired by (prompt_id-1)//4\n")
    out.write("Significance: *** p<.001  ** p<.01  * p<.05  † p<.10  ns\n\n")
    out.write("KEY HYPOTHESIS:\n"
              "  H1: relevant context (DI) → lower entropy than SI\n"
              "      DI mean < SI mean on eas_mean, H_fb_ratio, half_life_rel\n"
              "  H2: beam_sbert_cosine higher for DI vs SI\n"
              "      (beams semantically converge under relevant context)\n")

    all_wilcoxon: list[dict] = []
    for model, path in MODEL_PATHS.items():
        rows = analyse_model(model, path, out)
        all_wilcoxon.extend(rows)

    # ── Cross-model summary ───────────────────────────────────────────────────
    if all_wilcoxon:
        wdf = pd.DataFrame(all_wilcoxon)

        out.write("\n" + "═" * 64 + "\n  CROSS-MODEL SUMMARY\n" + "═" * 64 + "\n")
        out.write("\nSignificant results (p < .05) across all models:\n")
        sig = wdf[wdf["p"] < 0.05].sort_values(["metric", "cond_a", "cond_b"])
        if sig.empty:
            out.write("  (none at p < .05)\n")
        else:
            for _, r in sig.iterrows():
                sa = CONDITION_SHORT.get(r["cond_a"], r["cond_a"])
                sb = CONDITION_SHORT.get(r["cond_b"], r["cond_b"])
                out.write(
                    f"  {r['model']:10s}  {r['metric']:35s}  {sa} vs {sb}:"
                    f"  Δ={r['delta']:+.4f}  p={r['p']:.4f} {r['sig']}\n"
                )

        out.write("\nNear-significant trends (p < .10):\n")
        trend = wdf[(wdf["p"] >= 0.05) & (wdf["p"] < 0.10)].sort_values(["metric", "cond_a"])
        if trend.empty:
            out.write("  (none)\n")
        else:
            for _, r in trend.iterrows():
                sa = CONDITION_SHORT.get(r["cond_a"], r["cond_a"])
                sb = CONDITION_SHORT.get(r["cond_b"], r["cond_b"])
                out.write(
                    f"  {r['model']:10s}  {r['metric']:35s}  {sa} vs {sb}:"
                    f"  Δ={r['delta']:+.4f}  p={r['p']:.4f} {r['sig']}\n"
                )

        out.write("\nDirectional consistency (DI < SI on eas_mean across models):\n")
        di_si = wdf[(wdf["metric"] == "eas_mean") &
                    (wdf["cond_a"] == "direct_information") &
                    (wdf["cond_b"] == "stochastic_information")]
        for _, r in di_si.iterrows():
            direction = "DI < SI ✓" if r["delta"] < 0 else "DI > SI ✗"
            out.write(f"  {r['model']:10s}  Δ={r['delta']:+.4f}  p={r['p']:.4f}  {direction}\n")

        # Save Wilcoxon CSV
        wdf.to_csv(RESULTS / "full_analysis_wilcoxon.csv", index=False)
        out.write(f"\nWilcoxon table → results/full_analysis_wilcoxon.csv\n")

    # ── Write report ──────────────────────────────────────────────────────────
    report = out.getvalue()
    print(report)
    report_path = RESULTS / "full_analysis_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report → {report_path}")


if __name__ == "__main__":
    main()
