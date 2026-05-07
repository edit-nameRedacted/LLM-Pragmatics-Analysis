# uncertainty_fingerprint.py
# ==========================
# Multi-level uncertainty fingerprint visualisation.

# Each condition gets one strip with four layered encodings:

# EAS blob (background)
#     A variable-width tube of circles at 20% opacity.
#     Circle area at each time point ∝ EAS value (entropy area score).
#     Adjacent circles are connected by a smooth filled region —
#     so the shape gets thicker where the model is carrying more
#     cumulative uncertainty and thinner where it is more resolved.

# Logit entropy circles (foreground)
#     At each time point, N filled circles are placed inside the EAS blob.
#     N ∝ mean token logit entropy — more circles = more token-level
#     uncertainty at that stage.  Total area of all circles ≤ EAS blob area,
#     so the logit circles never exceed the envelope set by EAS.

# Beam diversity → size variation
#     The relative size variation across the N logit circles is controlled
#     by beam_score_raw_std.  Higher beam diversity (the model is uncertain
#     which complete response to commit to) produces a wider spread of
#     circle sizes — some very small, some large.  Low beam diversity
#     produces uniform circles.

# CAA displacement → y offset
#     The vertical position of each condition strip's centre is offset by
#     the condition's mean CAA L2 displacement from the no-context baseline.
#     Higher displacement (context moved the model further from its default
#     representational state) pushes the strip upward relative to others.

# Time axis
#     Four windows derived from the token entropy sequences:
#       early       = mean H over first third of generation
#       mid         = mean H over middle third
#       late        = mean H over final third
#       final_quarter = mean H over final 25%

# Usage
# -----
#     python uncertainty_fingerprint.py \\
#         --extended   extended_metrics_qwen.csv \\
#         --pilot      pilot_summary.csv \\
#         --output     fingerprint.png          # omit to display
#         --model_name "Qwen2.5-7B"
# """

from __future__ import annotations
 
import argparse
import warnings
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
warnings.filterwarnings("ignore")
 
# ── Constants ─────────────────────────────────────────────────────────────────
COND_ORDER = [
    "no_context",
    "direct_information",
    "implicature_information",
    "stochastic_information",
]
LABELS = {
    "no_context":             "No Context",
    "direct_information":     "Direct Information",
    "implicature_information":"Implicature",
    "stochastic_information": "Stochastic",
}
COLORS = {
    "no_context":             "#888888",
    "direct_information":     "#2166AC",
    "implicature_information":"#1A9850",
    "stochastic_information": "#C0392B",
}
 
# EAS time windows
TIME_KEYS   = ["eas_early", "eas_mid", "eas_late", "eas_final_quarter"]
TIME_X      = np.array([0.12, 0.38, 0.65, 0.85])   # normalized [0,1] x positions
TIME_LABELS = ["Early", "Mid", "Late", "Final Q"]
 
ROW_H           = 2.2      # height of each condition strip in data units
EAS_MAX_FILL    = 0.40     # EAS max radius as fraction of ROW_H / 2
MAX_LOGIT_CIRC  = 8        # maximum number of logit circles at peak entropy
LOGIT_H_MAX_REF = 0.55     # reference entropy value (nats) = MAX_LOGIT_CIRC circles
 
 
# ── Data helpers ──────────────────────────────────────────────────────────────
 
def parse_seq(s: str, max_len: int = 150) -> np.ndarray:
    vals = [max(float(x), 0.0) for x in str(s).split(";") if x.strip()]
    return np.array(vals[:max_len])
 
 
def mean_seqs_by_window(group_df: pd.DataFrame) -> list[float]:
    """Mean token entropy across all samples for each of 4 temporal windows."""
    all_seqs = []
    for i in range(10):
        col = f"token_entropy_sequence_sample_{i}"
        if col not in group_df.columns:
            continue
        for _, row in group_df.iterrows():
            if pd.notna(row[col]):
                all_seqs.append(parse_seq(row[col]))
    if not all_seqs:
        return [0.0] * 4
    max_len = max(len(s) for s in all_seqs)
    mat = np.full((len(all_seqs), max_len), np.nan)
    for i, s in enumerate(all_seqs):
        mat[i, : len(s)] = s
    mean_curve = np.nanmean(mat, axis=0)
    n   = len(mean_curve)
    t1  = int(n / 3)
    t2  = int(2 * n / 3)
    t3  = int(0.75 * n)
    windows = [
        float(np.nanmean(mean_curve[:t1])),
        float(np.nanmean(mean_curve[t1:t2])),
        float(np.nanmean(mean_curve[t2:])),
        float(np.nanmean(mean_curve[t3:])),
    ]
    return windows
 
 
def load_condition_data(
    em_path: Path,
    pilot_path: Path,
) -> dict[str, dict]:
    em  = pd.read_csv(em_path)
    pil = pd.read_csv(pilot_path)
 
    # Attach CAA from pilot summary
    caa_cols = ["prompt_id", "condition", "caa_mean_l2"]
    pil_caa  = pil[[c for c in caa_cols if c in pil.columns]].copy()
    if "caa_mean_l2" in pil_caa.columns:
        em = em.merge(pil_caa, on=["prompt_id", "condition"], how="left")
    else:
        em["caa_mean_l2"] = np.nan
 
    cond_data: dict[str, dict] = {}
    for cond in COND_ORDER:
        sub = em[em.condition == cond]
        if sub.empty:
            cond_data[cond] = dict(
                eas=[0.4] * 4, logit_h=[0.4] * 4,
                beam_std=0.1, caa_l2=np.nan,
            )
            continue
        eas_vals = [float(sub[k].mean()) for k in TIME_KEYS if k in sub.columns]
        if len(eas_vals) < 4:
            eas_vals = [sub["eas_mean"].mean()] * 4 if "eas_mean" in sub.columns else [0.4] * 4
        logit_h  = mean_seqs_by_window(sub)
        beam_std = float(sub["beam_score_raw_std"].mean()) if "beam_score_raw_std" in sub.columns else 0.1
        caa_l2   = float(sub["caa_mean_l2"].mean()) if "caa_mean_l2" in sub.columns else np.nan
        cond_data[cond] = dict(eas=eas_vals, logit_h=logit_h,
                               beam_std=beam_std, caa_l2=caa_l2)
    return cond_data
 
 
# ── Geometry ──────────────────────────────────────────────────────────────────
 
def eas_radius(val: float, row_h: float) -> float:
    """Convert EAS value to display radius."""
    # Max EAS (~0.55 nats) maps to EAS_MAX_FILL * row_h / 2
    max_r  = EAS_MAX_FILL * row_h / 2
    ref    = 0.55
    return max_r * np.sqrt(max(val, 0.01) / ref)
 
 
def caa_y_offset(caa_l2: float, all_l2: np.ndarray, row_h: float) -> float:
    """Map CAA L2 displacement to y offset (higher displacement = higher y)."""
    if np.isnan(caa_l2):
        return 0.0
    valid = all_l2[~np.isnan(all_l2)]
    if len(valid) < 2 or valid.max() == valid.min():
        return 0.0
    norm = (caa_l2 - valid.min()) / (valid.max() - valid.min())
    return (norm - 0.5) * row_h * 0.28   # ±14% of row height
 
 
# ── Drawing ───────────────────────────────────────────────────────────────────
 
def draw_eas_blob(
    ax,
    x_positions: np.ndarray,
    y_center: float,
    eas_values: list[float],
    row_h: float,
    color: str,
    alpha: float = 0.20,
) -> list[float]:
    """Draw EAS as filled circles connected by a smooth variable-width tube."""
    radii = [eas_radius(v, row_h) for v in eas_values]
 
    # Tube fill between adjacent circles
    for i in range(len(x_positions) - 1):
        x0, x1 = x_positions[i], x_positions[i + 1]
        r0, r1 = radii[i], radii[i + 1]
        xs = np.linspace(x0, x1, 80)
        rs = np.interp(xs, [x0, x1], [r0, r1])
        ax.fill_between(xs, y_center - rs, y_center + rs,
                        color=color, alpha=alpha, linewidth=0, zorder=1)
 
    # Circles at each time point
    for x, r in zip(x_positions, radii):
        body = mpatches.Circle((x, y_center), r, color=color,
                                alpha=alpha, zorder=2, linewidth=0)
        ring = mpatches.Circle((x, y_center), r, fill=False,
                                edgecolor=color, alpha=alpha * 1.8,
                                linewidth=0.6, zorder=3)
        ax.add_patch(body)
        ax.add_patch(ring)
 
    return radii
 
 
def draw_logit_circles(
    ax,
    x_positions: np.ndarray,
    y_center: float,
    logit_h_vals: list[float],
    eas_radii: list[float],
    beam_raw_std: float,
    row_h: float,
    color: str,
    seed: int = 42,
) -> None:
    """
    Scatter N small circles at each time point.
    N ∝ token logit entropy.
    Total area ≤ EAS blob area.
    Size variation ∝ beam_raw_std.
    """
    rng = np.random.default_rng(seed)
 
    for x, H, r_eas in zip(x_positions, logit_h_vals, eas_radii):
        if H < 0.01 or r_eas < 0.001:
            continue
 
        # Number of circles
        n_circ = max(1, round(MAX_LOGIT_CIRC * H / LOGIT_H_MAX_REF))
        n_circ = min(n_circ, MAX_LOGIT_CIRC)
 
        eas_area = np.pi * r_eas ** 2
 
        # Sizes: beam_raw_std drives size variation via lognormal
        sigma = float(np.clip(beam_raw_std, 0.0, 0.5)) * 1.5
        if sigma < 0.02 or n_circ == 1:
            sizes = np.ones(n_circ)
        else:
            raw  = rng.lognormal(0.0, sigma, n_circ)
            sizes = raw / raw.mean()   # normalise so mean = 1
 
        # Rescale so total area = EAS area
        r_base = np.sqrt(eas_area / (np.pi * n_circ))
        radii  = sizes * r_base
 
        # Arrange centres — sunflower-ish pattern within EAS circle
        if n_circ == 1:
            centres = [(0.0, 0.0)]
        else:
            angles  = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
            spread  = r_eas * 0.48
            centres = [(spread * np.cos(a), spread * np.sin(a)) for a in angles]
 
        for (dx, dy), r in zip(centres, radii):
            r_draw = min(r, r_eas * 0.82)
            circ   = mpatches.Circle((x + dx, y_center + dy), r_draw,
                                      color=color, alpha=0.72,
                                      zorder=5, linewidth=0)
            ax.add_patch(circ)
 
 
# ── Main plot ─────────────────────────────────────────────────────────────────
 
def plot(
    em_path: Path,
    pilot_path: Path,
    out_path: Path | None = None,
    model_name: str = "LLM",
) -> None:
    cond_data = load_condition_data(em_path, pilot_path)
 
    all_caa = np.array([cond_data[c]["caa_l2"] for c in COND_ORDER])
    n_conds = len(COND_ORDER)
    X_LO, X_HI = 0.02, 0.98
 
    fig, axes = plt.subplots(
        n_conds, 1,
        figsize=(13, n_conds * ROW_H + 0.9),
        facecolor="#111111",
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.96,
                        bottom=0.04, hspace=0.04)
 
    for row_idx, cond in enumerate(COND_ORDER):
        ax  = axes[row_idx]
        ax.set_facecolor("#111111")
        ax.set_xlim(X_LO, X_HI)
        ax.set_ylim(-ROW_H / 2, ROW_H / 2)
        ax.axis("off")
 
        color = COLORS[cond]
        d     = cond_data[cond]
 
        # CAA → y offset
        y_off = caa_y_offset(d["caa_l2"], all_caa, ROW_H)
 
        # Map normalised time positions into axis x-space
        xpos = X_LO + TIME_X * (X_HI - X_LO)
 
        # 1. EAS blob
        eas_radii = draw_eas_blob(ax, xpos, y_off, d["eas"],
                                   ROW_H, color, alpha=0.20)
 
        # 2. Logit circles
        draw_logit_circles(ax, xpos, y_off, d["logit_h"],
                           eas_radii, d["beam_std"], ROW_H, color)
 
        # 3. Dashed spine at y_off
        ax.plot(xpos, [y_off] * 4, color=color, linewidth=0.4,
                alpha=0.25, linestyle="--", zorder=1)
 
        # 4. CAA offset indicator (small arrow from baseline)
        if abs(y_off) > 0.01:
            ax.annotate("", xy=(X_LO + 0.006, y_off),
                        xytext=(X_LO + 0.006, 0),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        alpha=0.45, lw=0.7),
                        zorder=6)
 
        # Labels
        ax.text(X_LO + 0.007, ROW_H / 2 * 0.87, LABELS[cond],
                color="white", fontsize=9, fontweight="bold",
                va="top", ha="left", alpha=0.92)
 
        caa_str = f"{d['caa_l2']:.1f}" if not np.isnan(d["caa_l2"]) else "—"
        ax.text(X_HI - 0.006, ROW_H / 2 * 0.87,
                f"CAA L2 = {caa_str}  |  beam σ = {d['beam_std']:.3f}",
                color=color, fontsize=7.5, va="top", ha="right", alpha=0.80)
 
        # Time labels (bottom strip only)
        if row_idx == n_conds - 1:
            for xt, lbl in zip(xpos, TIME_LABELS):
                ax.text(xt, -ROW_H / 2 * 0.91, lbl, color="#888888",
                        fontsize=7, ha="center", va="bottom")
 
        # Row divider
        if row_idx < n_conds - 1:
            ax.axhline(-ROW_H / 2 + 0.02, color="#2A2A2A", linewidth=0.6)
 
    # Caption
    fig.text(
        0.50, 0.007,
        "EAS blob = entropy area (20% opacity, area ∝ cumulative uncertainty)  ·  "
        "Circles = token entropy (count ∝ H)  ·  "
        "Circle size variation = beam diversity  ·  "
        "Y position = CAA displacement",
        ha="center", color="#666666", fontsize=7.0,
    )
    fig.suptitle(
        f"Multi-level uncertainty fingerprint — {model_name}",
        fontsize=12, fontweight="bold", color="#E0E0E0", y=0.98,
    )
 
    if out_path:
        plt.savefig(out_path, dpi=180, bbox_inches="tight",
                    facecolor="#111111")
        print(f"Saved → {out_path}")
    else:
        plt.show()
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Multi-level uncertainty fingerprint visualisation"
    )
    p.add_argument("--extended",   type=Path, required=True,
                   help="Path to extended_metrics_<model>.csv")
    p.add_argument("--pilot",      type=Path, required=True,
                   help="Path to pilot_summary_<model>.csv")
    p.add_argument("--output",     type=Path, default=None,
                   help="Output path (.png / .pdf); displays if omitted")
    p.add_argument("--model_name", type=str,  default="LLM",
                   help="Model name for the figure title")
    args = p.parse_args()
    plot(args.extended, args.pilot, args.output, args.model_name)