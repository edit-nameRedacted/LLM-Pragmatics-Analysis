"""
uncertainty_fingerprint.py
==========================
Multi-level uncertainty fingerprint — hue-differentiated per entropy type.

Each condition gets one strip with four layered encodings:

EAS blob  (base hue, 22% opacity)
    Variable-width tube. Area at each time point ∝ EAS value.

Logit entropy circles  (base hue +40°, 80% opacity)
    N circles where N ∝ mean token entropy. Total area ≤ EAS area.

Beam diversity  (base hue −35°, circle outlines)
    Outline colour and thickness of logit circles both encode
    beam_score_raw_std. Higher diversity = thicker, more varied outlines.

CAA displacement  (y offset)
    Strip centre offset ∝ CAA L2 displacement from no-context baseline.

Usage
-----
    python uncertainty_fingerprint.py \\
        --extended   extended_metrics_qwen.csv \\
        --pilot      pilot_summary.csv \\
        --output     fingerprint.png \\
        --model_name "Qwen2.5-7B"
"""

from __future__ import annotations

import argparse
import colorsys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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
BASE_COLORS = {
    "no_context":             "#888888",
    "direct_information":     "#2166AC",
    "implicature_information":"#1A9850",
    "stochastic_information": "#C0392B",
}

TIME_KEYS   = ["eas_early", "eas_mid", "eas_late", "eas_final_quarter"]
TIME_X      = np.array([0.12, 0.38, 0.65, 0.85])
TIME_LABELS = ["Early", "Mid", "Late", "Final Q"]

ROW_H           = 2.6
EAS_MAX_FILL    = 0.46
MAX_LOGIT_CIRC  = 8
LOGIT_H_MAX_REF = 0.55


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

def _rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

def hue_variant(base_hex, shift_deg, sat_factor=1.0, val_factor=1.0, min_sat=0.0):
    r, g, b = _hex_to_rgb(base_hex)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + shift_deg / 360) % 1.0
    s = min(1.0, max(min_sat, s * sat_factor))
    v = min(1.0, v * val_factor)
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return _rgb_to_hex(r2, g2, b2)

def make_triad(base):
    """
    Returns (eas_color, logit_color, beam_color):
      EAS   = base hue, desaturated slightly
      Logit = base hue +40 degrees, higher saturation
      Beam  = base hue -35 degrees, used for circle outlines
    """
    eas   = hue_variant(base,   0, sat_factor=0.80, val_factor=0.90)
    logit = hue_variant(base, +40, sat_factor=1.15, val_factor=1.05, min_sat=0.40)
    beam  = hue_variant(base, -35, sat_factor=1.15, val_factor=1.10, min_sat=0.35)
    return eas, logit, beam

TRIADS = {c: make_triad(BASE_COLORS[c]) for c in COND_ORDER}


# ── Data helpers ───────────────────────────────────────────────────────────────

def parse_seq(s, max_len=150):
    return np.array([max(float(x), 0.0) for x in str(s).split(";") if x.strip()][:max_len])

def mean_seqs_by_window(group_df):
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
        mat[i, :len(s)] = s
    mc = np.nanmean(mat, axis=0)
    n = len(mc)
    t1, t2, t3 = int(n/3), int(2*n/3), int(0.75*n)
    return [
        float(np.nanmean(mc[:t1])),
        float(np.nanmean(mc[t1:t2])),
        float(np.nanmean(mc[t2:])),
        float(np.nanmean(mc[t3:])),
    ]

def load_condition_data(em_path, pilot_path):
    em  = pd.read_csv(em_path)
    pil = pd.read_csv(pilot_path)
    if "caa_mean_l2" in pil.columns:
        em = em.merge(pil[["prompt_id","condition","caa_mean_l2"]],
                      on=["prompt_id","condition"], how="left")
    else:
        em["caa_mean_l2"] = np.nan

    cond_data = {}
    for cond in COND_ORDER:
        sub = em[em.condition == cond]
        if sub.empty:
            cond_data[cond] = dict(eas=[0.4]*4, logit_h=[0.4]*4,
                                   beam_std=0.1, caa_l2=np.nan)
            continue
        eas_vals = [float(sub[k].mean()) for k in TIME_KEYS if k in sub.columns]
        if len(eas_vals) < 4:
            eas_vals = [sub["eas_mean"].mean()] * 4 if "eas_mean" in sub.columns else [0.4]*4
        cond_data[cond] = dict(
            eas      = eas_vals,
            logit_h  = mean_seqs_by_window(sub),
            beam_std = float(sub["beam_score_raw_std"].mean()) if "beam_score_raw_std" in sub.columns else 0.1,
            caa_l2   = float(sub["caa_mean_l2"].mean()) if "caa_mean_l2" in sub.columns else np.nan,
        )
    return cond_data


# ── Geometry helpers ───────────────────────────────────────────────────────────

def eas_radius(val, row_h):
    return EAS_MAX_FILL * (row_h / 2) * np.sqrt(max(val, 0.01) / LOGIT_H_MAX_REF)

def caa_y_offset(caa_l2, all_l2, row_h):
    valid = np.array([v for v in all_l2 if not np.isnan(v)])
    if len(valid) < 2 or valid.max() == valid.min() or np.isnan(caa_l2):
        return 0.0
    return ((caa_l2 - valid.min()) / (valid.max() - valid.min()) - 0.5) * row_h * 0.30


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_eas_blob(ax, xpos, yc, eas_vals, row_h, eas_color):
    """EAS envelope — base hue, 22% opacity."""
    radii = [eas_radius(v, row_h) for v in eas_vals]
    for i in range(len(xpos) - 1):
        xs = np.linspace(xpos[i], xpos[i+1], 80)
        rs = np.interp(xs, [xpos[i], xpos[i+1]], [radii[i], radii[i+1]])
        ax.fill_between(xs, yc - rs, yc + rs, color=eas_color,
                        alpha=0.22, linewidth=0, zorder=1)
    for x, r in zip(xpos, radii):
        ax.add_patch(mpatches.Circle((x, yc), r, color=eas_color,
                                      alpha=0.22, zorder=2, linewidth=0))
        ax.add_patch(mpatches.Circle((x, yc), r, fill=False,
                                      edgecolor=eas_color, alpha=0.45,
                                      linewidth=0.9, zorder=3))
    return radii


def draw_logit_circles(ax, xpos, yc, logit_h, eas_radii,
                       beam_std, logit_color, beam_color, seed=42):
    """
    Logit circles — hue +40° fill; beam hue −35° outline.
    N circles ∝ entropy; outline thickness ∝ beam_std.
    """
    rng = np.random.default_rng(seed)
    outline_lw = 0.4 + float(np.clip(beam_std, 0, 0.5)) * 5.0

    for x, H, r_eas in zip(xpos, logit_h, eas_radii):
        if H < 0.01 or r_eas < 0.001:
            continue
        n = max(1, min(MAX_LOGIT_CIRC, round(MAX_LOGIT_CIRC * H / LOGIT_H_MAX_REF)))
        eas_area = np.pi * r_eas ** 2

        sigma = float(np.clip(beam_std, 0, 0.5)) * 1.6
        if sigma < 0.02 or n == 1:
            sizes = np.ones(n)
        else:
            raw   = rng.lognormal(0.0, sigma, n)
            sizes = raw / raw.mean()

        r_base = np.sqrt(eas_area / (np.pi * n))
        radii  = sizes * r_base

        if n == 1:
            centres = [(0.0, 0.0)]
        else:
            angles  = np.linspace(0, 2 * np.pi, n, endpoint=False)
            spread  = r_eas * 0.46
            centres = [(spread * np.cos(a), spread * np.sin(a)) for a in angles]

        for (dx, dy), r in zip(centres, radii):
            r_draw = min(r, r_eas * 0.82)
            ax.add_patch(mpatches.Circle(
                (x + dx, yc + dy), r_draw,
                color=logit_color, alpha=0.80, zorder=5, linewidth=0))
            ax.add_patch(mpatches.Circle(
                (x + dx, yc + dy), r_draw,
                fill=False, edgecolor=beam_color,
                alpha=0.72, linewidth=outline_lw, zorder=6))


# ── Main plot ──────────────────────────────────────────────────────────────────

def plot(em_path, pilot_path, out_path=None, model_name="LLM"):
    cond_data = load_condition_data(em_path, pilot_path)
    all_caa   = [cond_data[c]["caa_l2"] for c in COND_ORDER]
    N_CONDS   = len(COND_ORDER)
    X_LO, X_HI = 0.03, 0.97

    fig, axes = plt.subplots(
        N_CONDS, 1,
        figsize=(14, N_CONDS * ROW_H + 1.5),
        facecolor="#0E0E0E",
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.06, hspace=0.04)

    for ri, cond in enumerate(COND_ORDER):
        ax = axes[ri]
        ax.set_facecolor("#0E0E0E")
        ax.set_xlim(X_LO, X_HI)
        ax.set_ylim(-ROW_H / 2, ROW_H / 2)
        ax.axis("off")

        eas_col, logit_col, beam_col = TRIADS[cond]
        d    = cond_data[cond]
        yc   = caa_y_offset(d["caa_l2"], all_caa, ROW_H)
        xpos = X_LO + TIME_X * (X_HI - X_LO)

        # 1. EAS blob
        radii = draw_eas_blob(ax, xpos, yc, d["eas"], ROW_H, eas_col)

        # 2. Logit + beam circles
        draw_logit_circles(ax, xpos, yc, d["logit_h"], radii,
                           d["beam_std"], logit_col, beam_col)

        # 3. CAA arrow
        if abs(yc) > 0.04:
            ax.annotate("", xy=(X_LO + 0.009, yc),
                        xytext=(X_LO + 0.009, 0),
                        arrowprops=dict(arrowstyle="-|>", color=eas_col,
                                        alpha=0.55, lw=0.8), zorder=7)

        # Labels
        ax.text(X_LO + 0.012, ROW_H / 2 * 0.86, LABELS[cond],
                color="white", fontsize=9.5, fontweight="bold",
                va="top", ha="left")

        caa_str = f"{d['caa_l2']:.1f}" if not np.isnan(d["caa_l2"]) else "—"
        ax.text(X_HI - 0.008, ROW_H / 2 * 0.86,
                f"CAA L2 = {caa_str}  |  beam σ = {d['beam_std']:.3f}",
                color=eas_col, fontsize=7.5, va="top", ha="right", alpha=0.85)

        if ri == N_CONDS - 1:
            for xt, lb in zip(xpos, TIME_LABELS):
                ax.text(xt, -ROW_H / 2 * 0.88, lb,
                        color="#777777", fontsize=7.5, ha="center", va="bottom")

        if ri < N_CONDS - 1:
            ax.axhline(-ROW_H / 2 + 0.03, color="#222222", linewidth=0.7)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_ax = fig.add_axes([0.08, 0.005, 0.90, 0.050])
    legend_ax.set_facecolor("#0E0E0E")
    legend_ax.axis("off")
    ec, lc, bc = TRIADS["direct_information"]
    handles = [
        mpatches.Patch(color=ec, alpha=0.55,
                       label="EAS blob — entropy area score  (base hue, 22% opacity)"),
        mpatches.Patch(color=lc, alpha=0.80,
                       label="Logit circles — token entropy: count ∝ H  (hue +40°)"),
        mpatches.Patch(color=bc, alpha=0.72,
                       label="Circle outline — beam diversity: thickness ∝ beam_σ  (hue −35°)"),
        Line2D([0], [0], color="white", alpha=0.0, linewidth=0,
               marker=r"$\uparrow$", markersize=9,
               label="Y offset — CAA L2 displacement from no-context baseline"),
    ]
    legend_ax.legend(handles=handles, loc="center", ncol=2, fontsize=7.5,
                     framealpha=0, labelcolor="#CCCCCC",
                     handlelength=1.6, handletextpad=0.6, columnspacing=1.5)

    fig.suptitle(f"Multi-level uncertainty fingerprint — {model_name}",
                 fontsize=12, fontweight="bold", color="#E0E0E0", y=0.975)

    if out_path:
        plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#0E0E0E")
        print(f"Saved → {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--extended",   type=Path, required=True)
    p.add_argument("--pilot",      type=Path, required=True)
    p.add_argument("--output",     type=Path, default=None)
    p.add_argument("--model_name", type=str,  default="LLM")
    args = p.parse_args()
    plot(args.extended, args.pilot, args.output, args.model_name)
