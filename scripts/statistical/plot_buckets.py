"""
plot_buckets.py
===============

Per-model bucket visualization showing how each context condition (DI / II / SI)
shapes the model's processing across three "buckets":

  1. State of mind  — internal representational trajectory across layers
  2. Output spread  — beam agreement at each layer of the prompt encoding
  3. Output timing  — when uncertainty is located within the generated response

Layout per model
----------------
  ┌──────────────────────────────────┬─────────────────────┬─────────────────────┐
  │                                  │ output spread (DI)  │ output timing (DI)  │
  │                                  ├─────────────────────┼─────────────────────┤
  │   STATE OF MIND                  │ output spread (II)  │ output timing (II)  │
  │   (single panel, 3 trajectories) ├─────────────────────┼─────────────────────┤
  │                                  │ output spread (SI)  │ output timing (SI)  │
  └──────────────────────────────────┴─────────────────────┴─────────────────────┘

Encoding choices (one per panel)
--------------------------------
STATE OF MIND
  - Single panel, all three conditions overlaid, common origin at (0, 0).
  - Each condition's path is built segment-by-segment, one segment per layer.
  - Segment length    ∝ caa_per_layer_l2 at that layer.
  - Segment angle     = (NC_cosine - condition_cosine) / max_dev * 45°, downward.
  - The total path length is normalized so the longest condition's trajectory
    fits in the panel — relative segment lengths preserved across conditions.
  - The horizontal arrow at y=0 represents "no displacement, no rotation"
    (i.e., the NC baseline).

OUTPUT SPREAD
  - One row per condition, layer index on x-axis.
  - Each layer rendered as a column of 10 dots.
  - High beam_per_layer_cosine → dots in a tight horizontal row (certain).
  - Low cosine → dots scatter vertically (uncertain).
  - Dot positions deterministic (fixed RNG seed per layer × condition) so the
    same data always produces the same image.

OUTPUT TIMING
  - Per-token entropy averaged across the 10 generated samples, binned.
  - Bar HEIGHT = mean entropy at that token position (fixed scale 0..2.5 nats
    so models are comparable).
  - Bar OPACITY = condition-level shape_PC2 mean (front-loaded vs back-loaded
    response shape), mapped onto a fixed PC2 range [-1.6, +1.6].
  - First bin DROPPED — the very first generated token always has high
    entropy (model choosing how to start) which dominates everything else and
    is uninformative about response shape.
  - Vertical dashed line marks the response halflife (token position at which
    cumulative entropy reaches 50% of the total) — also computed on the
    trimmed sequence so the start-of-response artifact doesn't pull it left.

Why these choices
-----------------
  - State-of-mind is collapsed into one panel because the meaningful comparison
    is across conditions in shared geometric space — splitting them lost that.
  - Output-spread and output-timing are split per condition because each row's
    pattern is read individually and the layer/token structure is busy enough
    that overlaying conditions hurts legibility.
  - Cross-model comparability requires fixed visual scales: ENTROPY_CEILING
    fixes the height range, PC2_RANGE fixes the opacity range. Don't change
    these between model figures or the cross-model comparison stops being valid.

Inputs (DATA_DIR configurable)
------------------------------
  extended_metrics_<model>.csv     — per-prompt entropy sequences and layer-wise data
  analysis_base.csv                — for shape_PC2 (computed by run_mlm_analysis.py)

Usage
-----
  python plot_buckets.py [--data-dir DATA_DIR] [--out-dir OUT_DIR]
                         [--models qwen,deepseek,llama,mistral,deepseek_v2_lite]

Dependencies
------------
  pandas numpy matplotlib

Outputs
-------
  buckets_<model>.png  per requested model
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (CONSTANTS THAT MUST BE THE SAME ACROSS MODELS)
# ─────────────────────────────────────────────────────────────────────────────

# Fixed scales for cross-model comparability
ENTROPY_CEILING = 2.5             # nats — y-axis max for output-timing bars
PC2_RANGE = (-1.6, +1.6)          # PC2 z-score range used for opacity mapping
PC2_OPACITY_RANGE = (0.10, 1.0)   # min/max bar opacity (widened for visibility)

# Display map: figure-name → file-stem-suffix in extended_metrics_<file_key>.csv
MODELS = {
    "Qwen":       "qwen",
    "DeepSeek":   "deepseek",
    "LLaMA":      "llama",
    "Mistral":    "mistral",
    "DS-V2-Lite": "deepseek_v2_lite",
}

COND_ORDER = ["direct_information", "implicature_information", "stochastic_information"]
COND_LBL = {
    "direct_information": "DI",
    "implicature_information": "II",
    "stochastic_information": "SI",
}
COND_COLOR = {
    "direct_information":      "#1976D2",
    "implicature_information": "#F57C00",
    "stochastic_information":  "#C62828",
}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def parse_seq(s) -> np.ndarray | None:
    """Parse semicolon-delimited string of floats."""
    if pd.isna(s) or not isinstance(s, str):
        return None
    try:
        vals = [float(x) for x in s.split(";") if x.strip()]
        return np.asarray(vals) if vals else None
    except ValueError:
        return None


def parse_layer_matrix(df: pd.DataFrame, col: str) -> np.ndarray | None:
    """Stack semicolon-delimited per-layer columns into (n_rows, n_layers) array."""
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


def get_token_entropy_mean(row: pd.Series, max_len: int = 140) -> np.ndarray | None:
    """
    Average per-token entropy across the 10 generated samples for one prompt.
    Truncates at max_len to avoid the EOS-cap artifact at position 150.
    """
    cols = [f"token_entropy_sequence_sample_{i}" for i in range(10)]
    seqs = [parse_seq(row[c]) for c in cols if c in row.index]
    seqs = [s for s in seqs if s is not None]
    if not seqs:
        return None
    padded = np.full((len(seqs), max_len), np.nan)
    for i, s in enumerate(seqs):
        n = min(len(s), max_len)
        padded[i, :n] = s[:n]
    return np.nanmean(padded, axis=0)


def pc2_to_opacity(pc2_val: float) -> float:
    """Map PC2 score to opacity using fixed cross-model range."""
    pc2_clipped = max(PC2_RANGE[0], min(PC2_RANGE[1], pc2_val))
    t = (pc2_clipped - PC2_RANGE[0]) / (PC2_RANGE[1] - PC2_RANGE[0])
    return PC2_OPACITY_RANGE[0] + t * (PC2_OPACITY_RANGE[1] - PC2_OPACITY_RANGE[0])


def compute_halflife_position(entropy_seq: np.ndarray, drop_first_n: int) -> int | None:
    """
    Token position where cumulative entropy reaches 50% of total.

    Computed on the SEQUENCE WITH THE FIRST `drop_first_n` TOKENS DROPPED, so
    the start-of-response artifact (the very high entropy of the first generated
    token) doesn't dominate the cumulative sum and pull halflife to the left.
    Returns position in the original (un-trimmed) sequence's coordinates.
    """
    if entropy_seq is None or len(entropy_seq) <= drop_first_n + 1:
        return None
    trimmed = entropy_seq[drop_first_n:]
    valid = trimmed[~np.isnan(trimmed)]
    if len(valid) < 2:
        return None
    total = np.sum(valid)
    if total <= 0:
        return None
    cum = np.cumsum(valid)
    half_idx = int(np.searchsorted(cum, 0.5 * total))
    # Convert back to position in the un-trimmed coordinate system
    return drop_first_n + half_idx


# ─────────────────────────────────────────────────────────────────────────────
# CORE DATA EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_condition_data(ext: pd.DataFrame, base_m: pd.DataFrame) -> dict:
    """
    For each condition, compute:
      - per-layer caa_l2 / caa_cos / beam_pl_cos averaged across prompts
      - per-token entropy averaged across prompts × samples
      - condition-level shape_PC2 mean (from analysis_base.csv)
    """
    caa_l2_mat  = parse_layer_matrix(ext, "caa_per_layer_l2")
    caa_cos_mat = parse_layer_matrix(ext, "caa_per_layer_cosine")
    beam_pl_mat = parse_layer_matrix(ext, "beam_per_layer_cosine")

    nc_idx = ext[ext["condition"] == "no_context"].index.tolist()
    nc_cos = np.nanmean(caa_cos_mat[nc_idx], axis=0) if caa_cos_mat is not None else None

    cond_data = {"_nc_cos": nc_cos}
    for cond in COND_ORDER:
        idx = ext[ext["condition"] == cond].index.tolist()
        if not idx:
            continue
        token_means = []
        for i in idx:
            tm = get_token_entropy_mean(ext.loc[i])
            if tm is not None:
                token_means.append(tm)
        token_avg = (np.nanmean(np.stack(token_means), axis=0)
                     if token_means else np.zeros(140))
        pc2_mean = float(base_m[base_m["condition"] == cond]["shape_PC2"].mean())
        cond_data[cond] = {
            "caa_l2":        np.nanmean(caa_l2_mat[idx], axis=0),
            "caa_cos":       np.nanmean(caa_cos_mat[idx], axis=0),
            "beam_pl_cos":   np.nanmean(beam_pl_mat[idx], axis=0),
            "token_entropy": token_avg,
            "pc2_mean":      pc2_mean,
        }
    return cond_data


# ─────────────────────────────────────────────────────────────────────────────
# PANEL DRAWING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def draw_state_of_mind(ax, cond_data: dict, n_layers: int) -> None:
    """
    Single panel — all three conditions as continuous segment trajectories from
    a shared origin at (0, 0). Each segment one layer, length ∝ caa_l2,
    angle ∝ (1 - cosine_to_NC) mapped to a 45° downward fan.
    """
    nc_cos = cond_data["_nc_cos"]

    # Compute max deviation across all conditions × layers for fan-scale normalization
    all_devs = np.concatenate([
        nc_cos - cond_data[c]["caa_cos"] for c in COND_ORDER if c in cond_data
    ])
    max_dev = float(np.nanmax(all_devs)) if np.any(~np.isnan(all_devs)) else 1.0
    if max_dev < 1e-8:
        max_dev = 1.0

    # Total trajectory length per condition (unscaled). Use the largest to
    # determine the visual normalization so the longest fits panel_width.
    panel_width = 1.0
    total_l2_per_cond = {
        c: float(np.nansum(cond_data[c]["caa_l2"])) for c in COND_ORDER if c in cond_data
    }
    max_total_l2 = max(total_l2_per_cond.values()) if total_l2_per_cond else 1.0
    normalize_scale = panel_width / max_total_l2 if max_total_l2 > 0 else 1.0

    # Setup axes
    ax.set_facecolor("white")
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # NC reference arrow
    ax.annotate("", xy=(panel_width * 1.02, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#BBB", lw=0.7))
    ax.text(panel_width * 1.02, 0.015, "NC reference", fontsize=8, color="#888", ha="right")

    # Plot each condition's trajectory
    for cond in COND_ORDER:
        if cond not in cond_data:
            continue
        color = COND_COLOR[cond]
        d = cond_data[cond]
        prev_x, prev_y = 0.0, 0.0
        for L in range(n_layers):
            l2  = d["caa_l2"][L]
            cos = d["caa_cos"][L]
            if np.isnan(l2) or np.isnan(cos):
                continue
            length = l2 * normalize_scale
            dev = nc_cos[L] - cos
            angle_deg = (dev / max_dev) * 45.0
            angle_rad = np.deg2rad(-angle_deg)  # downward = negative y
            next_x = prev_x + length * np.cos(angle_rad)
            next_y = prev_y + length * np.sin(angle_rad)
            ax.plot([prev_x, next_x], [prev_y, next_y], color=color,
                    lw=2.5, alpha=0.85, solid_capstyle="round")
            prev_x, prev_y = next_x, next_y
        # Endpoint marker + label
        ax.scatter([prev_x], [prev_y], color=color, s=80, zorder=10,
                   edgecolors="white", lw=1.0)
        ax.text(prev_x + 0.025, prev_y - 0.012, COND_LBL[cond],
                fontsize=11, fontweight="bold", color=color, va="center")

    ax.set_xlim(-0.05, panel_width * 1.15)
    ax.set_ylim(-panel_width * 0.95, panel_width * 0.15)
    ax.set_title("State of mind\n"
                 "segment length = CAA L2  |  segment angle = CAA cosine deviation from NC",
                 fontsize=10, pad=10)
    ax.text(0.005, 0.97, "origin (layer 0)", fontsize=8, color="#888",
            transform=ax.transAxes, ha="left", va="top")
    ax.text(0.005, 0.03, "each line: 1 segment per layer →", fontsize=7,
            color="#888", transform=ax.transAxes, ha="left")


def draw_output_spread(ax, cond: str, cond_data: dict, n_layers: int,
                        beam_min: float, beam_range: float, is_top_row: bool) -> None:
    """
    Per-condition row showing 10 dots per layer.
    Tight cosine → dots in a flat row. Low cosine → dots scattered vertically.
    Dot positions are deterministic via fixed RNG seed.
    """
    color = COND_COLOR[cond]
    d = cond_data[cond]

    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)

    for L in range(n_layers):
        beam_cos = d["beam_pl_cos"][L]
        if np.isnan(beam_cos):
            continue
        # Map cosine into [0..1]: 1 at max cosine (tight), 0 at min cosine (loose)
        t = (beam_cos - beam_min) / beam_range if beam_range > 0 else 1.0
        spread_y = (1 - t) * 0.45
        # Deterministic vertical jitter
        rng = np.random.default_rng(seed=L * 100 + hash(cond) % 1000)
        y_offsets = (rng.uniform(-spread_y, spread_y, size=10)
                     if spread_y > 0.005 else np.zeros(10))
        x_positions = np.linspace(L - 0.25, L + 0.25, 10)
        ax.scatter(x_positions, y_offsets, color=color, s=8, alpha=0.7, edgecolors="none")

    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xticks([0, n_layers // 2, n_layers - 1])
    ax.set_xticklabels(["layer 0", f"L{n_layers // 2}", f"L{n_layers - 1}"], fontsize=8)
    ax.text(-0.07, 0.5, COND_LBL[cond], fontsize=13, fontweight="bold",
            color=color, ha="right", va="center", transform=ax.transAxes)
    if is_top_row:
        ax.set_title("Output spread\nrow = certain  |  scatter = uncertain",
                     fontsize=10, pad=8)


def draw_output_timing(ax, cond: str, cond_data: dict, is_top_row: bool,
                        n_bins: int = 25) -> None:
    """
    Per-token entropy as binned bars. First bin DROPPED (start-of-response artifact).
    Bar height = mean entropy in bin (fixed scale 0..ENTROPY_CEILING).
    Bar opacity = condition-level PC2 (fixed scale).
    Vertical dashed line = response halflife computed on the trimmed sequence.
    """
    color = COND_COLOR[cond]
    d = cond_data[cond]

    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)

    te = d["token_entropy"]
    valid_te = te[~np.isnan(te)]
    n_tokens = len(valid_te)
    if n_tokens < n_bins:
        return

    chunks = np.array_split(valid_te, n_bins)
    bin_means = np.array([np.nanmean(c) for c in chunks])
    bin_centers = np.linspace(0.5, n_tokens - 0.5, n_bins)
    bar_alpha = pc2_to_opacity(d["pc2_mean"])

    # Draw bars (skip first bin which contains the start-of-response artifact)
    for j, (x, h_raw) in enumerate(zip(bin_centers, bin_means)):
        if j == 0:
            continue
        height = min(h_raw, ENTROPY_CEILING) / ENTROPY_CEILING
        ax.bar(x, height, width=(n_tokens / n_bins) * 0.9,
               bottom=0, color=color, alpha=bar_alpha, edgecolor="none")

    # Halflife marker — computed on the trimmed sequence (drop first bin's tokens)
    drop_n = max(1, n_tokens // n_bins)  # number of tokens in the first bin
    halflife_pos = compute_halflife_position(valid_te, drop_first_n=drop_n)
    if halflife_pos is not None:
        ax.axvline(halflife_pos, color="black", lw=1.0, ls="--", alpha=0.55, zorder=5)
        ax.text(halflife_pos, 1.02, "halflife", fontsize=7, color="black", alpha=0.7,
                ha="center", va="bottom")

    ax.set_xlim(0, n_tokens)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", f"{ENTROPY_CEILING/2:.1f}", f"{ENTROPY_CEILING}"], fontsize=8)
    ax.set_xticks([0, n_tokens // 2, n_tokens - 1])
    ax.set_xticklabels(["token 0", f"tok {n_tokens // 2}", f"tok {n_tokens - 1}"], fontsize=8)
    ax.set_ylabel("entropy (nats)", fontsize=8)
    if is_top_row:
        ax.set_title("Output timing\nbar height = entropy  |  opacity = PC2 (front-loaded → bright)",
                     fontsize=10, pad=8)
    # PC2 annotation
    ax.text(0.99, 0.95, f"PC2 = {d['pc2_mean']:+.2f}",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            color="#444", bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PLOT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_model(model_name: str, file_key: str, data_dir: Path, base_path: Path,
                out_path: Path) -> None:
    ext_path = data_dir / "model" / file_key / f"extended_metrics_{file_key}.csv"
    if not ext_path.exists():
        print(f"  [skip] {ext_path} not found")
        return
    ext = pd.read_csv(ext_path)

    base = pd.read_csv(base_path)
    base_m = base[base["model"] == model_name]
    if base_m.empty:
        print(f"  [skip] no rows for {model_name} in {base_path}")
        return

    cond_data = extract_condition_data(ext, base_m)
    n_layers = len(cond_data["_nc_cos"])

    # Determine beam scatter normalization (shared across conditions in this model)
    all_beam_cos = np.concatenate([
        cond_data[c]["beam_pl_cos"] for c in COND_ORDER if c in cond_data
    ])
    beam_min, beam_max = float(np.nanmin(all_beam_cos)), float(np.nanmax(all_beam_cos))
    beam_range = beam_max - beam_min if beam_max > beam_min else 1.0

    # Build figure
    fig = plt.figure(figsize=(16, 8.5))
    fig.patch.set_facecolor("#FAFAFA")
    outer_gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.13,
                                  width_ratios=[1.4, 1.2])

    ax_state = fig.add_subplot(outer_gs[0, 0])
    draw_state_of_mind(ax_state, cond_data, n_layers)

    inner_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[0, 1],
                                                  wspace=0.13, hspace=0.55)
    for row, cond in enumerate(COND_ORDER):
        if cond not in cond_data:
            continue
        ax_spread = fig.add_subplot(inner_gs[row, 0])
        draw_output_spread(ax_spread, cond, cond_data, n_layers,
                            beam_min, beam_range, is_top_row=(row == 0))

    fig.suptitle(
        f"{model_name}: how each condition shapes processing\n"
        f"Left: state-of-mind trajectories (all 3 conditions, common origin) | "
        f"Right: per-condition output spread (layer x-axis)",
        fontsize=11, fontweight="bold", y=0.99,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    _repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-dir", type=Path,
                        default=_repo / "data",
                        help="Repo data/ directory (expects data/model/<name>/ subdirectories)")
    parser.add_argument("--base-path", type=Path,
                        default=_repo / "results" / "MLM" / "analysis_base.csv",
                        help="Path to analysis_base.csv produced by run_mlm_analysis.py")
    parser.add_argument("--out-dir", type=Path,
                        default=_repo / "results" / "plots",
                        help="Where to write buckets_<model>.png files")
    parser.add_argument("--models", type=str, default=",".join(MODELS.keys()),
                        help="Comma-separated list of model display names to plot. "
                             f"Available: {', '.join(MODELS.keys())}")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    for model_name in requested:
        if model_name not in MODELS:
            print(f"  [skip] unknown model '{model_name}' — available: {list(MODELS.keys())}")
            continue
        file_key = MODELS[model_name]
        out_path = args.out_dir / f"buckets_{file_key}.png"
        print(f"Plotting {model_name}...")
        plot_model(model_name, file_key, args.data_dir, args.base_path, out_path)


if __name__ == "__main__":
    main()
