"""
averages_bars.py — stand-alone port of the ipywidgets "Averages" viz.

Produces the fragmented-bar plot (navy = mean EAS, sky-blue extensions = SD,
purple ghost bars = entropy-without-EAS) for any of three modes:

    --mode global                         # all prompts averaged
    --mode domain --domain social         # averaged within a domain
    --mode single --question-id 0         # for a single question across conditions

Each run writes one PNG to --out (default: questions_x_context/data).

This is a direct port of the notebook code; it does not add new analysis, it
just lets the figure be regenerated outside Jupyter.  For the real
information-theoretic findings (half-life, front/back-loading, Wilcoxon
tests) use entropy_trajectory_analysis.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

TARGET_LEN = 150
COND_ORDER = [
    "no_context",
    "direct_information",
    "implicature_information",
    "stochastic_information",
]

NAVY = "#2c3e50"
SKY = "#87ceeb"
PURPLE = "#8e44ad"


def _parse_seq(s: object) -> np.ndarray:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.zeros(TARGET_LEN)
    try:
        vals = np.array(str(s).split(";"), dtype=float)
    except ValueError:
        return np.zeros(TARGET_LEN)
    return np.pad(vals[:TARGET_LEN], (0, max(0, TARGET_LEN - len(vals))), mode="constant")


def _get_metrics(subset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eas_list = [_parse_seq(r["eas_cross_sample_variance_profile"]) for _, r in subset.iterrows()]
    ent_list = [
        np.mean(
            [_parse_seq(r[f"token_entropy_sequence_sample_{j}"]) for j in range(10)],
            axis=0,
        )
        for _, r in subset.iterrows()
    ]
    m_eas = np.mean(eas_list, axis=0) if eas_list else np.zeros(TARGET_LEN)
    s_proxy = np.sqrt(np.maximum(m_eas, 0))  # keeps sky extensions visible where EAS exists
    m_ent = np.mean(ent_list, axis=0) if ent_list else np.zeros(TARGET_LEN)
    return m_eas, s_proxy, m_ent


def run_plot(df: pd.DataFrame, mode: str, domain: str | None, q_id: int, out_path: Path) -> None:
    if mode == "global":
        source_df = df
        title_suffix = "Global Average"
    elif mode == "domain":
        assert domain is not None, "--domain required when --mode domain"
        source_df = df[df["domain"] == domain.lower()]
        title_suffix = f"Domain: {domain}"
    elif mode == "single":
        source_df = df[df["question_id"] == q_id]
        title_suffix = f"Question ID {q_id}"
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    for ax, cond in zip(axes, COND_ORDER):
        subset = source_df[source_df["condition"] == cond]
        if subset.empty:
            ax.text(0.5, 0.5, "NO DATA FOUND", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cond.replace("_", " ").title(), fontweight="bold", loc="left")
            ax.set_xlim(0, TARGET_LEN)
            ax.set_ylim(-0.4, 0.4)
            continue

        m_eas, s_sd, m_ent = _get_metrics(subset)

        navy_patches, sky_patches, purple_patches = [], [], []

        for i in range(TARGET_LEN):
            h, sd, ent = m_eas[i], s_sd[i], m_ent[i]

            if h > 1e-4:
                mult = 6 if mode == "single" else 4
                n = max(1, int(np.ceil(ent * mult)))
                y_min = -h / 2
                fill_ratio = 0.5
                frag_h = (h * fill_ratio) / n
                gap_h = (h * (1.0 - fill_ratio)) / (n + 1)

                curr_y = y_min + gap_h
                for _ in range(n):
                    navy_patches.append(Rectangle((i + 0.1, curr_y), 0.8, frag_h))
                    curr_y += frag_h + gap_h

                if mode != "single" and sd > h:
                    ext_h = sd - h
                    n_ext = max(1, int(np.ceil(ent * 1.5)))
                    ext_frag_h = (ext_h * 0.5) / n_ext
                    ext_gap_h = (ext_h * 0.5) / (n_ext + 1)

                    ty = h / 2 + ext_gap_h
                    for _ in range(n_ext):
                        sky_patches.append(Rectangle((i + 0.1, ty), 0.8, ext_frag_h))
                        ty += ext_frag_h + ext_gap_h
                    by = -h / 2 - ext_gap_h - ext_frag_h
                    for _ in range(n_ext):
                        sky_patches.append(Rectangle((i + 0.1, by), 0.8, ext_frag_h))
                        by -= ext_frag_h + ext_gap_h

            elif ent > 0.05:
                gh = ent * 0.02
                purple_patches.append(Rectangle((i + 0.1, -gh / 2), 0.8, gh))

        ax.add_collection(PatchCollection(navy_patches, facecolor=NAVY, alpha=0.9, edgecolor="none"))
        ax.add_collection(PatchCollection(sky_patches, facecolor=SKY, alpha=0.6, edgecolor="none"))
        ax.add_collection(
            PatchCollection(purple_patches, facecolor=PURPLE, alpha=0.3, edgecolor="none", hatch="///")
        )

        ax.set_title(cond.replace("_", " ").title(), fontweight="bold", loc="left")
        ax.set_xlim(0, TARGET_LEN)
        ax.set_ylim(-0.4, 0.4)
        ax.axhline(0, color="black", alpha=0.2, linewidth=1)
        ax.set_facecolor("#fdfdfd")
        ax.grid(True, axis="y", alpha=0.1)

    fig.suptitle(f"EAS / Entropy fragmented bars — {title_suffix}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="Path to extended_metrics_*.csv")
    p.add_argument("--out", default="questions_x_context/data", help="Output directory")
    p.add_argument("--mode", choices=["global", "domain", "single"], default="global")
    p.add_argument("--domain", default=None,
                   help="Required when --mode domain (e.g. social, natural, economic)")
    p.add_argument("--question-id", type=int, default=0,
                   help="Used when --mode single; 0..14")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.csv)
    df["question_id"] = (df["prompt_id"] - 1) // 4

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "global":
        out_path = out_dir / "averages_bars_global.png"
    elif args.mode == "domain":
        out_path = out_dir / f"averages_bars_domain_{args.domain}.png"
    else:
        out_path = out_dir / f"averages_bars_q{args.question_id:02d}.png"

    run_plot(df, mode=args.mode, domain=args.domain, q_id=args.question_id, out_path=out_path)
