"""
sbert_similarity.py — Compute context–question cosine similarity via SBERT,
with token-count analysis and optional entropy visualisation.

For each row where context is non-null/non-empty, embeds context and question
independently using the configured SBERT model, then computes cosine similarity.
Runs the embedding twice with two different random seeds to check numerical
stability; flags rows where the two runs differ by more than 0.01.

Output CSV is a drop-in replacement for the input prompts CSV (all original
columns preserved) with additional columns:
  context_question_similarity  – cosine similarity (run 1, canonical value)
  similarity_variance_flag     – 1 if |run1 - run2| > 0.01, else 0
  ctx_tokens                   – context subword token count (context rows only)
  q_tokens                     – question subword token count (context rows only)
  combined_tokens              – ctx + question token count (context rows only)
  token_limit_flag             – 1 if combined_tokens >= 400, else 0

Plots written to questions_x_context/plots/ (fixed location, always):
  token_counts.png   – boxplots of question and context token counts per condition
  entropy_scores.png – boxplots of mean_token_entropy and semantic_entropy
                       per condition (only when --results_csv is supplied)

Usage (from llm_entropy_study/):
    python questions_x_context/sbert_similarity.py \\
        questions_x_context/data/prompts/prompts_v3_full.csv

    python questions_x_context/sbert_similarity.py \\
        questions_x_context/data/prompts/prompts_v3_full.csv \\
        --results_csv questions_x_context/results/pilot_summary.csv \\
        --output questions_x_context/data/prompts/prompts_v3_full_sim.csv
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

SBERT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
STABILITY_THRESHOLD = 0.01
TOKEN_WARN_THRESHOLD = 400  # conservative margin below 512-token SBERT limit

_HERE = Path(__file__).resolve().parent   # questions_x_context/
PLOTS_DIR = _HERE / "plots"               # always questions_x_context/plots/

# Canonical condition order and display labels
CONDITIONS = [
    "no_context",
    "stochastic_information",
    "implicature_information",
    "direct_information",
]
COND_LABELS = {
    "no_context": "No\nContext",
    "stochastic_information": "Stochastic\nInfo",
    "implicature_information": "Implicature\nInfo",
    "direct_information": "Direct\nInfo",
}


# ── Seeding ────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Embedding helpers ──────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings, shape (N, D)."""
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    return embs / norms


def run_similarity_pass(
    model: SentenceTransformer,
    contexts: list[str],
    questions: list[str],
    seed: int,
) -> list[float]:
    """Embed contexts and questions under a given seed, return per-row cosine sims."""
    set_seed(seed)
    ctx_embs = embed_texts(model, contexts)
    set_seed(seed)
    q_embs = embed_texts(model, questions)
    return [cosine_similarity(ctx_embs[i], q_embs[i]) for i in range(len(contexts))]


# ── Token counting ─────────────────────────────────────────────────────────────
def count_tokens(tokenizer, texts: list[str]) -> list[int]:
    """Return subword token counts for each text using the SBERT tokenizer."""
    return [
        len(tokenizer.encode(t, add_special_tokens=False))
        for t in texts
    ]


# ── Plotting helpers ───────────────────────────────────────────────────────────
def _boxplot_by_condition(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    """
    Draw a boxplot on ax, one box per condition in CONDITIONS order.
    Conditions absent from df produce no box for that slot.
    Overlays individual data points (jittered) so low-N runs are readable.

    Condition matching is normalised (strip whitespace, lowercase,
    spaces→underscores) so minor formatting differences in the CSV don't
    silently produce empty plots.
    """
    # Normalise condition strings so "Irrelevant_Context ", "irrelevant context",
    # etc. all match the canonical "irrelevant_context" key.
    norm_cond = (
        df["condition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    data_by_cond = [
        df.loc[norm_cond == cond, value_col].dropna().tolist()
        for cond in CONDITIONS
    ]
    labels = [COND_LABELS[c] for c in CONDITIONS]

    positions = [i + 1 for i, d in enumerate(data_by_cond) if d]
    plot_data = [d for d in data_by_cond if d]
    plot_labels = [labels[i] for i, d in enumerate(data_by_cond) if d]

    if not plot_data:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    ax.boxplot(
        plot_data,
        positions=positions,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(facecolor=color, alpha=0.55),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    # Overlay individual points (jittered) so low-N runs show all values
    rng = np.random.default_rng(0)
    for pos, vals in zip(positions, plot_data):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            [pos + j for j in jitter],
            vals,
            color="black",
            s=18,
            alpha=0.7,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(plot_labels, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_token_counts(df: pd.DataFrame) -> None:
    """
    Two-panel figure: question token counts and context token counts per condition.
    Context panel only shows rows where context is non-empty.
    Saved to PLOTS_DIR/token_counts.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Token Counts by Condition", fontsize=13, fontweight="bold", y=1.01)

    _boxplot_by_condition(
        axes[0], df,
        value_col="q_tokens",
        title="Question Token Count",
        ylabel="Tokens (SBERT tokenizer)",
        color="#4C72B0",
    )

    ctx_df = df[df["context"].str.len() > 0].copy()
    _boxplot_by_condition(
        axes[1], ctx_df,
        value_col="ctx_tokens",
        title="Context Token Count\n(context rows only)",
        ylabel="Tokens (SBERT tokenizer)",
        color="#DD8452",
    )

    fig.tight_layout()
    out = PLOTS_DIR / "token_counts.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out}")


def plot_entropy_scores(results_df: pd.DataFrame) -> None:
    """
    Two-panel figure: mean_token_entropy and semantic_entropy per condition.
    Saved to PLOTS_DIR/entropy_scores.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Entropy Scores by Condition", fontsize=13, fontweight="bold", y=1.01)

    _boxplot_by_condition(
        axes[0], results_df,
        value_col="mean_token_entropy",
        title="Mean Token Entropy\n(logit entropy)",
        ylabel="Entropy (nats)",
        color="#55A868",
    )

    _boxplot_by_condition(
        axes[1], results_df,
        value_col="semantic_entropy",
        title="Semantic Entropy\n(NLI cluster entropy)",
        ylabel="Entropy (nats)",
        color="#C44E52",
    )

    fig.tight_layout()
    out = PLOTS_DIR / "entropy_scores.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Add context–question SBERT cosine similarity to a prompts CSV "
            "and produce diagnostic plots."
        )
    )
    p.add_argument(
        "prompts_csv",
        type=Path,
        help="Path to input prompts CSV (e.g. prompts_v5.csv)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path for output CSV. Defaults to <input_stem>_sim.csv "
            "in the same directory."
        ),
    )
    p.add_argument(
        "--results_csv",
        type=Path,
        default=None,
        help=(
            "Path to pilot_summary.csv (output of pilot_beam_fix.py). "
            "When provided, produces entropy_scores.png in addition to "
            "token_counts.png."
        ),
    )
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    in_path: Path = args.prompts_csv.resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"Prompts CSV not found: {in_path}")

    out_path: Path = (
        args.output.resolve()
        if args.output
        else in_path.with_name(in_path.stem + "_sim.csv")
    )
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Input  : {in_path}")
    print(f"Output : {out_path}")
    print(f"Plots  : {PLOTS_DIR}")

    # ── Load prompts data ──────────────────────────────────────────────────────
    df = pd.read_csv(in_path)
    df.columns = [c.strip() for c in df.columns]
    df["context"] = df["context"].fillna("").astype(str).str.strip().replace("nan", "")

    has_context = df["context"].str.len() > 0
    ctx_rows = df[has_context].copy()
    print(f"Rows with context: {len(ctx_rows)} / {len(df)}")

    if "condition" in df.columns:
        print(f"Unique condition values in CSV: {sorted(df['condition'].dropna().unique().tolist())}")
    else:
        print("[WARNING] No 'condition' column found — boxplots will show no data")

    # ── Load SBERT model (must come before tokenizer use) ─────────────────────
    print(f"\nLoading SBERT model: {SBERT_MODEL}")
    model = SentenceTransformer(SBERT_MODEL)
    tokenizer = model.tokenizer

    # ── Token count analysis ───────────────────────────────────────────────────
    # Compute for ALL rows using the full df columns — this avoids .loc
    # write-back alignment issues and ensures every condition has data for
    # the boxplots.  Empty context strings produce 0 tokens naturally.
    print("\nToken count analysis (Qwen3-Embedding tokenizer):")
    df["q_tokens"] = count_tokens(tokenizer, df["question"].tolist())
    df["ctx_tokens"] = count_tokens(tokenizer, df["context"].tolist())
    df["combined_tokens"] = df["q_tokens"] + df["ctx_tokens"]
    df["token_limit_flag"] = (df["combined_tokens"] >= TOKEN_WARN_THRESHOLD).astype(int)

    # Per-row warnings for context rows that approach the 512-token limit
    for orig_idx, row in df[has_context].iterrows():
        if row["token_limit_flag"]:
            pid = row["prompt_id"] if "prompt_id" in df.columns else orig_idx
            print(
                f"  [TOKEN WARNING] prompt_id={pid} — "
                f"combined={int(row['combined_tokens'])} tokens "
                f"(ctx={int(row['ctx_tokens'])}, q={int(row['q_tokens'])}) "
                f"— approaching 512 limit"
            )

    ctx_combined = df.loc[has_context, "combined_tokens"]
    print(
        f"  Combined token counts (context rows) — "
        f"mean={ctx_combined.mean():.1f}  "
        f"max={int(ctx_combined.max())}  "
        f"flagged={int(df.loc[has_context, 'token_limit_flag'].sum())}"
    )

    # contexts / questions lists used by the similarity pass below
    contexts = ctx_rows["context"].tolist()
    questions = ctx_rows["question"].tolist()

    # ── Pick two random seeds, log them ───────────────────────────────────────
    rng = random.Random(0)
    seed1 = rng.randint(1, 99999)
    seed2 = rng.randint(1, 99999)
    while seed2 == seed1:
        seed2 = rng.randint(1, 99999)

    print(f"\nRun 1 seed: {seed1}")
    print(f"Run 2 seed: {seed2}")

    # ── Two embedding passes ───────────────────────────────────────────────────
    print("\nRun 1: embedding contexts and questions...")
    sims_run1 = run_similarity_pass(model, contexts, questions, seed1)

    print("Run 2: embedding contexts and questions...")
    sims_run2 = run_similarity_pass(model, contexts, questions, seed2)

    # ── Stability check ────────────────────────────────────────────────────────
    diffs = [abs(s1 - s2) for s1, s2 in zip(sims_run1, sims_run2)]
    flags = [1 if d > STABILITY_THRESHOLD else 0 for d in diffs]

    n_flagged = sum(flags)
    if n_flagged:
        print(
            f"\n[WARNING] {n_flagged} row(s) show |run1 - run2| > {STABILITY_THRESHOLD} "
            f"(numerical instability):"
        )
        for idx, (orig_idx, diff, s1, s2) in enumerate(
            zip(ctx_rows.index, diffs, sims_run1, sims_run2)
        ):
            if flags[idx]:
                pid = df.loc[orig_idx, "prompt_id"] if "prompt_id" in df.columns else orig_idx
                print(
                    f"  prompt_id={pid}  run1={s1:.6f}  run2={s2:.6f}  diff={diff:.6f}"
                )
    else:
        print(f"\nAll rows stable (|run1 - run2| ≤ {STABILITY_THRESHOLD}).")

    # ── Write similarity columns back to full DataFrame ───────────────────────
    df["context_question_similarity"] = np.nan
    df["similarity_variance_flag"] = np.nan
    df.loc[ctx_rows.index, "context_question_similarity"] = sims_run1
    df.loc[ctx_rows.index, "similarity_variance_flag"] = flags

    df.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")

    # ── Summary stats ──────────────────────────────────────────────────────────
    print(
        f"\nSimilarity stats (run 1):"
        f"\n  mean={np.mean(sims_run1):.4f}  std={np.std(sims_run1):.4f}"
        f"\n  min={np.min(sims_run1):.4f}   max={np.max(sims_run1):.4f}"
        f"\n  seeds used: run1={seed1}, run2={seed2}"
    )
    print(
        f"\nToken count stats:"
        f"\n  question tokens (all rows)     — "
        f"mean={df['q_tokens'].mean():.1f}  "
        f"min={int(df['q_tokens'].min())}  max={int(df['q_tokens'].max())}"
        f"\n  context tokens (context rows)  — "
        f"mean={df.loc[has_context, 'ctx_tokens'].mean():.1f}  "
        f"min={int(df.loc[has_context, 'ctx_tokens'].min())}  "
        f"max={int(df.loc[has_context, 'ctx_tokens'].max())}"
        f"\n  combined tokens (context rows) — "
        f"mean={df.loc[has_context, 'combined_tokens'].mean():.1f}  "
        f"min={int(df.loc[has_context, 'combined_tokens'].min())}  "
        f"max={int(df.loc[has_context, 'combined_tokens'].max())}"
    )

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_token_counts(df)

    if args.results_csv is not None:
        res_path = args.results_csv.resolve()
        if not res_path.exists():
            print(f"[WARNING] --results_csv not found, skipping entropy plot: {res_path}")
        else:
            results_df = pd.read_csv(res_path)
            results_df.columns = [c.strip() for c in results_df.columns]
            missing = [
                c for c in ("condition", "mean_token_entropy", "semantic_entropy")
                if c not in results_df.columns
            ]
            if missing:
                print(
                    f"[WARNING] results CSV missing columns {missing}, "
                    f"skipping entropy plot."
                )
            else:
                plot_entropy_scores(results_df)
    else:
        print(
            "  (entropy_scores.png skipped — pass --results_csv pilot_summary.csv "
            "to enable)"
        )


if __name__ == "__main__":
    main()
