"""
caa_variance_sanity.py — Sanity-check CAA displacement as an IB "effort" variable.

For each instruct model with an extended_metrics CSV, checks whether caa_mean_l2
varies meaningfully *within* conditions (DI/II/SI) or is dominated by between-
condition differences.  Base models (qwen_base, llama_base) are absent because
the CAA pipeline was never run on them; see report header.

Extension (v2): also computes caa_at_peak_layer (L2 at each model's empirically-
derived peak layer from the patched RDM notebook) and produces a second ANOVA
table plus a side-by-side comparison.

Outputs (all to questions_x_context/results/probe_inventory/):
  caa_variance_sanity.md                 — ANOVA tables + descriptive tables
  effort_utility_scatter_{model}.png     — scatter: x=caa_mean_l2, y=AV, colour=condition
"""
from __future__ import annotations

import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent.parent
OUT_DIR  = ROOT / "results" / "probe_inventory"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT   = OUT_DIR / "caa_variance_sanity.md"

SCORES_PATH = ROOT / "data" / "human" / "QuestionContext_Scores.csv"

# Models
MODELS_INSTRUCT = ["deepseek", "deepseek_v2_lite", "llama", "mistral", "qwen"]
MODELS_BASE     = ["llama_base", "qwen_base"]
QWEN_NAN_WARN   = "qwen"   # hidden states 100% NaN — values unreliable

# Peak layers per model — empirically derived from the patched RDM notebook
# (cell 22 output: argmax of partial-rho profile over layers, last_token_hs,
# standardised, partial correlation controlling for prompt_token_len).
# Source: rdm_analysis_colab_fixed_again.ipynb, printed table "model L* partial rho"
PEAK_LAYERS: dict[str, int] = {
    "deepseek":         15,
    "deepseek_v2_lite": 12,
    "llama":            15,
    "mistral":          16,
    "qwen":             19,   # NaN-masked; included for completeness only
}

CONDITION_ORDER   = ["no_context", "direct_information",
                     "implicature_information", "stochastic_information"]
CONDITION_ALIASES = {
    "no_context":              "NC",
    "direct_information":      "DI",
    "implicature_information": "II",
    "stochastic_information":  "SI",
}
COND_COLORS = {
    "direct_information":      "tab:blue",
    "implicature_information": "tab:orange",
    "stochastic_information":  "tab:green",
}

NON_NC = ["direct_information", "implicature_information", "stochastic_information"]

# ─── Statistics helpers ────────────────────────────────────────────────────────
def eta_squared(groups: list[np.ndarray]) -> float:
    all_vals   = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total   = float(np.sum((all_vals - grand_mean) ** 2))
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def verdict(eta2: float) -> str:
    if eta2 < 0.30:
        return "OK"
    if eta2 <= 0.60:
        return "MARGINAL"
    return "FAIL"


def anova_stats(df: pd.DataFrame, col: str) -> dict:
    """One-way ANOVA (DI/II/SI) on `col`.  Returns within-SDs, F, p, eta2, verdict."""
    non_nc = df[df["condition"] != "no_context"]
    groups = {
        cond: non_nc.loc[non_nc["condition"] == cond, col].dropna().values
        for cond in NON_NC
    }
    f_stat, p_val = stats.f_oneway(*groups.values())
    eta2          = eta_squared(list(groups.values()))
    return {
        "within_SD_DI":  float(np.std(groups["direct_information"],      ddof=1)),
        "within_SD_II":  float(np.std(groups["implicature_information"],  ddof=1)),
        "within_SD_SI":  float(np.std(groups["stochastic_information"],   ddof=1)),
        "F":             float(f_stat),
        "p":             float(p_val),
        "eta2":          eta2,
        "verdict":       verdict(eta2),
    }


# ─── Peak-layer column extraction ─────────────────────────────────────────────
def add_peak_layer_col(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Parse the semicolon-delimited `caa_per_layer_l2` column and extract the
    value at this model's peak layer into a new `caa_at_peak_layer` column.
    NC rows are always 0.0 (baseline = itself).
    Returns df with the new column added in-place.
    """
    if "caa_per_layer_l2" not in df.columns:
        df["caa_at_peak_layer"] = np.nan
        return df

    peak_l = PEAK_LAYERS.get(model)
    if peak_l is None:
        df["caa_at_peak_layer"] = np.nan
        return df

    def _extract(s: str) -> float:
        parts = str(s).split(";")
        if peak_l >= len(parts):
            return np.nan
        try:
            return float(parts[peak_l])
        except ValueError:
            return np.nan

    df = df.copy()
    df["caa_at_peak_layer"] = df["caa_per_layer_l2"].apply(_extract)
    return df


# ─── Data loading ──────────────────────────────────────────────────────────────
def _model_folder(model: str) -> str:
    return "deepseek" if model.startswith("deepseek") else model


def find_csv(model: str) -> Path | None:
    p = ROOT / "data" / "model" / _model_folder(model) / f"extended_metrics_{model}.csv"
    return p if p.exists() else None


def load_metrics(model: str) -> tuple[pd.DataFrame | None, Path | None]:
    p = find_csv(model)
    if p is None:
        return None, None
    df = pd.read_csv(p)
    required = {"prompt_id", "condition", "caa_mean_l2"}
    if not required.issubset(df.columns):
        return None, p
    df = add_peak_layer_col(df, model)
    return df, p


def load_scores() -> pd.DataFrame:
    df = pd.read_csv(SCORES_PATH)
    df = df[df["Num"].notna() & df["AV"].notna()].copy()
    df["prompt_id"] = df["Num"].astype(int)
    return df[["prompt_id", "Condition", "AV", "SEM"]]


# ─── Markdown helpers ──────────────────────────────────────────────────────────
def _md_row(cells: list) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _p_str(p: float) -> str:
    return f"{p:.4f}" if p >= 0.0001 else "< 0.0001"


def _nan_tag_md(model: str) -> str:
    return " ⚠" if model == QWEN_NAN_WARN else ""


# ─── Descriptive table (mean-layer only; peak-layer descriptives omitted for brevity) ──
def descriptive_md(df: pd.DataFrame, model: str) -> str:
    nan_tag = " ⚠ hidden states NaN" if model == QWEN_NAN_WARN else ""
    lines = [f"### {model}{nan_tag}\n",
             "| Condition | n | mean | SD | min | max |",
             "| --------- | - | ---- | -- | --- | --- |"]
    for cond in CONDITION_ORDER:
        subset = df.loc[df["condition"] == cond, "caa_mean_l2"].dropna()
        alias  = CONDITION_ALIASES[cond]
        if len(subset) == 0:
            lines.append(f"| {alias} | 0 | – | – | – | – |")
        else:
            lines.append(
                f"| {alias} | {len(subset)} "
                f"| {subset.mean():.4f} "
                f"| {subset.std(ddof=1):.4f} "
                f"| {subset.min():.4f} "
                f"| {subset.max():.4f} |"
            )
    return "\n".join(lines)


# ─── ANOVA summary table (generic, works for either column) ───────────────────
def anova_md_table(model_data: dict, col: str, label: str) -> list[str]:
    hdr = ["model", "within_SD_DI", "within_SD_II", "within_SD_SI",
           "F", "p", "eta_squared", "verdict"]
    lines = [
        f"## ANOVA — {label}\n",
        "η² interpretation: **< 0.30 → OK** (within-condition variance dominates); "
        "**0.30–0.60 → MARGINAL**; **> 0.60 → FAIL** (CAA encodes condition identity).\n",
        _md_row(hdr),
        _md_row(["---"] * len(hdr)),
    ]
    for model in MODELS_INSTRUCT:
        if model not in model_data:
            lines.append(_md_row([f"{model} *(no data)*"] + ["–"] * (len(hdr) - 1)))
            continue
        s       = model_data[model][col]
        nt      = _nan_tag_md(model)
        verd    = s["verdict"] + nt
        has_peak = not np.isnan(s["eta2"]) if col == "peak" else True
        if not has_peak:
            lines.append(_md_row([f"{model}{nt}", "–", "–", "–", "–", "–", "–", "–"]))
            continue
        lines.append(_md_row([
            f"{model}{nt}",
            f"{s['within_SD_DI']:.4f}",
            f"{s['within_SD_II']:.4f}",
            f"{s['within_SD_SI']:.4f}",
            f"{s['F']:.3f}",
            _p_str(s["p"]),
            f"{s['eta2']:.4f}",
            verd,
        ]))
    lines += [
        "",
        "_⚠ = qwen hidden states are 100% NaN-masked; CAA values unverified._",
        "",
    ]
    return lines


# ─── Comparison table ─────────────────────────────────────────────────────────
def comparison_md_table(model_data: dict) -> list[str]:
    hdr = ["model", "peak_L*",
           "eta2_mean_layer", "verdict_mean",
           "eta2_peak_layer", "verdict_peak",
           "delta_eta2", "interpretation"]
    lines = [
        "## Comparison: all-layer mean vs peak layer\n",
        "`delta_eta2 = eta2_peak_layer - eta2_mean_layer`  "
        "(negative = peak layer is *cleaner* for IB framing).\n",
        _md_row(hdr),
        _md_row(["---"] * len(hdr)),
    ]
    for model in MODELS_INSTRUCT:
        nt   = _nan_tag_md(model)
        peak_l = PEAK_LAYERS.get(model, "–")
        if model not in model_data:
            lines.append(_md_row([f"{model}{nt}", peak_l] + ["–"] * (len(hdr) - 2)))
            continue
        sm   = model_data[model]["mean"]
        sp   = model_data[model]["peak"]
        e2m  = sm["eta2"]
        e2p  = sp["eta2"]
        if np.isnan(e2p):
            lines.append(_md_row([
                f"{model}{nt}", peak_l,
                f"{e2m:.4f}", sm["verdict"] + nt,
                "–", "–", "–", "peak layer data unavailable",
            ]))
            continue
        delta = e2p - e2m
        # Interpretation
        if sp["verdict"] == "OK" and sm["verdict"] == "OK":
            interp = "Both OK"
        elif sp["verdict"] == "OK" and sm["verdict"] != "OK":
            interp = "Peak layer rescues IB framing"
        elif sp["verdict"] != "OK" and sm["verdict"] == "OK":
            interp = "Peak layer worse than mean"
        else:
            interp = f"Both {sp['verdict']}"
        lines.append(_md_row([
            f"{model}{nt}",
            peak_l,
            f"{e2m:.4f}", sm["verdict"] + nt,
            f"{e2p:.4f}", sp["verdict"] + nt,
            f"{delta:+.4f}",
            interp,
        ]))
    lines += [
        "",
        "_⚠ = qwen hidden states are 100% NaN-masked._",
        "_Peak layers from RDM notebook cell 22: argmax of partial-rho profile "
        "(last\\_token\\_hs, standardised, controlling for prompt\\_token\\_len)._",
        "",
    ]
    return lines


# ─── Scatter plot ──────────────────────────────────────────────────────────────
def make_scatter(df: pd.DataFrame, scores: pd.DataFrame, model: str) -> Path:
    merged = df.merge(scores, on="prompt_id", how="inner")
    merged = merged[merged["condition"] != "no_context"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for cond, color in COND_COLORS.items():
        sub = merged[merged["condition"] == cond]
        ax.scatter(
            sub["caa_mean_l2"], sub["AV"],
            c=color, label=CONDITION_ALIASES[cond],
            s=55, alpha=0.82, edgecolors="white", linewidths=0.4,
        )

    ax.set_xlabel("CAA mean L2 displacement (NC baseline)", fontsize=11)
    ax.set_ylabel("Rater utility (AV, higher = more relevant)", fontsize=11)

    nan_suffix = "\n[hidden states 100% NaN - values unverified]" \
                 if model == QWEN_NAN_WARN else ""
    ax.set_title(f"{model} - CAA displacement vs rater utility{nan_suffix}",
                 fontsize=11)

    ax.legend(title="Condition", fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    if model == QWEN_NAN_WARN:
        ax.text(0.02, 0.02, "hidden states NaN",
                transform=ax.transAxes, fontsize=8,
                color="firebrick", va="bottom")

    fig.tight_layout()
    out = OUT_DIR / f"effort_utility_scatter_{model}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ─── Main report ───────────────────────────────────────────────────────────────
def write_report() -> None:
    scores       = load_scores()
    # model_data[model] = {"mean": stats_dict, "peak": stats_dict, "df": df}
    model_data:  dict[str, dict] = {}
    missing:     list[str]       = []
    scatter_paths: list[Path]    = []

    for model in MODELS_INSTRUCT:
        df, path = load_metrics(model)
        if df is None:
            missing.append(model)
            continue
        mean_stats = anova_stats(df, "caa_mean_l2")
        # Peak-layer stats — NaN-safe if column not populated
        if "caa_at_peak_layer" in df.columns and df["caa_at_peak_layer"].notna().any():
            peak_stats = anova_stats(df, "caa_at_peak_layer")
        else:
            peak_stats = {k: np.nan for k in
                          ["within_SD_DI","within_SD_II","within_SD_SI","F","p","eta2","verdict"]}
        model_data[model] = {"mean": mean_stats, "peak": peak_stats, "df": df}

    # ── Build report ──────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []

    # Header
    lines += [
        "# CAA Displacement Variance Sanity Check",
        "",
        f"**Generated:** {ts}  ",
        f"**Working directory:** `{ROOT}`  ",
        "",
        "## Scope and data availability",
        "",
        "This report checks whether CAA L2 displacement from the NC baseline varies "
        "meaningfully *within* each non-NC condition (DI/II/SI), or whether "
        "between-condition differences dominate.  Two variants are tested: "
        "`caa_mean_l2` (mean across all layers) and `caa_at_peak_layer` (value at "
        "each model's empirically-derived peak layer from the RDM notebook).",
        "",
        "**Models analysed (instruct):** " +
        ", ".join(f"`{m}`" for m in MODELS_INSTRUCT),
        "",
        "**Base models (`llama_base`, `qwen_base`): absent from this analysis.**  ",
        "No `extended_metrics_*` file exists for either base model — the CAA "
        "computation pipeline was never run on them.  "
        "Extending this sanity check to base models requires running that pipeline "
        "first.  Defer that decision until the instruct results below have been "
        "evaluated.",
        "",
        "> **`qwen` warning:** the underlying `hidden_states_qwen.npz` is "
        "100% NaN-masked (confirmed in probe inventory).  All CAA values for "
        "qwen-instruct are derived from those NaN hidden states and are **unverified**.  "
        "Rows are included for completeness only.",
        "",
        "**Peak layers (from RDM notebook `rdm_analysis_colab_fixed_again.ipynb`, "
        "cell 22 — argmax of partial-rho profile, `last_token_hs`, standardised):**  ",
    ]
    for m, L in PEAK_LAYERS.items():
        lines.append(f"- `{m}`: layer {L}")
    lines += ["", "---", ""]

    # Per-model descriptive tables
    lines.append("## Per-condition descriptive statistics (`caa_mean_l2`)\n")
    for model, d in model_data.items():
        lines.append(descriptive_md(d["df"], model))
        lines.append("")

    lines += ["---", ""]

    # Table 1 — mean-layer ANOVA
    lines += anova_md_table(model_data, "mean",
                            "`caa_mean_l2` (mean L2 across all layers)")
    lines += ["---", ""]

    # Table 2 — peak-layer ANOVA
    lines += anova_md_table(model_data, "peak",
                            "`caa_at_peak_layer` (L2 at model-specific peak layer)")
    lines += ["---", ""]

    # Table 3 — comparison
    lines += comparison_md_table(model_data)
    lines += ["---", ""]

    # Overall assessment
    verified = [m for m in MODELS_INSTRUCT
                if m in model_data and m != QWEN_NAN_WARN]
    def _count_bad(col_key):
        return sum(1 for m in verified
                   if model_data[m][col_key]["verdict"] in ("MARGINAL", "FAIL"))

    n_bad_mean = _count_bad("mean")
    n_bad_peak = _count_bad("peak")

    lines += [
        "## Overall assessment",
        "",
        f"**Mean-layer (`caa_mean_l2`):** "
        f"{len(verified) - n_bad_mean}/{len(verified)} verified models OK.  "
        + (f"{n_bad_mean} MARGINAL/FAIL." if n_bad_mean else "No MARGINAL or FAIL."),
        "",
        f"**Peak-layer (`caa_at_peak_layer`):** "
        f"{len(verified) - n_bad_peak}/{len(verified)} verified models OK.  "
        + (f"{n_bad_peak} MARGINAL/FAIL." if n_bad_peak else "No MARGINAL or FAIL."),
        "",
    ]

    if n_bad_mean >= 3 or n_bad_peak >= 3:
        lines += [
            "> **WARNING:** Most verified models are MARGINAL or FAIL on at least one "
            "variant.  Between-condition separation is comparable to within-condition "
            "spread.  Treat these results as inconclusive pending layer-level decomposition.",
            "",
        ]

    # Scatter plots
    lines += ["---", "", "## Scatter plots", ""]
    for model in MODELS_INSTRUCT:
        if model in model_data:
            p = make_scatter(model_data[model]["df"], scores, model)
            scatter_paths.append(p)
            lines.append(f"- `{p.relative_to(ROOT)}`")
    lines += [""]

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    # ── Stdout ────────────────────────────────────────────────────────────────
    print(f"\n[OK] Report -> {REPORT}")
    print(f"[OK] Scatter plots -> {len(scatter_paths)} files\n")

    # Print both ANOVA tables and comparison to stdout
    col_w_anova = [22, 14, 14, 14, 9, 10, 12, 10]
    col_w_comp  = [22, 8, 16, 14, 16, 14, 12, 30]
    hdr_anova   = ["model", "wSD_DI", "wSD_II", "wSD_SI", "F", "p", "eta2", "verdict"]
    hdr_comp    = ["model", "peak_L*", "eta2_mean", "v_mean",
                   "eta2_peak", "v_peak", "delta_eta2", "interpretation"]

    def _row(cells, widths):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    for label, col_key in [("MEAN-LAYER", "mean"), ("PEAK-LAYER", "peak")]:
        print(f"--- {label} ANOVA ---")
        print(_row(hdr_anova, col_w_anova))
        print(_row(["-" * w for w in col_w_anova], col_w_anova))
        for model in MODELS_INSTRUCT:
            if model not in model_data:
                print(_row([f"{model} (no data)"] + ["–"] * 7, col_w_anova))
                continue
            s      = model_data[model][col_key]
            nt     = " [!]" if model == QWEN_NAN_WARN else ""
            e2     = s["eta2"]
            if np.isnan(e2):
                print(_row([f"{model}{nt}"] + ["–"] * 7, col_w_anova))
                continue
            print(_row([
                f"{model}{nt}",
                f"{s['within_SD_DI']:.4f}",
                f"{s['within_SD_II']:.4f}",
                f"{s['within_SD_SI']:.4f}",
                f"{s['F']:.3f}",
                _p_str(s["p"]),
                f"{e2:.4f}",
                s["verdict"] + nt,
            ], col_w_anova))
        print()

    print("--- COMPARISON ---")
    print(_row(hdr_comp, col_w_comp))
    print(_row(["-" * w for w in col_w_comp], col_w_comp))
    for model in MODELS_INSTRUCT:
        nt     = " [!]" if model == QWEN_NAN_WARN else ""
        peak_l = PEAK_LAYERS.get(model, "–")
        if model not in model_data:
            print(_row([f"{model}{nt}", peak_l] + ["–"] * 6, col_w_comp))
            continue
        sm, sp = model_data[model]["mean"], model_data[model]["peak"]
        e2m, e2p = sm["eta2"], sp["eta2"]
        if np.isnan(e2p):
            print(_row([f"{model}{nt}", peak_l, f"{e2m:.4f}", sm["verdict"] + nt,
                        "–", "–", "–", "peak unavailable"], col_w_comp))
            continue
        delta  = e2p - e2m
        if sp["verdict"] == "OK" and sm["verdict"] == "OK":
            interp = "Both OK"
        elif sp["verdict"] == "OK" and sm["verdict"] != "OK":
            interp = "Peak layer rescues IB framing"
        elif sp["verdict"] != "OK" and sm["verdict"] == "OK":
            interp = "Peak layer worse"
        else:
            interp = f"Both {sp['verdict']}"
        print(_row([
            f"{model}{nt}", peak_l,
            f"{e2m:.4f}", sm["verdict"] + nt,
            f"{e2p:.4f}", sp["verdict"] + nt,
            f"{delta:+.4f}", interp,
        ], col_w_comp))

    # Final flag
    flagged_mean = [(m, model_data[m]["mean"]["verdict"])
                    for m in MODELS_INSTRUCT
                    if m in model_data and model_data[m]["mean"]["verdict"] != "OK"]
    flagged_peak = [(m, model_data[m]["peak"]["verdict"])
                    for m in MODELS_INSTRUCT
                    if m in model_data
                    and not np.isnan(model_data[m]["peak"]["eta2"])
                    and model_data[m]["peak"]["verdict"] != "OK"]
    print()
    if flagged_mean:
        print("MARGINAL/FAIL (mean-layer):", ", ".join(f"{m}:{v}" for m, v in flagged_mean))
    else:
        print("Mean-layer: all verified models OK")
    if flagged_peak:
        print("MARGINAL/FAIL (peak-layer):", ", ".join(f"{m}:{v}" for m, v in flagged_peak))
    else:
        print("Peak-layer: all verified models OK")

    print(f"\nScatter plots:")
    for p in scatter_paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    write_report()
