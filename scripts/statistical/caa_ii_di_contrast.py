"""
caa_ii_di_contrast.py

Per-model, per-condition mean CAA (caa_mean_l2) plus a bootstrap CI on the
II minus DI contrast.  Reports to stdout and writes a markdown table.

Bootstrap: 10 000 resamples of (II - DI) sample means (resampling within each
condition independently), BCa percentile CI at 95%.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "results" / "probe_inventory"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT  = OUT_DIR / "caa_ii_di_contrast.md"

MODELS  = ["deepseek", "deepseek_v2_lite", "llama", "mistral", "qwen"]
QWEN_WARN   = "qwen"

CONDITIONS  = ["no_context", "direct_information",
               "implicature_information", "stochastic_information"]
ALIASES     = {"no_context": "NC", "direct_information": "DI",
               "implicature_information": "II", "stochastic_information": "SI"}

N_BOOT      = 10_000
RNG_SEED    = 42
CI_LEVEL    = 0.95

# ─── Data loading ─────────────────────────────────────────────────────────────
def _model_folder(model: str) -> str:
    return "deepseek" if model.startswith("deepseek") else model


def _find_csv(model: str) -> Path | None:
    p = ROOT / "data" / "model" / _model_folder(model) / f"extended_metrics_{model}.csv"
    return p if p.exists() else None


def load_model(model: str) -> pd.DataFrame | None:
    path = _find_csv(model)
    if path is None:
        return None
    df = pd.read_csv(path, low_memory=False)
    if "caa_mean_l2" not in df.columns or "condition" not in df.columns:
        return None
    return df[["condition", "caa_mean_l2"]].copy()

# ─── Bootstrap BCa ────────────────────────────────────────────────────────────
def _bca_ci(data_ii: np.ndarray, data_di: np.ndarray,
            n_boot: int = N_BOOT, seed: int = RNG_SEED,
            level: float = CI_LEVEL) -> tuple[float, float]:
    """
    BCa bootstrap CI for contrast = mean(II) - mean(DI).
    Resamples each condition independently (paired-design not assumed).
    """
    rng      = np.random.default_rng(seed)
    n_ii     = len(data_ii)
    n_di     = len(data_di)
    obs_stat = data_ii.mean() - data_di.mean()

    # Bootstrap distribution
    boot = np.empty(n_boot)
    for i in range(n_boot):
        s_ii = rng.choice(data_ii, size=n_ii, replace=True)
        s_di = rng.choice(data_di, size=n_di, replace=True)
        boot[i] = s_ii.mean() - s_di.mean()

    # Bias-correction z0
    z0 = float(np.percentile(np.sum(boot < obs_stat) / n_boot, 50,
                             method="linear") if False else
               _probit(np.mean(boot < obs_stat)))

    # Acceleration a (jackknife)
    combined = np.concatenate([data_ii, data_di])
    n_total  = len(combined)
    jack     = np.empty(n_total)
    for k in range(n_total):
        left     = np.delete(combined, k)
        ii_mask  = np.arange(n_ii)      # first n_ii are II
        jk_ii    = left[:n_ii - (1 if k < n_ii else 0)]
        jk_di    = left[len(jk_ii):]
        # simpler: jackknife on combined means
        jack[k]  = combined[np.arange(n_total) != k].mean()

    # Use standard acceleration estimate on the contrast statistic
    jk_vals = np.empty(n_total)
    for k in range(n_total):
        mask     = np.arange(n_total) != k
        jk_ii_   = data_ii[data_ii != combined[k]] if k < n_ii else data_ii
        jk_di_   = data_di[data_di != combined[k]] if k >= n_ii else data_di
        # simple jackknife: leave out obs k from its own group
        if k < n_ii:
            jk_ii_ = np.delete(data_ii, k)
            jk_di_ = data_di
        else:
            jk_ii_ = data_ii
            jk_di_ = np.delete(data_di, k - n_ii)
        jk_vals[k] = jk_ii_.mean() - jk_di_.mean()

    jk_mean = jk_vals.mean()
    num      = np.sum((jk_mean - jk_vals) ** 3)
    den      = 6.0 * (np.sum((jk_mean - jk_vals) ** 2) ** 1.5)
    a        = num / den if den != 0 else 0.0

    alpha   = (1 - level) / 2
    z_alpha = _probit(alpha)
    z_1ma   = _probit(1 - alpha)

    a1 = _norm_cdf(z0 + (z0 + z_alpha)  / (1 - a * (z0 + z_alpha)))
    a2 = _norm_cdf(z0 + (z0 + z_1ma)    / (1 - a * (z0 + z_1ma)))

    lo = float(np.percentile(boot, 100 * a1))
    hi = float(np.percentile(boot, 100 * a2))
    return lo, hi


def _probit(p: float) -> float:
    from scipy.special import ndtri
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return float(ndtri(p))


def _norm_cdf(z: float) -> float:
    from scipy.special import ndtr
    return float(ndtr(z))

# ─── Report ───────────────────────────────────────────────────────────────────
def run() -> None:
    rows: list[dict] = []

    for model in MODELS:
        df = load_model(model)
        warn = " [!]" if model == QWEN_WARN else ""

        if df is None:
            print(f"[SKIP] {model} — no extended_metrics CSV found")
            continue

        # Per-condition descriptive stats
        cond_stats: dict[str, dict] = {}
        for cond in CONDITIONS:
            vals = df.loc[df["condition"] == cond, "caa_mean_l2"].dropna().values
            cond_stats[ALIASES[cond]] = {
                "n":    len(vals),
                "mean": float(vals.mean()) if len(vals) else float("nan"),
                "sd":   float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
            }

        # II - DI contrast + bootstrap BCa CI
        ii_vals = df.loc[df["condition"] == "implicature_information",
                         "caa_mean_l2"].dropna().values
        di_vals = df.loc[df["condition"] == "direct_information",
                         "caa_mean_l2"].dropna().values

        contrast = float(ii_vals.mean() - di_vals.mean()) if (
            len(ii_vals) and len(di_vals)) else float("nan")

        ci_lo, ci_hi = _bca_ci(ii_vals, di_vals) if (
            len(ii_vals) >= 2 and len(di_vals) >= 2) else (float("nan"), float("nan"))

        includes_zero = (ci_lo <= 0 <= ci_hi)

        rows.append({
            "model":       model + warn,
            "NC_mean":     cond_stats["NC"]["mean"],
            "DI_mean":     cond_stats["DI"]["mean"],
            "DI_sd":       cond_stats["DI"]["sd"],
            "II_mean":     cond_stats["II"]["mean"],
            "II_sd":       cond_stats["II"]["sd"],
            "SI_mean":     cond_stats["SI"]["mean"],
            "SI_sd":       cond_stats["SI"]["sd"],
            "II_DI":       contrast,
            "CI_lo":       ci_lo,
            "CI_hi":       ci_hi,
            "zero_in_CI":  includes_zero,
        })

    # ── Stdout ────────────────────────────────────────────────────────────────
    w = 110
    print("-" * w)
    print(f"{'model':<24} {'NC':>8} {'DI mean':>10} {'DI sd':>8} "
          f"{'II mean':>10} {'II sd':>8} {'SI mean':>10} {'SI sd':>8} "
          f"{'II-DI':>9} {'95% CI':>20} {'0 in CI':>8}")
    print("-" * w)
    for r in rows:
        ci_str = f"[{r['CI_lo']:+.3f}, {r['CI_hi']:+.3f}]"
        print(
            f"{r['model']:<24} {r['NC_mean']:>8.3f} {r['DI_mean']:>10.3f} "
            f"{r['DI_sd']:>8.3f} {r['II_mean']:>10.3f} {r['II_sd']:>8.3f} "
            f"{r['SI_mean']:>10.3f} {r['SI_sd']:>8.3f} "
            f"{r['II_DI']:>+9.3f} {ci_str:>20} {'yes' if r['zero_in_CI'] else 'no':>8}"
        )
    print("-" * w)

    # ── Markdown ──────────────────────────────────────────────────────────────
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    md  = [
        "# CAA II-DI Contrast Report",
        "",
        f"**Generated:** {now}  ",
        f"**Metric:** `caa_mean_l2` (mean L2 displacement from NC baseline, all layers)  ",
        f"**Bootstrap:** {N_BOOT:,} BCa resamples, 95 % CI, seed={RNG_SEED}  ",
        "",
        "> ⚠ `qwen [!]`: hidden states 100 % NaN-masked — CAA values unverified.",
        "",
        "---",
        "",
        "## Per-condition descriptive statistics",
        "",
        "| model | n | NC mean | DI mean (SD) | II mean (SD) | SI mean (SD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        md.append(
            f"| {r['model']} | 15 "
            f"| {r['NC_mean']:.3f} "
            f"| {r['DI_mean']:.3f} ({r['DI_sd']:.3f}) "
            f"| {r['II_mean']:.3f} ({r['II_sd']:.3f}) "
            f"| {r['SI_mean']:.3f} ({r['SI_sd']:.3f}) |"
        )

    md += [
        "",
        "---",
        "",
        "## II − DI contrast with bootstrap BCa 95 % CI",
        "",
        "Positive contrast = II elicits more representational displacement than DI.  ",
        "CI excludes zero → contrast is reliably non-zero at 95 % level.",
        "",
        "| model | II mean | DI mean | II − DI | 95 % BCa CI | 0 in CI |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        ci_str = f"[{r['CI_lo']:+.3f}, {r['CI_hi']:+.3f}]"
        zero   = "yes" if r["zero_in_CI"] else "**no**"
        md.append(
            f"| {r['model']} | {r['II_mean']:.3f} | {r['DI_mean']:.3f} "
            f"| {r['II_DI']:+.3f} | {ci_str} | {zero} |"
        )

    md += ["", "---", ""]

    REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[OK] Report -> {REPORT}")


if __name__ == "__main__":
    run()
