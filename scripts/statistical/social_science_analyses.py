"""
Social Science Statistical Analyses
=====================================
Five analyses translating the IB/MI findings into standard statistical formats:
  1. Repeated-Measures ANOVA   — condition -> output entropy (per model)
  2. Moderated Regression      — architecture moderates which accuracy signal drives entropy
  3. Fisher z-tests            — GQA vs MLA correlation profiles
  4. One-way ANOVA             — communicative efficiency across conditions
  5. Mixed ANOVA               — condition x response phase interaction on delta-EAS

Multiple comparison corrections:
  Within-analysis:  Bonferroni on post-hoc pairwise comparisons
  Across analyses:  Benjamini-Hochberg FDR on primary test statistics
"""

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols
import pingouin as pg
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "results" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALPHA    = 0.05
RANDOM_SEED = 42

MODELS          = ["mistral", "qwen", "llama", "deepseek_v2_lite", "deepseek"]
MODELS_WITH_EXT = ["qwen", "deepseek_v2_lite"]

ARCH = {
    "mistral":          {"MLA": 0, "MTP": 0, "k": 1, "label": "Mistral\n(k=1, GQA)"},
    "qwen":             {"MLA": 0, "MTP": 0, "k": 1, "label": "Qwen\n(k=1, GQA)"},
    "llama":            {"MLA": 0, "MTP": 0, "k": 1, "label": "LLaMA\n(k=1, GQA)"},
    "deepseek_v2_lite": {"MLA": 1, "MTP": 0, "k": 1, "label": "DS-V2-Lite\n(k=1, MLA)"},
    "deepseek":         {"MLA": 1, "MTP": 1, "k": 2, "label": "DeepSeek-V3\n(k=2, MLA)"},
}
MODEL_COLORS = {"mistral": "#F57C00", "qwen": "#1976D2", "llama": "#7B1FA2",
                "deepseek_v2_lite": "#00897B", "deepseek": "#388E3C"}
COND_ORDER  = ["direct_information", "implicature_information", "stochastic_information"]
COND_NC     = ["no_context"] + COND_ORDER
COND_LABELS = {"no_context":"NC","direct_information":"DI",
               "implicature_information":"II","stochastic_information":"SI"}
COND_COLORS = {"no_context":"#9E9E9E","direct_information":"#2196F3",
               "implicature_information":"#FF9800","stochastic_information":"#E91E63"}


# ── Helpers ───────────────────────────────────────────────────────
def cohen_d(a, b):
    a, b = np.array(a), np.array(b)
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / sd if sd > 0 else 0.0

def fdr_correct(pvals):
    pvals = list(pvals)
    reject, p_adj, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")
    return list(reject), list(p_adj)

def bonferroni(p, k): return min(float(p) * k, 1.0)

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

def fisher_z_test(r1, r2, n1, n2):
    z1, z2 = np.arctanh(np.clip(r1,-0.9999,0.9999)), np.arctanh(np.clip(r2,-0.9999,0.9999))
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    z  = (z1 - z2) / se
    p  = 2*(1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def _model_folder(model: str) -> str:
    return "deepseek" if model.startswith("deepseek") else model


# ── Data loading ──────────────────────────────────────────────────
def load_all_data():
    rater = (pd.read_csv(DATA_DIR / "human" / "QuestionContext_Scores.csv")
             [["Num","AV"]].dropna()
             .rename(columns={"Num":"prompt_id","AV":"context_score"}))
    rater["prompt_id"] = rater["prompt_id"].astype(int)

    frames = []
    for model in MODELS:
        df = pd.read_csv(DATA_DIR / "model" / _model_folder(model) / f"pilot_summary_{model}.csv")
        df["model"] = model
        df["question_id"] = (df["prompt_id"] - 1) // 4 + 1
        df = df.merge(rater, on="prompt_id", how="left")
        df["acc_rater"] = 6 - df["context_score"]
        df["acc_sim"]   = df["context_question_similarity"]
        df["MLA"] = ARCH[model]["MLA"]
        df["MTP"] = ARCH[model]["MTP"]
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    nc   = (full[full["condition"]=="no_context"]
            [["model","question_id","mean_token_entropy"]]
            .rename(columns={"mean_token_entropy":"nc_entropy"}))
    full = full.merge(nc, on=["model","question_id"], how="left")
    full["delta_entropy"]   = full["mean_token_entropy"] - full["nc_entropy"]
    full["eff_beam_proxy"]  = -full["delta_entropy"] / full["acc_rater"].replace(0,np.nan)
    return full


def load_phase_data():
    frames = []
    for model in MODELS_WITH_EXT:
        ext = pd.read_csv(DATA_DIR / "model" / _model_folder(model) / f"extended_metrics_{model}.csv")
        ext["model"]       = model
        ext["question_id"] = (ext["prompt_id"] - 1) // 4 + 1
        nc_ph = (ext[ext["condition"]=="no_context"]
                 [["question_id","eas_early","eas_mid","eas_late"]]
                 .rename(columns={"eas_early":"nc_e","eas_mid":"nc_m","eas_late":"nc_l"}))
        sub = ext[ext["condition"]!="no_context"].merge(nc_ph, on="question_id", how="left")
        sub["delta_early"] = sub["eas_early"] - sub["nc_e"]
        sub["delta_mid"]   = sub["eas_mid"]   - sub["nc_m"]
        sub["delta_late"]  = sub["eas_late"]  - sub["nc_l"]
        frames.append(sub[["model","prompt_id","question_id","condition",
                            "delta_early","delta_mid","delta_late"]])
    return pd.concat(frames, ignore_index=True)


# ── Analysis 1: RM-ANOVA ──────────────────────────────────────────
def analysis_1(full_df):
    print("\n"+"="*70)
    print("ANALYSIS 1: Repeated-Measures ANOVA")
    print("DV: Mean Token Entropy  |  IV: Condition (NC/DI/II/SI)")
    print("="*70)
    results, primary_p = {}, []

    for model in MODELS:
        df_m = (full_df[full_df["model"]==model]
                [full_df["condition"].isin(COND_NC)]
                [["question_id","condition","mean_token_entropy"]]
                .dropna()
                .rename(columns={"mean_token_entropy":"entropy","question_id":"subject"}))
        counts = df_m.groupby("subject")["condition"].count()
        df_m   = df_m[df_m["subject"].isin(counts[counts==4].index)]

        aov     = pg.rm_anova(data=df_m, dv="entropy", within="condition",
                               subject="subject", detailed=True)
        crow    = aov[aov["Source"]=="condition"].iloc[0]
        p_main  = float(crow["p-unc"])
        f_val   = float(crow["F"])
        eta2    = float(crow["ng2"])
        df1, df2 = int(crow["DF"]), int(crow["DF"])

        posthoc = pg.pairwise_tests(data=df_m, dv="entropy", within="condition",
                                     subject="subject", padjust="bonf")
        means   = df_m.groupby("condition")["entropy"].agg(["mean","sem"]).reindex(COND_NC)
        primary_p.append(p_main)
        results[model] = dict(F=f_val, p=p_main, eta2=eta2, df=(df1,df2),
                               means=means, posthoc=posthoc, n=len(df_m["subject"].unique()))
        print(f"\n  {model.upper():<20}  F({df1},{df2})={f_val:.3f},  "
              f"p={p_main:.4f}{sig_stars(p_main)},  eta2={eta2:.3f},  N={results[model]['n']}")
        for _, row in posthoc[posthoc["p-corr"]<ALPHA].iterrows():
            print(f"    {row['A']:<30} vs {row['B']:<30}  t={row['T']:+.3f}  p_bonf={row['p-corr']:.4f}{sig_stars(row['p-corr'])}")

    reject, p_adj = fdr_correct(primary_p)
    print("\n  BH-FDR correction across 5 models:")
    for i, model in enumerate(MODELS):
        results[model]["p_fdr"] = p_adj[i]
        results[model]["reject_fdr"] = reject[i]
        print(f"    {model:<22}  p_fdr={p_adj[i]:.4f}  {'REJECT' if reject[i] else 'retain'}")
    return results


def plot_1(results, ax1, ax2):
    x = np.arange(len(MODELS)); w = 0.18
    offs = np.linspace(-(len(COND_NC)-1)*w/2, (len(COND_NC)-1)*w/2, len(COND_NC))
    for ci, cond in enumerate(COND_NC):
        ms = [results[m]["means"].loc[cond,"mean"] if cond in results[m]["means"].index else np.nan for m in MODELS]
        ss = [results[m]["means"].loc[cond,"sem"]  if cond in results[m]["means"].index else 0       for m in MODELS]
        ax1.bar(x+offs[ci], ms, w, color=COND_COLORS[cond], label=COND_LABELS[cond], alpha=0.85, zorder=3)
        ax1.errorbar(x+offs[ci], ms, yerr=ss, fmt="none", color="black", capsize=2, lw=1, zorder=4)
    y_top = max(results[m]["means"]["mean"].max() for m in MODELS)+0.06
    for i, model in enumerate(MODELS):
        s = sig_stars(results[model]["p_fdr"])
        col = "#C62828" if results[model]["reject_fdr"] else "#9E9E9E"
        ax1.text(x[i], y_top, s, ha="center", fontsize=11, fontweight="bold", color=col)
    ax1.set_xticks(x); ax1.set_xticklabels([ARCH[m]["label"] for m in MODELS], fontsize=8.5)
    ax1.set_ylabel("Mean Token Entropy (nats)", fontsize=9)
    ax1.set_title("Analysis 1: RM-ANOVA — Token Entropy by Condition & Model\n"
                  "±1 SEM  |  Stars = FDR-corrected main effect (red=significant)",
                  fontsize=9.5, fontweight="bold")
    ax1.legend(title="Condition", fontsize=8, ncol=4, loc="upper right", framealpha=0.7)
    ax1.spines[["top","right"]].set_visible(False)

    eta2s = [results[m]["eta2"] for m in MODELS]
    cols  = ["#C62828" if results[m]["reject_fdr"] else "#9E9E9E" for m in MODELS]
    ax2.bar(range(len(MODELS)), eta2s, color=cols, alpha=0.85, zorder=3)
    for i,(v,s) in enumerate(zip(eta2s,[sig_stars(results[m]["p_fdr"]) for m in MODELS])):
        ax2.text(i, v+0.003, f"{v:.3f}\n{s}", ha="center", fontsize=8.5, fontweight="bold")
    ax2.axhline(0.06, color="#757575", lw=1.2, ls="--", alpha=0.7, label="Medium η²=.06")
    ax2.axhline(0.14, color="#424242", lw=1.2, ls=":", alpha=0.7,  label="Large η²=.14")
    ax2.set_xticks(range(len(MODELS))); ax2.set_xticklabels([ARCH[m]["label"] for m in MODELS], fontsize=8.5)
    ax2.set_ylabel("Generalised η² (Effect Size)", fontsize=9)
    ax2.set_title("Effect Size by Model\nRed = FDR-significant", fontsize=9.5, fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.7); ax2.spines[["top","right"]].set_visible(False)


# ── Analysis 2: Moderated Regression ─────────────────────────────
def analysis_2(full_df):
    print("\n"+"="*70)
    print("ANALYSIS 2: Moderated Multiple Regression")
    print("DV: delta_entropy  |  IVs: acc_rater(z), acc_sim(z), MLA, MTP + interactions")
    print("="*70)
    df_r = (full_df[full_df["condition"].isin(COND_ORDER)]
            [["delta_entropy","acc_rater","acc_sim","MLA","MTP","model"]].dropna().copy())
    for col in ["acc_rater","acc_sim"]:
        df_r[f"{col}_z"] = (df_r[col]-df_r[col].mean())/df_r[col].std()
    df_r["rater_x_MLA"] = df_r["acc_rater_z"]*df_r["MLA"]
    df_r["sim_x_MLA"]   = df_r["acc_sim_z"]*df_r["MLA"]
    df_r["rater_x_MTP"] = df_r["acc_rater_z"]*df_r["MTP"]

    fit = ols("delta_entropy ~ acc_rater_z + acc_sim_z + MLA + MTP + "
              "rater_x_MLA + sim_x_MLA + rater_x_MTP", data=df_r).fit()
    tbl = fit.summary2().tables[1]
    preds = [n for n in tbl.index if n != "Intercept"]
    raw_p = [float(tbl.loc[n,"P>|t|"]) for n in preds]
    reject, p_adj = fdr_correct(raw_p)

    coef_res = {}
    print(f"\n  {'Predictor':<22} {'beta':>8} {'SE':>7} {'t':>7} {'p_raw':>8} {'p_fdr':>8}  sig")
    print(f"  {'-'*68}")
    for name, p_r, p_f, rej in zip(preds, raw_p, p_adj, reject):
        b  = float(tbl.loc[name,"Coef."]); se = float(tbl.loc[name,"Std.Err."]); t = float(tbl.loc[name,"t"])
        print(f"  {name:<22} {b:>+8.4f} {se:>7.4f} {t:>+7.3f} {p_r:>8.4f} {p_f:>8.4f}  {'*' if rej else ' '}")
        coef_res[name] = dict(beta=b, se=se, t=t, p_raw=p_r, p_fdr=p_f, reject=rej)
    print(f"\n  R²={fit.rsquared:.4f}  Adj-R²={fit.rsquared_adj:.4f}  "
          f"F({int(fit.df_model)},{int(fit.df_resid)})={fit.fvalue:.3f}  p={fit.f_pvalue:.4f}{sig_stars(fit.f_pvalue)}")
    for mla_val, label in [(0,"GQA (MLA=0)"),(1,"MLA (MLA=1)")]:
        sub = df_r[df_r["MLA"]==mla_val].dropna(subset=["acc_rater_z","delta_entropy"])
        r,p = pearsonr(sub["acc_rater_z"], sub["delta_entropy"])
        print(f"  Simple slope {label}: r={r:+.3f}, p={p:.4f}{sig_stars(p)}, n={len(sub)}")
    return fit, coef_res, df_r


def plot_2(fit, coef_res, df_r, ax1, ax2):
    names  = list(coef_res.keys())
    betas  = [coef_res[n]["beta"] for n in names]
    ses    = [coef_res[n]["se"]   for n in names]
    cols   = ["#C62828" if coef_res[n]["reject"] else "#9E9E9E" for n in names]
    ax1.barh(range(len(names)), betas, xerr=[1.96*s for s in ses], color=cols,
              alpha=0.85, height=0.6, error_kw={"capsize":4,"elinewidth":1.5,"ecolor":"black"})
    ax1.axvline(0, color="black", lw=1, ls="--")
    pretty = {"acc_rater_z":"Pragmatic Relevance (z)","acc_sim_z":"Lexical Similarity (z)",
               "MLA":"Architecture: MLA (0/1)","MTP":"Multi-token Prediction (0/1)",
               "rater_x_MLA":"Relevance × MLA  ◀ key","sim_x_MLA":"Similarity × MLA",
               "rater_x_MTP":"Relevance × MTP"}
    ax1.set_yticks(range(len(names))); ax1.set_yticklabels([pretty.get(n,n) for n in names], fontsize=8.5)
    ax1.set_xlabel("Standardised β  (±95% CI)", fontsize=8.5)
    ax1.set_title("Analysis 2: Moderated Regression Coefficients\nRed = FDR-significant",
                  fontsize=9.5, fontweight="bold")
    ax1.text(0.97,0.05, f"R²={fit.rsquared:.3f}\nAdj-R²={fit.rsquared_adj:.3f}",
              transform=ax1.transAxes, ha="right", va="bottom", fontsize=8,
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax1.spines[["top","right"]].set_visible(False)

    for mla_val, color, label in [(0,"#1976D2","GQA (MLA=0)"),(1,"#388E3C","MLA (MLA=1)")]:
        sub = df_r[df_r["MLA"]==mla_val].dropna(subset=["acc_rater_z","delta_entropy"])
        ax2.scatter(sub["acc_rater_z"], sub["delta_entropy"], color=color, alpha=0.20, s=18, zorder=2)
        m,b = np.polyfit(sub["acc_rater_z"], sub["delta_entropy"], 1)
        xr  = np.linspace(sub["acc_rater_z"].min(), sub["acc_rater_z"].max(), 100)
        r,p = pearsonr(sub["acc_rater_z"], sub["delta_entropy"])
        ax2.plot(xr, m*xr+b, color=color, lw=2.5, label=f"{label}\n(r={r:+.3f}, {sig_stars(p)})")
    ax2.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax2.set_xlabel("Pragmatic Relevance — acc_rater (z-scored)", fontsize=8.5)
    ax2.set_ylabel("Δ Token Entropy vs NC Baseline (nats)", fontsize=8.5)
    ax2.set_title("Simple Slopes: Pragmatic Relevance × Architecture\nDiverging slopes = architecture moderates pragmatic tracking",
                  fontsize=9.5, fontweight="bold")
    ax2.legend(fontsize=8.5, framealpha=0.7); ax2.spines[["top","right"]].set_visible(False)


# ── Analysis 3: Fisher z-Tests ────────────────────────────────────
def analysis_3(full_df):
    print("\n"+"="*70)
    print("ANALYSIS 3: Fisher z-Tests on Correlation Profiles")
    print("="*70)
    df_c = (full_df[full_df["condition"].isin(COND_ORDER)]
            [["model","delta_entropy","acc_rater","acc_sim"]].dropna())
    corr_res = {}
    for model in MODELS:
        sub = df_c[df_c["model"]==model]
        r_r,p_r = pearsonr(sub["acc_rater"], sub["delta_entropy"])
        r_s,p_s = pearsonr(sub["acc_sim"],   sub["delta_entropy"])
        z_w,p_w = fisher_z_test(r_r, r_s, len(sub), len(sub))
        corr_res[model] = dict(r_rater=r_r,p_rater=p_r,r_sim=r_s,p_sim=p_s,n=len(sub),z_within=z_w,p_within=p_w)

    bonf_ps = [bonferroni(corr_res[m]["p_within"], len(MODELS)) for m in MODELS]
    print(f"\n  Within-model: r_rater vs r_sim — Bonferroni (k=5)")
    print(f"  {'Model':<22} {'r_rater':>8} {'r_sim':>8} {'Diff':>6} {'z':>7} {'p_bonf':>8}")
    print(f"  {'-'*65}")
    for model, p_b in zip(MODELS, bonf_ps):
        cr = corr_res[model]
        print(f"  {model:<22} {cr['r_rater']:>+8.3f} {cr['r_sim']:>+8.3f} "
              f"{cr['r_rater']-cr['r_sim']:>+6.3f} {cr['z_within']:>+7.3f} {p_b:>8.4f}{sig_stars(p_b)}")
        corr_res[model]["p_bonf"] = p_b

    gqa = [m for m in MODELS if ARCH[m]["MLA"]==0]
    mla = [m for m in MODELS if ARCH[m]["MLA"]==1]
    r_gqa = np.mean([corr_res[m]["r_rater"] for m in gqa])
    r_mla = np.mean([corr_res[m]["r_rater"] for m in mla])
    n_gqa = sum(corr_res[m]["n"] for m in gqa)
    n_mla = sum(corr_res[m]["n"] for m in mla)
    z_b, p_b = fisher_z_test(r_mla, r_gqa, n_mla, n_gqa)
    print(f"\n  Between-arch: GQA r={r_gqa:+.3f}, MLA r={r_mla:+.3f}, "
          f"Fisher z={z_b:+.3f}, p={p_b:.4f}{sig_stars(p_b)}")
    return corr_res, dict(z=z_b, p=p_b, r_gqa=r_gqa, r_mla=r_mla)


def plot_3(corr_res, between, ax1, ax2):
    x = np.arange(len(MODELS)); w = 0.3
    r_r = [corr_res[m]["r_rater"] for m in MODELS]
    r_s = [corr_res[m]["r_sim"]   for m in MODELS]
    ax1.bar(x-w/2, r_r, w, label="r(Δentropy, Pragmatic Relevance)", color="#FF7043", alpha=0.85, zorder=3)
    ax1.bar(x+w/2, r_s, w, label="r(Δentropy, Lexical Similarity)",  color="#42A5F5", alpha=0.85, zorder=3)
    for i, model in enumerate(MODELS):
        s = sig_stars(corr_res[model]["p_bonf"])
        if s != "n.s.":
            ax1.text(x[i], max(r_r[i],r_s[i])+0.02, s, ha="center", fontsize=11, fontweight="bold")
    ax1.axhline(0, color="black", lw=0.8, ls="--")
    ax1.set_xticks(x); ax1.set_xticklabels([ARCH[m]["label"] for m in MODELS], fontsize=8.5)
    ax1.set_ylabel("Pearson r with Δ Token Entropy", fontsize=9)
    ax1.set_title("Analysis 3: Correlation Profiles — Pragmatic vs Lexical Signal\n"
                  "Stars = Bonferroni-corrected Fisher z (rater r ≠ sim r)",
                  fontsize=9.5, fontweight="bold")
    ax1.legend(fontsize=8.5, framealpha=0.7, loc="lower right"); ax1.spines[["top","right"]].set_visible(False)

    arch_labels = ["GQA Models\n(Mistral, Qwen, LLaMA)", "MLA Models\n(DS-V2-Lite, DS-V3)"]
    arch_vals   = [between["r_gqa"], between["r_mla"]]
    bars = ax2.bar(arch_labels, arch_vals, color=["#1976D2","#388E3C"], alpha=0.85, width=0.4, zorder=3)
    for bar, val in zip(bars, arch_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, val+(0.005 if val>=0 else -0.015),
                  f"r = {val:+.3f}", ha="center", fontsize=9.5, fontweight="bold")
    y_sig = max(arch_vals)+0.04
    ax2.plot([0,1],[y_sig,y_sig], color="black", lw=1.5)
    ax2.text(0.5, y_sig+0.008, f"Fisher z={between['z']:+.2f}\n{sig_stars(between['p'])}",
              ha="center", fontsize=9, fontweight="bold")
    ax2.axhline(0, color="black", lw=0.8, ls="--")
    ax2.set_ylabel("Mean r(Δentropy, Pragmatic Relevance)", fontsize=9)
    ax2.set_title("Between-Architecture: MLA vs GQA\nFisher z-test on r_rater",
                  fontsize=9.5, fontweight="bold")
    ax2.spines[["top","right"]].set_visible(False)


# ── Analysis 4: Efficiency ANOVA ─────────────────────────────────
def analysis_4(full_df):
    print("\n"+"="*70)
    print("ANALYSIS 4: One-Way ANOVA — Communicative Efficiency")
    print("DV: epsilon = -delta_entropy / acc_rater  |  IV: Condition (DI/II/SI)")
    print("="*70)
    results, primary_p = {}, []
    for model in MODELS:
        df_m = (full_df[(full_df["model"]==model) & (full_df["condition"].isin(COND_ORDER))]
                [["condition","eff_beam_proxy"]].dropna())
        groups = [df_m[df_m["condition"]==c]["eff_beam_proxy"].values for c in COND_ORDER]
        groups = [g for g in groups if len(g)>=3]
        if len(groups)<2: primary_p.append(1.0); results[model]={"F":np.nan,"p":1.0,"eta2":0.0,"posthoc":[]}; continue
        F,p = f_oneway(*groups)
        gm = df_m["eff_beam_proxy"].mean()
        ss_b = sum(len(g)*(g.mean()-gm)**2 for g in groups)
        ss_t = sum(((g-gm)**2).sum() for g in groups)
        eta2 = ss_b/ss_t if ss_t>0 else 0.0
        pairs_def = [("direct_information","implicature_information"),
                     ("direct_information","stochastic_information"),
                     ("implicature_information","stochastic_information")]
        ph = []
        for c1,c2 in pairs_def:
            g1 = df_m[df_m["condition"]==c1]["eff_beam_proxy"].values
            g2 = df_m[df_m["condition"]==c2]["eff_beam_proxy"].values
            t_v,p_r = ttest_ind(g1,g2)
            ph.append({"pair":f"{COND_LABELS[c1]}-{COND_LABELS[c2]}","t":t_v,"p_bonf":bonferroni(p_r,3),"d":cohen_d(g1,g2)})
        primary_p.append(p)
        means = df_m.groupby("condition")["eff_beam_proxy"].agg(["mean","sem"]).reindex(COND_ORDER)
        results[model] = dict(F=F, p=p, eta2=eta2, posthoc=ph, means=means)
        n_t = sum(len(g) for g in groups)
        print(f"\n  {model.upper():<20}  F(2,{n_t-3})={F:.3f},  p={p:.4f}{sig_stars(p)},  eta2={eta2:.3f}")
        for row in ph:
            print(f"    {row['pair']:<10}  t={row['t']:+.3f}  p_bonf={row['p_bonf']:.4f}{sig_stars(row['p_bonf'])}  d={row['d']:+.3f}")

    reject, p_adj = fdr_correct(primary_p)
    print("\n  BH-FDR correction across 5 models:")
    for i,model in enumerate(MODELS):
        results[model]["p_fdr"] = p_adj[i]; results[model]["reject_fdr"] = reject[i]
        print(f"    {model:<22}  p_fdr={p_adj[i]:.4f}  {'REJECT' if reject[i] else 'retain'}")
    return results


def plot_4(results, ax1, ax2):
    x = np.arange(len(MODELS)); w = 0.22
    offs = np.linspace(-(len(COND_ORDER)-1)*w/2,(len(COND_ORDER)-1)*w/2,len(COND_ORDER))
    cond_cols_3 = ["#2196F3","#FF9800","#E91E63"]
    for ci,(cond,col) in enumerate(zip(COND_ORDER,cond_cols_3)):
        ms=[]; ss=[]
        for model in MODELS:
            m_df = results[model].get("means")
            if m_df is not None and cond in m_df.index: ms.append(m_df.loc[cond,"mean"]); ss.append(m_df.loc[cond,"sem"])
            else: ms.append(np.nan); ss.append(0)
        ax1.bar(x+offs[ci], ms, w, color=col, label=COND_LABELS[cond], alpha=0.85, zorder=3)
        ax1.errorbar(x+offs[ci], ms, yerr=ss, fmt="none", color="black", capsize=2, lw=1, zorder=4)
    ax1.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels([ARCH[m]["label"] for m in MODELS], fontsize=8.5)
    ax1.set_ylabel("Communicative Efficiency ε\n(−Δentropy / acc_rater)", fontsize=9)
    ax1.set_title("Analysis 4: ANOVA — Communicative Efficiency by Condition\nHigher ε = more entropy reduction per context token",
                  fontsize=9.5, fontweight="bold")
    ax1.legend(title="Condition", fontsize=8, framealpha=0.7); ax1.spines[["top","right"]].set_visible(False)

    eta2s = [results[m]["eta2"] for m in MODELS]
    fcols = ["#C62828" if results[m]["reject_fdr"] else "#9E9E9E" for m in MODELS]
    for i,(v,c) in enumerate(zip(eta2s,fcols)):
        ax2.vlines(i,0,v,color=c,lw=2.5,alpha=0.85); ax2.scatter(i,v,color=c,s=100,zorder=4)
        ax2.text(i,v+0.005,f"{v:.3f}",ha="center",fontsize=8.5,fontweight="bold")
    ax2.axhline(0.01, color="#9E9E9E",lw=1,ls=":",label="Small η²=.01")
    ax2.axhline(0.06, color="#616161",lw=1,ls="--",label="Medium η²=.06")
    ax2.set_xticks(range(len(MODELS))); ax2.set_xticklabels([ARCH[m]["label"] for m in MODELS], fontsize=8.5)
    ax2.set_ylabel("η² (Proportion of Variance Explained)", fontsize=9)
    ax2.set_title("Effect Size (η²): Condition → Efficiency\nRed = FDR-significant",
                  fontsize=9.5, fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.7); ax2.spines[["top","right"]].set_visible(False)


# ── Analysis 5: Mixed ANOVA ───────────────────────────────────────
def analysis_5(phase_df):
    print("\n"+"="*70)
    print("ANALYSIS 5: Mixed ANOVA — Condition x Response Phase")
    print("Between: Condition (DI vs SI)  |  Within: Phase (Early/Mid/Late)")
    print(f"Models: {MODELS_WITH_EXT}")
    print("="*70)
    results, pval_list = {}, []
    for model in MODELS_WITH_EXT:
        df_m = (phase_df[(phase_df["model"]==model) &
                         (phase_df["condition"].isin(["direct_information","stochastic_information"]))]
                .copy())
        df_long = df_m.melt(id_vars=["prompt_id","condition"],
                             value_vars=["delta_early","delta_mid","delta_late"],
                             var_name="phase", value_name="delta_eas")
        df_long["phase"] = df_long["phase"].map(
            {"delta_early":"Early","delta_mid":"Mid","delta_late":"Late"})
        df_long = df_long.dropna(subset=["delta_eas"])

        aov = pg.mixed_anova(data=df_long, dv="delta_eas", within="phase",
                              between="condition", subject="prompt_id")
        print(f"\n  {model.upper()}")
        print(aov[["Source","DF1","DF2","F","p-unc","np2"]].to_string(index=False))

        for source in ["condition","phase","condition * phase"]:
            row = aov[aov["Source"]==source]
            if not row.empty:
                pval_list.append((model, source, float(row["p-unc"].values[0]),
                                  float(row["F"].values[0]), float(row["np2"].values[0])))
        results[model] = {"aov":aov, "df_long":df_long}

    labels   = [f"{m} — {s}" for m,s,*_ in pval_list]
    raw_pvals = [p for _,_,p,*_ in pval_list]
    reject, p_adj = fdr_correct(raw_pvals)
    print("\n  BH-FDR correction across all effects:")
    for label, p_r, p_f, rej in zip(labels, raw_pvals, p_adj, reject):
        print(f"    {label:<55}  p_raw={p_r:.4f}  p_fdr={p_f:.4f}  {'REJECT' if rej else 'retain'}")
    for (model,source,_,f_val,np2), p_f, rej in zip(pval_list, p_adj, reject):
        key = source.replace(" * ","_x_")
        results[model][f"p_fdr_{key}"] = p_f
        results[model][f"F_{key}"]     = f_val
        results[model][f"np2_{key}"]   = np2
    return results


def plot_5(results, ax1, ax2):
    phase_order = ["Early","Mid","Late"]
    cond_specs  = [("direct_information","#2196F3","DI — Direct Info"),
                   ("stochastic_information","#E91E63","SI — Stochastic Info")]
    for ax, model in zip([ax1,ax2], MODELS_WITH_EXT):
        df_long = results[model]["df_long"]
        for cond,color,label in cond_specs:
            sub   = df_long[df_long["condition"]==cond]
            means = sub.groupby("phase")["delta_eas"].mean().reindex(phase_order)
            sems  = sub.groupby("phase")["delta_eas"].sem().reindex(phase_order)
            ax.plot(phase_order, means.values, color=color, lw=2.5,
                     marker="o", markersize=7, label=label, zorder=4)
            ax.fill_between(phase_order, means.values-sems.values, means.values+sems.values,
                             color=color, alpha=0.15)
        lines = []
        for sk, sl in [("condition","Cond"),("phase","Phase"),("condition_x_phase","Cond×Phase")]:
            fk = f"p_fdr_{sk}"
            if fk in results[model]:
                p_f = results[model][fk]
                lines.append(f"{sl}: p_fdr={p_f:.3f}{sig_stars(p_f)}")
        ax.text(0.97,0.97,"\n".join(lines), transform=ax.transAxes,
                 ha="right", va="top", fontsize=7.5,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("Response Phase", fontsize=9)
        ax.set_ylabel("ΔEAS vs NC Baseline", fontsize=9)
        ax.set_title(f"Analysis 5: Mixed ANOVA — {model.upper()}\n"
                      "Condition × Phase: does DI entropy drop faster than SI?",
                      fontsize=9.5, fontweight="bold")
        ax.legend(fontsize=8.5, framealpha=0.7); ax.spines[["top","right"]].set_visible(False)


# ── Global FDR Summary ────────────────────────────────────────────
def print_global_summary(res1, res3_between, res4, res5):
    print("\n"+"="*70)
    print("GLOBAL SUMMARY: One primary test per analysis (BH-FDR, k=5)")
    print("="*70)
    entries = [
        ("A1: RM-ANOVA main effect, DeepSeek-V3", res1["deepseek"]["p"]),
        ("A3: Fisher z, MLA vs GQA (between)",    res3_between["p"]),
        ("A4: Efficiency ANOVA, Qwen",             res4["qwen"]["p"]),
        ("A5: Interaction p, Qwen (raw)",
         [v for k,v in res5["qwen"].items() if "p_fdr_condition_x_phase" in k][0]
         if any("p_fdr_condition_x_phase" in k for k in res5["qwen"]) else 1.0),
    ]
    pvals  = [p for _,p in entries]
    labels = [l for l,_ in entries]
    reject, p_adj = fdr_correct(pvals)
    print(f"\n  {'Analysis':<48} {'p_raw':>8} {'p_fdr':>8}  Decision")
    print(f"  {'-'*72}")
    for label,p_r,p_f,rej in zip(labels,pvals,p_adj,reject):
        print(f"  {label:<48} {p_r:>8.4f} {p_f:>8.4f}  {'REJECT H0' if rej else 'retain H0'}")

# ── Custom Analysis: Architecture-Specific Regressions ────────────
def analysis_architecture_regressions(full_df):
    print("\n"+"="*70)
    print("ANALYSIS: Architecture-Specific Regressions")
    print("="*70)
    
    # Filter to only include the experimental conditions (exclude no_context)
    df_r = full_df[full_df["condition"].isin(COND_ORDER)].copy()
    
    # Standardize predictors (z-scoring) to match your standard statistical format
    # Assumes 'ctx_tokens' exists in your pilot summary CSVs
    predictors = ["acc_rater", "acc_sim", "ctx_tokens"]
    
    for col in predictors:
        if col in df_r.columns:
            df_r[f"{col}_z"] = (df_r[col] - df_r[col].mean()) / df_r[col].std()
        else:
            raise KeyError(f"Column '{col}' not found in the data. Please verify the column name in pilot CSVs.")

    # ── REGRESSION 1: MLA Models ──
    # DV: delta_entropy
    # IV: acc_rater_z (primary), ctx_tokens_z (covariate)
    df_mla = df_r[df_r["MLA"] == 1].dropna(subset=["delta_entropy", "acc_rater_z", "ctx_tokens_z"])
    
    print("\n[ Regression 1: MLA Architecture Models (DS-V2-Lite, DeepSeek-V3) ]")
    print("Primary Measure: Pragmatic Relevance (acc_rater) | Covariate: Context Tokens (ctx_tokens)")
    
    fit_mla = ols("delta_entropy ~ acc_rater_z + ctx_tokens_z", data=df_mla).fit()
    
    # Extracting coefficients in the style of your original script
    tbl_mla = fit_mla.summary2().tables[1]
    print(f"\n  {'Predictor':<18} {'beta':>8} {'SE':>7} {'t':>7} {'p_raw':>8}")
    print(f"  {'-'*55}")
    for name in [n for n in tbl_mla.index if n != "Intercept"]:
        b = float(tbl_mla.loc[name,"Coef."]); se = float(tbl_mla.loc[name,"Std.Err."])
        t = float(tbl_mla.loc[name,"t"]); p_r = float(tbl_mla.loc[name,"P>|t|"])
        print(f"  {name:<18} {b:>+8.4f} {se:>7.4f} {t:>+7.3f} {p_r:>8.4f}")
    
    print(f"  R²={fit_mla.rsquared:.4f}  Adj-R²={fit_mla.rsquared_adj:.4f}  F={fit_mla.fvalue:.3f}  p={fit_mla.f_pvalue:.4f}")


    # ── REGRESSION 2: GQA Models ──
    # DV: delta_entropy
    # IV: acc_sim_z (primary), ctx_tokens_z (covariate)
    df_gqa = df_r[df_r["MLA"] == 0].dropna(subset=["delta_entropy", "acc_sim_z", "ctx_tokens_z"])
    
    print("\n\n[ Regression 2: GQA Architecture Models (Mistral, Qwen, LLaMA) ]")
    print("Primary Measure: Lexical Similarity (acc_sim) | Covariate: Context Tokens (ctx_tokens)")
    
    fit_gqa = ols("delta_entropy ~ acc_sim_z + ctx_tokens_z", data=df_gqa).fit()
    
    tbl_gqa = fit_gqa.summary2().tables[1]
    print(f"\n  {'Predictor':<18} {'beta':>8} {'SE':>7} {'t':>7} {'p_raw':>8}")
    print(f"  {'-'*55}")
    for name in [n for n in tbl_gqa.index if n != "Intercept"]:
        b = float(tbl_gqa.loc[name,"Coef."]); se = float(tbl_gqa.loc[name,"Std.Err."])
        t = float(tbl_gqa.loc[name,"t"]); p_r = float(tbl_gqa.loc[name,"P>|t|"])
        print(f"  {name:<18} {b:>+8.4f} {se:>7.4f} {t:>+7.3f} {p_r:>8.4f}")
        
    print(f"  R²={fit_gqa.rsquared:.4f}  Adj-R²={fit_gqa.rsquared_adj:.4f}  F={fit_gqa.fvalue:.3f}  p={fit_gqa.f_pvalue:.4f}")

    return fit_mla, fit_gqa

# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    print("Loading data...")
    full_df  = load_all_data()
    phase_df = load_phase_data()
    fit_mla, fit_gqa = analysis_architecture_regressions(full_df)
    print(f"  Full: {len(full_df)} rows, {full_df['model'].nunique()} models")
    print(f"  Phase: {len(phase_df)} rows")

    res1                    = analysis_1(full_df)
    fit, coef_res, df_r     = analysis_2(full_df)
    corr_res, between3      = analysis_3(full_df)
    res4                    = analysis_4(full_df)
    res5                    = analysis_5(phase_df)

    print_global_summary(res1, between3, res4, res5)

    # ── Build figure ──
    print("\nBuilding figure...")
    fig = plt.figure(figsize=(20, 32))
    fig.patch.set_facecolor("#F7F7F7")
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.62, wspace=0.38, top=0.955, bottom=0.018)
    def wax(r,c):
        ax = fig.add_subplot(gs[r,c]); ax.set_facecolor("white"); return ax

    plot_1(res1,   wax(0,0), wax(0,1))
    plot_2(fit, coef_res, df_r, wax(1,0), wax(1,1))
    plot_3(corr_res, between3, wax(2,0), wax(2,1))
    plot_4(res4,   wax(3,0), wax(3,1))
    plot_5(res5,   wax(4,0), wax(4,1))

    fig.suptitle(
        "LLM Pragmatic Context Processing — Social Science Statistical Analyses\n"
        "Within-analysis post-hoc corrections: Bonferroni  |  Across models & analyses: BH-FDR\n"
        "* p < .05   ** p < .01   *** p < .001   n.s. = non-significant after correction",
        fontsize=11.5, fontweight="bold", y=0.998
    )
    out = str(OUT_DIR / "social_science_analyses.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Figure saved -> {out}")
    print("Done.")
