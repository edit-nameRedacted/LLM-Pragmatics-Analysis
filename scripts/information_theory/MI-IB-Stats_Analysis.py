from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent

# 1. Load the compiled data
df = pd.read_csv(ROOT / 'data/human/data_compiled_sim.csv')

# ==============================================================================
# --- DATA PREPARATION ---
# ==============================================================================

# Select relevant columns
df_entropy = df[['question', 'condition', 'rater_scores', 'context_question_similarity', 'ctx_tokens',
                 'DS_mean_token_entropy', 'D1_mean_token_entropy', 
                 'LL_mean_token_entropy', 'QW_mean_token_entropy', 'MS_mean_token_entropy']]

# Pivot to long format
df_long = df_entropy.melt(
    id_vars=['question', 'condition', 'rater_scores', 'context_question_similarity', 'ctx_tokens'],
    var_name='model', 
    value_name='mean_token_entropy'
)

# Clean up model names for easier reading
model_map = {'DS': 'DeepSeek-V3', 'D1': 'DS-V2-Lite', 'LL': 'LLaMA', 'QW': 'Qwen', 'MS': 'Mistral'}
df_long['model'] = df_long['model'].str.replace('_mean_token_entropy', '').map(model_map)

# Extract baselines (no_context)
df_nc = df_long[df_long['condition'] == 'no_context'][['question', 'model', 'mean_token_entropy']]
df_nc = df_nc.rename(columns={'mean_token_entropy': 'nc_entropy'})

# Merge and filter out no_context
df_analysis = df_long[df_long['condition'] != 'no_context'].merge(df_nc, on=['question', 'model'])

# ==============================================================================
# --- CALCULATE MI & IB METRICS ---
# ==============================================================================

# 1. Mutual Information (MI): H(Baseline) - H(Condition)
# Positive = Context reduced uncertainty. Negative = Context increased uncertainty.
df_analysis['mutual_information'] = df_analysis['nc_entropy'] - df_analysis['mean_token_entropy']

# Replace zeros in denominators with NaN to prevent infinity errors
df_analysis['rater_safe'] = df_analysis['rater_scores'].replace(0, np.nan)
df_analysis['sim_safe']   = df_analysis['context_question_similarity'].replace(0, np.nan)

# 2. Information Bottleneck (IB) Efficiencies
# Epsilon = Mutual Information / Signal Strength
df_analysis['ib_rater'] = df_analysis['mutual_information'] / df_analysis['rater_safe']
df_analysis['ib_sim']   = df_analysis['mutual_information'] / df_analysis['sim_safe']

# Standardize predictors for the regressions (z-scoring)
df_analysis['acc_rater_z'] = (df_analysis['rater_scores'] - df_analysis['rater_scores'].mean()) / df_analysis['rater_scores'].std()
df_analysis['acc_sim_z']   = (df_analysis['context_question_similarity'] - df_analysis['context_question_similarity'].mean()) / df_analysis['context_question_similarity'].std()

df_clean = df_analysis.replace([np.inf, -np.inf], np.nan).dropna(subset=['mutual_information', 'ib_rater', 'ib_sim']).copy()


# ==============================================================================
# --- ANALYSIS 1: RAW MUTUAL INFORMATION (MI) ---
# ==============================================================================
print("\n" + "="*80)
print("ANALYSIS 1: RAW MUTUAL INFORMATION (MI) BY MODEL")
print("How much raw uncertainty does the context resolve?")
print("="*80)

print("\n--- Mean Mutual Information (nats) ---")
mean_mi = df_clean.groupby(['model', 'condition'])['mutual_information'].mean().unstack()
print(mean_mi.round(4))

mi_anova = smf.ols("mutual_information ~ C(condition) * C(model)", data=df_clean).fit()
print("\n--- 2-Way ANOVA: Raw MI ---")
print(sm.stats.anova_lm(mi_anova, typ=2))


# ==============================================================================
# --- ANALYSIS 2: IB EFFICIENCY (PRAGMATIC RELEVANCE) ---
# ==============================================================================
print("\n\n" + "="*80)
print("ANALYSIS 2: IB EFFICIENCY (acc_rater)")
print("DV: ib_rater = MI / rater_scores")
print("How efficiently does the model extract information per unit of human relevance?")
print("="*80)

print("\n--- Mean IB Efficiency (Pragmatic) ---")
mean_ib_rater = df_clean.groupby(['model', 'condition'])['ib_rater'].mean().unstack()
print(mean_ib_rater.round(4))

ib_rater_anova = smf.ols("ib_rater ~ C(condition) * C(model)", data=df_clean).fit()
print("\n--- 2-Way ANOVA: IB_Rater ---")
print(sm.stats.anova_lm(ib_rater_anova, typ=2))


# ==============================================================================
# --- ANALYSIS 3: IB EFFICIENCY (LEXICAL SIMILARITY) ---
# ==============================================================================
print("\n\n" + "="*80)
print("ANALYSIS 3: IB EFFICIENCY (acc_sim)")
print("DV: ib_sim = MI / context_question_similarity")
print("How efficiently does the model extract information per unit of lexical overlap?")
print("="*80)

print("\n--- Mean IB Efficiency (Lexical) ---")
mean_ib_sim = df_clean.groupby(['model', 'condition'])['ib_sim'].mean().unstack()
print(mean_ib_sim.round(4))

ib_sim_anova = smf.ols("ib_sim ~ C(condition) * C(model)", data=df_clean).fit()
print("\n--- 2-Way ANOVA: IB_Sim ---")
print(sm.stats.anova_lm(ib_sim_anova, typ=2))


# ==============================================================================
# --- ANALYSIS 4: HEAD-TO-HEAD REGRESSIONS BY MODEL ---
# ==============================================================================
print("\n\n" + "="*80)
print("ANALYSIS 4: HEAD-TO-HEAD PREDICTORS BY MODEL")
print("DV: Mutual Information | Competing IVs: acc_rater_z vs acc_sim_z")
print("Which signal actually drives the information gain for each model?")
print("="*80)

models = df_clean['model'].unique()

for m in models:
    sub_df = df_clean[df_clean['model'] == m]
    fit = smf.ols("mutual_information ~ acc_rater_z + acc_sim_z", data=sub_df).fit()
    
    print(f"\n[ Model: {m} ]")
    
    # Extract just the coefficients table for clean printing
    tbl = fit.summary2().tables[1]
    print(f"{'Predictor':<15} {'Coef':>8} {'P-Value':>8}  {'Sig'}")
    print("-" * 40)
    for row in tbl.index:
        if row == 'Intercept': continue
        coef = tbl.loc[row, 'Coef.']
        pval = tbl.loc[row, 'P>|t|']
        sig = '*' if pval < 0.05 else ('.' if pval < 0.1 else '')
        
        # Rename for cleaner output
        pretty_name = "Pragmatic (Rater)" if 'rater' in row else "Lexical (Sim)"
        print(f"{pretty_name:<15} {coef:>8.4f} {pval:>8.4f}  {sig}")
    
    print(f"R-squared: {fit.rsquared:.4f}")


    