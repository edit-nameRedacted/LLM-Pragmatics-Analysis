from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent

# 1. Load the compiled data
df = pd.read_csv(ROOT / 'data/human/data_compiled_sim.csv')

# ==============================================================================
# --- DATA PREPARATION: BEAM & SEMANTIC ENTROPY ---
# ==============================================================================

base_cols = ['question', 'condition', 'rater_scores', 'context_question_similarity']

# Extract Beam Cluster Entropy
beam_cols = [c for c in df.columns if 'beam_mean_cosine_sim' in c and 'delta' not in c.lower()]
df_beam = df[base_cols + beam_cols].melt(
    id_vars=base_cols, var_name='model', value_name='beam_entropy'
)
df_beam['model'] = df_beam['model'].str.split('_').str[0]

# Extract Semantic Entropy (Substituting for missing EAS phase data)
sem_cols = [c for c in df.columns if 'mean_token_entropy' in c and 'delta' not in c.lower()]
df_sem = df[['question', 'condition'] + sem_cols].melt(
    id_vars=['question', 'condition'], var_name='model', value_name='mean_token_entropy'
)
df_sem['model'] = df_sem['model'].str.split('_').str[0]

# Merge into a single long dataframe
df_long = df_beam.merge(df_sem, on=['question', 'condition', 'model'])

# Map models to readable names
model_map = {'DS': 'DeepSeek-V3', 'D1': 'DS-V2-Lite', 'LL': 'LLaMA', 'QW': 'Qwen', 'MS': 'Mistral'}
df_long['model'] = df_long['model'].map(model_map)

# Extract Baseline (no_context) values
df_nc = df_long[df_long['condition'] == 'no_context'][['question', 'model', 'beam_entropy', 'mean_token_entropy']]
df_nc = df_nc.rename(columns={'beam_entropy': 'nc_beam', 'mean_token_entropy': 'nc_sem'})

# Calculate Deltas (Condition - Baseline)
df_analysis = df_long[df_long['condition'] != 'no_context'].merge(df_nc, on=['question', 'model'])
df_analysis['delta_beam'] = df_analysis['beam_entropy'] - df_analysis['nc_beam']
df_analysis['delta_sem']  = df_analysis['mean_token_entropy'] - df_analysis['nc_sem']

# Standardize Predictors (Z-scoring)
df_analysis['acc_rater_z'] = (df_analysis['rater_scores'] - df_analysis['rater_scores'].mean()) / df_analysis['rater_scores'].std()
df_analysis['acc_sim_z']   = (df_analysis['context_question_similarity'] - df_analysis['context_question_similarity'].mean()) / df_analysis['context_question_similarity'].std()

# Drop rows with missing values
df_clean = df_analysis.dropna(subset=['delta_beam', 'delta_sem', 'acc_rater_z', 'acc_sim_z'])


# ==============================================================================
# --- REGRESSIONS: MODEL x CONDITION x TARGET ---
# ==============================================================================
print("="*80)
print("SEQUENCE-LEVEL ENTROPY: HEAD-TO-HEAD PREDICTORS")
print("Targets: Delta Beam Entropy | Delta Semantic Entropy")
print("Competing IVs: Pragmatic Relevance (acc_rater) vs Lexical Similarity (acc_sim)")
print("(* p < 0.05, . p < 0.10)")
print("="*80)

models = df_clean['model'].unique()
conditions = df_clean['condition'].unique()

for m in models:
    print(f"\n\n{'#'*65}")
    print(f" MODEL: {m}")
    print(f"{'#'*65}")
    
    for cond in conditions:
        sub_df = df_clean[(df_clean['model'] == m) & (df_clean['condition'] == cond)]
        
        # Skip if not enough data points
        if len(sub_df) < 3:
            continue
            
        print(f"\n--- Condition: {cond.upper()} ---")
        
        # 1. Delta Beam Entropy Regression
        fit_beam = smf.ols("delta_beam ~ acc_rater_z + acc_sim_z", data=sub_df).fit()
        print(f"  [ Target: Delta Beam Entropy ]  R²: {fit_beam.rsquared:.3f}")
        tbl_beam = fit_beam.summary2().tables[1]
        
        for row in tbl_beam.index:
            if row == 'Intercept': continue
            sig = '*' if tbl_beam.loc[row, 'P>|t|'] < 0.05 else ('.' if tbl_beam.loc[row, 'P>|t|'] < 0.1 else ' ')
            name = "Pragmatic (Rater)" if 'rater' in row else "Lexical (Sim)"
            print(f"    {name:<18} Coef: {tbl_beam.loc[row, 'Coef.']:>8.4f} | P-Val: {tbl_beam.loc[row, 'P>|t|']:>6.4f} {sig}")

        # 2. Delta Semantic Entropy Regression
        fit_sem = smf.ols("delta_sem ~ acc_rater_z + acc_sim_z", data=sub_df).fit()
        print(f"  [ Target: Delta Semantic Entropy ] R²: {fit_sem.rsquared:.3f}")
        tbl_sem = fit_sem.summary2().tables[1]
        
        for row in tbl_sem.index:
            if row == 'Intercept': continue
            sig = '*' if tbl_sem.loc[row, 'P>|t|'] < 0.05 else ('.' if tbl_sem.loc[row, 'P>|t|'] < 0.1 else ' ')
            name = "Pragmatic (Rater)" if 'rater' in row else "Lexical (Sim)"
            print(f"    {name:<18} Coef: {tbl_sem.loc[row, 'Coef.']:>8.4f} | P-Val: {tbl_sem.loc[row, 'P>|t|']:>6.4f} {sig}")