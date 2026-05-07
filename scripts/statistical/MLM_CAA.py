from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent.parent

# 1. Load Data
df = pd.read_csv(ROOT / 'data/human/data_compiled_sim.csv')

# 2. Prep data (Standardize predictors and calculate deltas)
df_sub = df[['question', 'condition', 'rater_scores', 'context_question_similarity', 'ctx_tokens',
             'DS_caa_mean_l2', 'DS_caa_mean_cosine']].copy()

for col in ['rater_scores', 'context_question_similarity', 'ctx_tokens']:
    df_sub[f'{col}_z'] = (df_sub[col] - df_sub[col].mean()) / df_sub[col].std()

df_sub = df_sub.rename(columns={
    'rater_scores_z': 'acc_rater_z', 
    'context_question_similarity_z': 'acc_sim_z', 
    'ctx_tokens_z': 'ctx_token_z'
})

# Drop no_context rows and NaNs
df_analysis = df_sub[df_sub['condition'] != 'no_context'].dropna(
    subset=['DS_caa_mean_l2', 'DS_caa_mean_cosine', 'acc_rater_z', 'acc_sim_z', 'ctx_token_z']
)

print("\n" + "="*60)
print(" MULTILEVEL ANALYSIS: DEEPSEEK-V3 CONCEPTUAL TRAJECTORY (CAA)")
print("="*60)

# Model 1: Magnitude of the Conceptual Shift
print("\n--- 1. DV: Magnitude of Latent Shift (delta_CAA_mean) ---")
try:
    md_caa_mag = smf.mixedlm(
        "DS_caa_mean_l2 ~ acc_rater_z * C(condition) + acc_sim_z * C(condition) + ctx_token_z", 
        df_analysis, 
        groups=df_analysis["question"]
    ).fit()
    print(md_caa_mag.summary().tables[1])
except Exception as e:
    print(f"Error fitting magnitude model: {e}")

# Model 2: Direction/Angle of the Conceptual Shift
print("\n--- 2. DV: Angle of Latent Shift (caa_mean_cosine) ---")
try:
    md_caa_cos = smf.mixedlm(
        "DS_caa_mean_cosine ~ acc_rater_z * C(condition) + acc_sim_z * C(condition) + ctx_token_z", 
        df_analysis, 
        groups=df_analysis["question"]
    ).fit()
    print(md_caa_cos.summary().tables[1])
except Exception as e:
    print(f"Error fitting cosine model: {e}")