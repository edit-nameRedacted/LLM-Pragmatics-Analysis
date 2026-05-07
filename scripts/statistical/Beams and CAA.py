from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent.parent

# 1. Load Ground Truth Predictors from the Compiled Sheet
df_meta = pd.read_csv(ROOT / 'data/human/data_compiled_sim.csv')[['prompt_id', 'condition', 'rater_scores', 'context_question_similarity']].drop_duplicates()

# 2. Load and merge Extended Metrics for all five models
models_info = {
    'DeepSeek-V3': ROOT / 'data/model/deepseek/extended_metrics_deepseek.csv',
    'DS-V2-Lite':  ROOT / 'data/model/deepseek/extended_metrics_deepseek_v2_lite.csv',
    'LLaMA':       ROOT / 'data/model/llama/extended_metrics_llama.csv',
    'Mistral':     ROOT / 'data/model/mistral/extended_metrics_mistral.csv',
    'Qwen':        ROOT / 'data/model/qwen/extended_metrics_qwen.csv',
}

extended_frames = []
for model_name, file_name in models_info.items():
    df_ext = pd.read_csv(file_name)
    df_ext['model'] = model_name
    extended_frames.append(df_ext)

df_full = pd.concat(extended_frames, ignore_index=True).merge(df_meta, on=['prompt_id', 'condition'], how='left')

# 3. Calculate Deltas (Condition - No-Context Baseline)
df_full['question_id'] = (df_full['prompt_id'] - 1) // 4
nc_baseline = df_full[df_full['condition'] == 'no_context'][['question_id', 'model', 'eas_early', 'eas_late']]
nc_baseline = nc_baseline.rename(columns={'eas_early': 'nc_early', 'eas_late': 'nc_late'})

df_analysis = df_full[df_full['condition'] != 'no_context'].merge(nc_baseline, on=['question_id', 'model'])
df_analysis['delta_early'] = df_analysis['eas_early'] - df_analysis['nc_early']
df_analysis['delta_late']  = df_analysis['eas_late']  - df_analysis['nc_late']

# 4. Group by Architecture
df_analysis['architecture'] = df_analysis['model'].apply(lambda x: 'MLA' if 'DeepSeek' in x or 'DS-' in x else 'GQA')

# 5. Run the Phase Regression
for arch in ['MLA', 'GQA']:
    sub = df_analysis[df_analysis['architecture'] == arch]
    print(f"\n--- Architecture: {arch} ---")
    # Early Phase Model
    fit_early = smf.ols("delta_early ~ rater_scores + context_question_similarity", data=sub).fit()
    print(f"Early Phase (Pragmatic) p-val: {fit_early.pvalues['rater_scores']:.4f}")
    # Late Phase Model
    fit_late = smf.ols("delta_late ~ rater_scores + context_question_similarity", data=sub).fit()
    print(f"Late Phase (Lexical) p-val: {fit_late.pvalues['context_question_similarity']:.4f}")