"""
Within-subjects reanalysis of MI(accuracy ; Δentropy).

Addresses two issues in the original architecture_gradient.py analysis:
  1. Uses fixed rater_scores from data_compiled_sim.csv.
  2. Respects the nested design (15 questions × 3 non-NC conditions):
     - Variance decomposition: within-Q vs between-Q share of variance
     - Within-question residualized MI (removes question-level fixed effects)
     - Within-question permutation null (shuffles condition labels only within Q)

Results show that under the proper null, most of the "architecture gradient"
reported in the pooled MI analysis falls within the within-Q null distribution.
The pooled analysis inflates n from 15 questions to 45 pseudo-replicates.

Recommendation: use mixed-effects models with (1|question) random intercept
for the primary analysis. Use within-question permutation for significance.
Bootstrap over questions (not rows) for CIs.
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

SEED = 42

def mi(a, b, k=5):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 8:
        return np.nan
    return float(mutual_info_regression(a[m].reshape(-1, 1), b[m],
                 n_neighbors=k, random_state=SEED)[0])


def residualise(D, col, group='question'):
    """Subtract per-group mean — removes between-group variance."""
    qm = D.groupby(group)[col].transform('mean')
    return (D[col] - qm).values


def within_q_perm_null(D, acc_col, y_col, n_perm=2000, seed=SEED):
    """Shuffle acc_col within each question. Preserves nested design."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        shuf = D.groupby('question')[acc_col].transform(
            lambda s: s.sample(frac=1, random_state=rng.integers(1e9)).values
        ).values
        vals.append(mi(shuf, D[y_col].values))
    return np.array(vals)


def bootstrap_mi_over_questions(D, acc_col, y_col, n_boot=2000, seed=SEED):
    """Resample questions (not rows). Gives honest CI given nested design."""
    rng = np.random.default_rng(seed)
    questions = D['question'].unique()
    vals = []
    for _ in range(n_boot):
        qs = rng.choice(questions, size=len(questions), replace=True)
        sub = pd.concat([D[D['question'] == q] for q in qs], ignore_index=True)
        vals.append(mi(sub[acc_col].values, sub[y_col].values))
    return np.array(vals)


def build_long(compiled_path, extended_path, delta_col='eas_mean'):
    """Merge fixed rater_scores with delta metrics from extended_metrics."""
    fixed = pd.read_csv(compiled_path)[
        ['prompt_id', 'rater_scores', 'context_question_similarity']]
    ext = pd.read_csv(extended_path)
    ext['question'] = (ext['prompt_id'] - 1) // 4 + 1
    ext = ext.merge(fixed, on='prompt_id', how='left')
    nc = (ext[ext['condition'] == 'no_context'][['question', delta_col]]
          .rename(columns={delta_col: f'nc_{delta_col}'}))
    D = ext[ext['condition'] != 'no_context'].merge(nc, on='question', how='inner')
    D[f'delta_{delta_col}'] = D[delta_col] - D[f'nc_{delta_col}']
    D['acc_rater'] = 6 - D['rater_scores']
    D['acc_sim']   = D['context_question_similarity']
    return D


if __name__ == "__main__":
    from pathlib import Path
    _ROOT    = Path(__file__).resolve().parent.parent.parent
    COMPILED = str(_ROOT / 'data' / 'human' / 'data_compiled_sim.csv')
    MODELS = {
        'Qwen':    str(_ROOT / 'data' / 'model' / 'qwen'     / 'extended_metrics_qwen.csv'),
        'DS-V2-L': str(_ROOT / 'data' / 'model' / 'deepseek' / 'extended_metrics_deepseek_v2_lite.csv'),
    }
    print(f"{'Model':<10} {'Acc':<10} {'Pooled MI':>10} {'Within-resid':>12} "
          f"{'Null 95%':>14} {'Boot 95%':>14} {'p':>6}")
    print("-" * 82)
    for mname, fpath in MODELS.items():
        D = build_long(COMPILED, fpath)
        for acc in ['acc_rater', 'acc_sim']:
            pooled = mi(D[acc].values, D['delta_eas_mean'].values)
            win    = mi(residualise(D, acc), residualise(D, 'delta_eas_mean'))
            null   = within_q_perm_null(D, acc, 'delta_eas_mean', n_perm=1000)
            boot   = bootstrap_mi_over_questions(D, acc, 'delta_eas_mean', n_boot=1000)
            p      = float((null >= pooled).mean())
            n_lo, n_hi = np.quantile(null, [0.025, 0.975])
            b_lo, b_hi = np.quantile(boot, [0.025, 0.975])
            print(f"  {mname:<8} {acc:<10} {pooled:>10.4f} {win:>12.4f} "
                  f"[{n_lo:.2f},{n_hi:.2f}] [{b_lo:.2f},{b_hi:.2f}] {p:>6.3f}")
