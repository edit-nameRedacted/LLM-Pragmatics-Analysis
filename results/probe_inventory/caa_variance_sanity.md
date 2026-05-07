# CAA Displacement Variance Sanity Check

**Generated:** 2026-05-04 11:59  
**Working directory:** `C:\Users\Watcher\Documents\pysandbox\InfoTheoryBot\llm_entropy_study`  

## Scope and data availability

This report checks whether CAA L2 displacement from the NC baseline varies meaningfully *within* each non-NC condition (DI/II/SI), or whether between-condition differences dominate.  Two variants are tested: `caa_mean_l2` (mean across all layers) and `caa_at_peak_layer` (value at each model's empirically-derived peak layer from the RDM notebook).

**Models analysed (instruct):** `deepseek`, `deepseek_v2_lite`, `llama`, `mistral`, `qwen`

**Base models (`llama_base`, `qwen_base`): absent from this analysis.**  
No `extended_metrics_*` file exists for either base model — the CAA computation pipeline (`qxc_main_CAA_FIX.py`) was never run on them.  Extending this sanity check to base models requires running that pipeline first.  Defer that decision until the instruct results below have been evaluated.

> **`qwen` warning:** the underlying `hidden_states_qwen.npz` is 100% NaN-masked (confirmed in probe inventory).  All CAA values for qwen-instruct are derived from those NaN hidden states and are **unverified**.  Rows are included for completeness only.

**Peak layers (from RDM notebook `rdm_analysis_colab_fixed_again.ipynb`, cell 22 — argmax of partial-rho profile, `last_token_hs`, standardised):**  
- `deepseek`: layer 15
- `deepseek_v2_lite`: layer 12
- `llama`: layer 15
- `mistral`: layer 16
- `qwen`: layer 19

---

## Per-condition descriptive statistics (`caa_mean_l2`)

### deepseek

| Condition | n | mean | SD | min | max |
| --------- | - | ---- | -- | --- | --- |
| NC | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| DI | 15 | 36.7902 | 1.7855 | 32.3027 | 40.4473 |
| II | 15 | 38.1459 | 3.9644 | 31.3724 | 44.4279 |
| SI | 15 | 40.7341 | 5.0351 | 35.6832 | 52.4921 |

### deepseek_v2_lite

| Condition | n | mean | SD | min | max |
| --------- | - | ---- | -- | --- | --- |
| NC | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| DI | 15 | 11.4290 | 1.7652 | 9.7435 | 16.3504 |
| II | 15 | 11.6316 | 2.2605 | 8.2596 | 16.2902 |
| SI | 15 | 10.8185 | 2.4426 | 7.7618 | 17.6897 |

### llama

| Condition | n | mean | SD | min | max |
| --------- | - | ---- | -- | --- | --- |
| NC | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| DI | 15 | 11.9965 | 1.2417 | 10.0046 | 13.7418 |
| II | 15 | 10.2106 | 2.1137 | 7.1388 | 15.4657 |
| SI | 15 | 10.7263 | 2.3076 | 6.0850 | 14.3823 |

### mistral

| Condition | n | mean | SD | min | max |
| --------- | - | ---- | -- | --- | --- |
| NC | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| DI | 15 | 7.1626 | 1.7752 | 5.1620 | 11.5567 |
| II | 15 | 7.1450 | 1.6604 | 4.3520 | 10.4817 |
| SI | 15 | 8.9305 | 1.8188 | 6.4706 | 12.2179 |

### qwen ⚠ hidden states NaN

| Condition | n | mean | SD | min | max |
| --------- | - | ---- | -- | --- | --- |
| NC | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| DI | 15 | 28.8019 | 5.1171 | 21.4844 | 38.8162 |
| II | 15 | 32.5394 | 11.6253 | 19.4643 | 60.9413 |
| SI | 15 | 44.2664 | 10.8402 | 25.6525 | 59.8085 |

---

## ANOVA — `caa_mean_l2` (mean L2 across all layers)

η² interpretation: **< 0.30 → OK** (within-condition variance dominates); **0.30–0.60 → MARGINAL**; **> 0.60 → FAIL** (CAA encodes condition identity).

| model | within_SD_DI | within_SD_II | within_SD_SI | F | p | eta_squared | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek | 1.7855 | 3.9644 | 5.0351 | 4.083 | 0.0240 | 0.1628 | OK |
| deepseek_v2_lite | 1.7652 | 2.2605 | 2.4426 | 0.568 | 0.5709 | 0.0263 | OK |
| llama | 1.2417 | 2.1137 | 2.3076 | 3.354 | 0.0445 | 0.1377 | OK |
| mistral | 1.7752 | 1.6604 | 1.8188 | 5.138 | 0.0101 | 0.1966 | OK |
| qwen ⚠ | 5.1171 | 11.6253 | 10.8402 | 10.507 | 0.0002 | 0.3335 | MARGINAL ⚠ |

_⚠ = qwen hidden states are 100% NaN-masked; CAA values unverified._

---

## ANOVA — `caa_at_peak_layer` (L2 at model-specific peak layer)

η² interpretation: **< 0.30 → OK** (within-condition variance dominates); **0.30–0.60 → MARGINAL**; **> 0.60 → FAIL** (CAA encodes condition identity).

| model | within_SD_DI | within_SD_II | within_SD_SI | F | p | eta_squared | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek | 1.1041 | 2.0157 | 1.4213 | 3.660 | 0.0342 | 0.1484 | OK |
| deepseek_v2_lite | 0.5211 | 0.8221 | 0.8848 | 2.174 | 0.1264 | 0.0938 | OK |
| llama | 0.5385 | 0.7614 | 0.7618 | 8.909 | 0.0006 | 0.2979 | OK |
| mistral | 0.2096 | 0.4231 | 0.4029 | 16.394 | < 0.0001 | 0.4384 | MARGINAL |
| qwen ⚠ | 2.0937 | 7.3709 | 4.9337 | 18.557 | < 0.0001 | 0.4691 | MARGINAL ⚠ |

_⚠ = qwen hidden states are 100% NaN-masked; CAA values unverified._

---

## Comparison: all-layer mean vs peak layer

`delta_eta2 = eta2_peak_layer - eta2_mean_layer`  (negative = peak layer is *cleaner* for IB framing).

| model | peak_L* | eta2_mean_layer | verdict_mean | eta2_peak_layer | verdict_peak | delta_eta2 | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek | 15 | 0.1628 | OK | 0.1484 | OK | -0.0143 | Both OK |
| deepseek_v2_lite | 12 | 0.0263 | OK | 0.0938 | OK | +0.0675 | Both OK |
| llama | 15 | 0.1377 | OK | 0.2979 | OK | +0.1602 | Both OK |
| mistral | 16 | 0.1966 | OK | 0.4384 | MARGINAL | +0.2418 | Peak layer worse than mean |
| qwen ⚠ | 19 | 0.3335 | MARGINAL ⚠ | 0.4691 | MARGINAL ⚠ | +0.1356 | Both MARGINAL |

_⚠ = qwen hidden states are 100% NaN-masked._
_Peak layers from RDM notebook cell 22: argmax of partial-rho profile (last\_token\_hs, standardised, controlling for prompt\_token\_len)._

---

## Overall assessment

**Mean-layer (`caa_mean_l2`):** 4/4 verified models OK.  No MARGINAL or FAIL.

**Peak-layer (`caa_at_peak_layer`):** 3/4 verified models OK.  1 MARGINAL/FAIL.

---

## Scatter plots

- `questions_x_context\results\probe_inventory\effort_utility_scatter_deepseek.png`
- `questions_x_context\results\probe_inventory\effort_utility_scatter_deepseek_v2_lite.png`
- `questions_x_context\results\probe_inventory\effort_utility_scatter_llama.png`
- `questions_x_context\results\probe_inventory\effort_utility_scatter_mistral.png`
- `questions_x_context\results\probe_inventory\effort_utility_scatter_qwen.png`
