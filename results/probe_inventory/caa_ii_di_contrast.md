# CAA II-DI Contrast Report

**Generated:** 2026-05-04 12:34  
**Metric:** `caa_mean_l2` (mean L2 displacement from NC baseline, all layers)  
**Bootstrap:** 10,000 BCa resamples, 95 % CI, seed=42  

> ⚠ `qwen [!]`: hidden states 100 % NaN-masked — CAA values unverified.

---

## Per-condition descriptive statistics

| model | n | NC mean | DI mean (SD) | II mean (SD) | SI mean (SD) |
| --- | --- | --- | --- | --- | --- |
| deepseek | 15 | 0.000 | 36.790 (1.786) | 38.146 (3.964) | 40.734 (5.035) |
| deepseek_v2_lite | 15 | 0.000 | 11.429 (1.765) | 11.632 (2.261) | 10.818 (2.443) |
| llama | 15 | 0.000 | 11.996 (1.242) | 10.211 (2.114) | 10.726 (2.308) |
| mistral | 15 | 0.000 | 7.163 (1.775) | 7.145 (1.660) | 8.931 (1.819) |
| qwen [!] | 15 | 0.000 | 28.802 (5.117) | 32.539 (11.625) | 44.266 (10.840) |

---

## II − DI contrast with bootstrap BCa 95 % CI

Positive contrast = II elicits more representational displacement than DI.  
CI excludes zero → contrast is reliably non-zero at 95 % level.

| model | II mean | DI mean | II − DI | 95 % BCa CI | 0 in CI |
| --- | --- | --- | --- | --- | --- |
| deepseek | 38.146 | 36.790 | +1.356 | [-0.754, +3.519] | yes |
| deepseek_v2_lite | 11.632 | 11.429 | +0.203 | [-1.218, +1.581] | yes |
| llama | 10.211 | 11.996 | -1.786 | [-2.868, -0.494] | **no** |
| mistral | 7.145 | 7.163 | -0.018 | [-1.228, +1.119] | yes |
| qwen [!] | 32.539 | 28.802 | +3.738 | [-1.746, +10.976] | yes |

---
