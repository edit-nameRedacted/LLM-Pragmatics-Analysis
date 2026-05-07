# Probe Inventory Report

**Generated:** 2026-05-04 11:18  
**Working directory:** `C:\Users\Watcher\Documents\pysandbox\InfoTheoryBot\llm_entropy_study`  
**Output directory:** `questions_x_context\results\probe_inventory`  

This report audits what is already on disk for the probing-classifier extension to the RDM analysis. It answers five questions per model before any probe code is written.

---

## §1 — Pilot results JSON audit

| File (rel. results/)                   | Model            | n_prompts | responses_key? | SE type | cluster_keys? |
| -------------------------------------- | ---------------- | --------- | -------------- | ------- | ------------- |
| pilot_results_mistral.json             | mistral          | 60        | ✗              | float   | ✗             |
| V2\pilot_results_deepseek.json         | deepseek         | 60        | ✗              | float   | ✗             |
| V2\pilot_results_deepseek_v2_lite.json | deepseek_v2_lite | 60        | ✗              | float   | ✗             |
| V2\pilot_results_llama.json            | llama            | 60        | ✗              | float   | ✗             |
| V2\pilot_results_qwen.json             | qwen             | 60        | ✗              | float   | ✗             |
| Pass 1\pilot_results_deepseek.json     | deepseek         | 60        | ✗              | float   | ✗             |
| Pass 1\pilot_results_llama.json        | llama            | 60        | ✗              | float   | ✗             |
| Pass 1\pilot_results_mistral.json      | mistral          | 60        | ✗              | float   | ✗             |

**Finding:** `semantic_entropy` is a bare `float` scalar in every file. No `responses` key, no cluster assignments, and no `cluster_*` sub-keys exist at any prompt entry in any audited JSON file. The cluster labels that generated each SE scalar were computed transiently inside `_mutual_entailment_cluster()` and were never persisted to disk.


---

## §2 — Raw response text audit

| File (rel. results/)    | Model    | n_prompts | responses_key? | n_responses/prompt |
| ----------------------- | -------- | --------- | -------------- | ------------------ |
| responses_deepseek.json | deepseek | 60        | ✓              | 10                 |

**Associated CSV files:**

| File (rel. results/)   | Model    | n_rows | Columns                                                                                                                                                                                                      |
| ---------------------- | -------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| responses_deepseek.csv | deepseek | 600    | prompt_id, condition, domain, question, context, pairwise, context_question_similarity, similarity_variance_flag, q_tokens, ctx_tokens, combined_tokens, token_limit_flag, response_idx, seed, response_text |

**Finding:** Raw response text is saved for **DeepSeek-instruct only** (`responses_deepseek.json`: 60 prompts × 10 responses; `responses_deepseek.csv`: 600 rows). No `responses_*` file exists for llama, mistral, qwen, deepseek-v2-lite, llama-base, or qwen-base. Response text for those six models is unrecoverable without re-generation.


---

## §3 — `compute_semantic_entropy` side-effect check

**Source:** `questions_x_context/analysis_metrics.py`

**Relevant source excerpts (first 22 lines each):**

```python
# ── compute_semantic_entropy ──
def compute_semantic_entropy(responses: list[str], nli_tok, nli_clf) -> float:
    """Shannon entropy (nats) over NLI-derived semantic cluster distribution."""
    return _cluster_entropy(responses, nli_tok, nli_clf)
```

```python
# ── _cluster_entropy ──
def _cluster_entropy(texts: list[str], nli_tok, nli_clf) -> float:
    """Shannon entropy (nats) over the NLI semantic cluster distribution."""
    texts = [t for t in texts if t.strip()]
    if not texts:
        return 0.0
    labels = _mutual_entailment_cluster(texts, nli_tok, nli_clf)
    counts = np.bincount(labels)
    probs = counts / counts.sum()
    return float(max(0.0, -np.sum(probs * np.log(probs + 1e-10))))


def _nanmean_or_none(vals: list[float]) -> float | None:
    """
    np.nanmean over vals, returning None instead of nan when all values are NaN
    or vals is empty.  np.nanmean raises RuntimeWarning and returns nan for
    all-NaN input; this helper suppresses that case cleanly.
    """
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    if np.all(np.isnan(arr)):
        return None
```

```python
# ── _mutual_entailment_cluster ──
def _mutual_entailment_cluster(texts: list[str], nli_tok, nli_clf) -> list[int]:
    """
    Assign each text to a semantic cluster via mutual NLI entailment.
    Two responses are in the same cluster iff they mutually entail each other.
    Returns a list of integer cluster labels (one per input text).
    """
    texts = [t for t in texts if t.strip()]
    clusters: list[int] = []
    reps: list[str] = []
    for text in texts:
        assigned = False
        for i, rep in enumerate(reps):
            if _nli_entails(text, rep, nli_tok, nli_clf) and _nli_entails(
                rep, text, nli_tok, nli_clf
            ):
                clusters.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append(len(reps))
            reps.append(text)
    return clusters
```

**Call chain:**

```
compute_semantic_entropy(responses, nli_tok, nli_clf)   →  float
  └─ _cluster_entropy(texts, nli_tok, nli_clf)          →  float
       └─ _mutual_entailment_cluster(texts, ...)        →  list[int]  (local variable only)
            ├─ np.bincount(labels)
            └─ return float entropy   [list[int] discarded here]
```

**Finding:** Cluster labels are computed inside `_mutual_entailment_cluster()` as a local `list[int]`. They are consumed immediately by `np.bincount()` and are never returned to the caller, never written to disk, and not stored in any mutable container outside the call frame. `compute_semantic_entropy` returns only the entropy `float`. **No side effect — cluster labels must be re-derived from scratch.**


---

## §4 — `hidden_states_*.npz` schema audit

| Model            | n_prompts | n_layers (L) | hidden_dim (D) | dtype   | nan_prompt%           | meta_csv? |
| ---------------- | --------- | ------------ | -------------- | ------- | --------------------- | --------- |
| deepseek         | 60        | 31           | 4096           | float16 | 0%                    | ✓         |
| deepseek_v2_lite | 60        | 28           | 2048           | float16 | 0%                    | ✓         |
| llama            | 60        | 33           | 4096           | float16 | 0%                    | ✓         |
| llama_base       | 60        | 33           | 4096           | float16 | 0%                    | ✓         |
| mistral          | 60        | 33           | 4096           | float16 | 0%                    | ✓         |
| qwen             | 60        | 29           | 3584           | float16 | **100% — ⚠ EXCLUDED** | ✓         |
| qwen_base        | 60        | 29           | 3584           | float16 | 21.7% (partial)       | ✓         |

**Pooling variants present in all files:** `last_token_hs`, `mean_pool_hs` (reported above). `last_n_hs` (last-8-token window) is present but excluded from probe reporting per scope.

> ⚠ **WARNING — Probe not viable without hidden-state re-extraction. Excluded from probe planning.**
>
> **`qwen`**: `nan_mask` is `True` for 100 % of prompts. The `.npz` was written but every hidden-state vector is NaN. Do not use this file for probe training until a clean hidden-state extraction pass is completed.

**Finding:** Five models (deepseek, deepseek\_v2\_lite, llama, llama\_base, mistral) have 0 % NaN-affected prompts and are probe-ready from disk. Qwen-base has 21.7 % NaN-affected prompts (≈13/60), leaving ≈47 clean prompts — a partial probe is possible. Qwen-instruct is excluded from probe planning (100 % NaN).


---

## §5 — Recoverable cluster-count distribution (DeepSeek-instruct only)

**Source:** `questions_x_context\results\V2\pilot_results_deepseek.json`

_No NLI model loaded. Cluster-count is proxied from stored SE scalars: SE > 0 implies cluster\_count ≥ 2. This proxy cannot distinguish cluster\_count = 2 from cluster\_count = 3+._

| Condition | n_prompts | SE > 0 → cluster≥2 | % prompts cluster≥2 (proxy) | mean SE (nats) |
| --------- | --------- | ------------------ | --------------------------- | -------------- |
| NC        | 15        | 15                 | 100%                        | 1.909          |
| DI        | 15        | 15                 | 100%                        | 2.028          |
| II        | 15        | 15                 | 100%                        | 1.854          |
| SI        | 15        | 15                 | 100%                        | 2.035          |

**Finding:** The SE proxy gives a rough picture of multi-cluster prevalence per condition. Actual per-prompt cluster labels require running `_mutual_entailment_cluster()` on the saved `responses_deepseek.json` texts (60 prompts × C(10, 2) = 45 NLI pairs each = **2 700 NLI pairs total**). This is CPU-feasible with DeBERTa-MNLI in a few minutes — no GPU required.


---

## §6 — Cost estimate: cheapest path to per-sample cluster labels

| Model               | Responses saved?     | Path         | Required action                                                                                                                |
| ------------------- | -------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| deepseek (instruct) | ✓  (60 × 10)         | **Path A**   | Run `_mutual_entailment_cluster()` on `responses_deepseek.json`. DeBERTa only; 2 700 NLI pairs; CPU; ~minutes. **START HERE.** |
| llama (instruct)    | ✗                    | Path B       | Run `qxc_collect_responses.py --llama`; load 8 B model; generate 600 responses; then cluster.                                  |
| mistral (instruct)  | ✗                    | Path B       | Run `qxc_collect_responses.py --mistral`; load 7 B model; generate 600 responses; then cluster.                                |
| deepseek\_v2\_lite  | ✗                    | Path B       | Run `qxc_collect_responses.py --deepseek_v2_lite`; load 16 B model; generate 600 responses; then cluster.                      |
| llama\_base         | ✗                    | Path B       | Run `qxc_collect_responses.py --llama_base`; load 8 B base; generate 600 responses; then cluster.                              |
| qwen\_base          | ✗ + 21.7 % NaN hs    | Path B+      | Generate responses + re-extract hidden states for ~13 NaN-affected prompts.                                                    |
| **qwen (instruct)** | ✗ + **100 % NaN hs** | **EXCLUDED** | ⚠ **Probe not viable without hidden-state re-extraction. Excluded from probe planning.**                                       |

**Path definitions:**

- **Path A** — Call `_mutual_entailment_cluster(responses, nli_tok, nli_clf)` directly on texts already on disk. Requires only DeBERTa-MNLI (CPU-feasible). No generation, no LLM loading.
- **Path B** — Run `qxc_collect_responses.py` for the target model to save response texts, then run `_mutual_entailment_cluster`. Requires loading and running the generative LLM.
- **Path B+** — Path B, plus a partial hidden-state re-extraction pass for NaN-affected prompts.

> ⚠ **WARNING — Qwen-instruct: Probe not viable without hidden-state re-extraction. Excluded from probe planning.**
> `hidden_states_qwen.npz` contains 100 % NaN-masked prompts. Do not include qwen-instruct in any probe training run until a clean extraction pass is completed and the NaN mask is cleared.

**Recommendation:** Implement and validate the probe on DeepSeek-instruct first (Path A). Only extend to other models after validating the probe design on DeepSeek.


---

## Next step

Write `questions_x_context/probe_classifier.py`. Add a function `extract_cluster_labels(model_label: str) -> dict[str, list[int]]` to that file. The function should load `results/responses_{model_label}.json`, call `_mutual_entailment_cluster()` imported from `analysis_metrics.py` on the stored response texts, and return a dict mapping each `prompt_id` (string key) to its list of per-response integer cluster labels. The **probe target variable** is the per-response NLI cluster ID (one integer label per response, yielding a label vector of shape `(n_prompts × n_responses,)`). These labels are to be predicted from the corresponding CAA hidden state (drawn from `rdm/data/hidden_states_{model_label}.npz`, using either `last_token_hs` or `mean_pool_hs` at a chosen layer) via a linear classifier trained to maximise held-out accuracy. The Fano lower bound on I(hidden\_state ; cluster\_structure) is then computed from the probe's test-set error rate as I ≥ H(cluster) − H(cluster | hidden\_state) ≥ H(cluster) − h(ε), where ε is the probe error rate and h(·) is the binary entropy function.
