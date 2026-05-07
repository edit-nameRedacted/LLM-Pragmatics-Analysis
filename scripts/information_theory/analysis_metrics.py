"""
analysis_metrics.py — Information-theoretic metrics for the QxC experiment.

Contains the four core analysis functions plus shared helpers. Imported by
qxc_main.py; can also be used by future experiment scripts.

Functions
---------
run_logit_entropy        – per-token Shannon entropy over sampled LLM outputs
run_beam_divergence      – pairwise cosine similarity + semantic entropy across beams
get_layerwise_hidden_states  – CAA: layer-wise hidden-state extraction
compute_caa_for_question_group  – CAA: per-layer L2/cosine vs no_context baseline
compute_semantic_entropy – NLI-cluster Shannon entropy over a set of responses
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

# ── Constants shared with qxc_main ────────────────────────────────────────────
ENTAILMENT_IDX = 2  # DeBERTa-MNLI label order: contradiction=0, neutral=1, entailment=2
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.8


# ── Logits sanitiser ──────────────────────────────────────────────────────────
class SanitizeLogitsProcessor(LogitsProcessor):
    """
    Replace NaN / ±Inf in the raw logit tensor before any downstream warper
    (temperature, top-p, …) or the sampler sees them.

    Root cause: bitsandbytes 8-bit quantisation on Windows can produce NaN
    logits for a small fraction of vocabulary positions.  After temperature
    scaling those NaNs survive into softmax, which returns NaN probabilities,
    and torch.multinomial raises a CUDA assertion when the distribution
    integrates to zero.

    Fix: clamp NaN → -1e9 (suppress token), +Inf → 1e4, -Inf → -1e9.
    """

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        return torch.nan_to_num(scores, nan=-1e9, posinf=1e4, neginf=-1e9)


_SANITIZE = LogitsProcessorList([SanitizeLogitsProcessor()])


# ── Shared NLI helpers ─────────────────────────────────────────────────────────
def _nli_entails(premise: str, hypothesis: str, nli_tok, nli_clf) -> bool:
    """True if premise entails hypothesis according to DeBERTa-MNLI.

    Moves tokenised inputs onto the NLI model's device so this works whether
    the classifier lives on CPU or GPU.
    """
    inputs = nli_tok(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    # Match the classifier's device — supports both CPU and CUDA deployments
    device = next(nli_clf.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = nli_clf(**inputs).logits
    return int(logits.argmax().item()) == ENTAILMENT_IDX


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
    return float(np.nanmean(arr))


# ── Analysis 1: Per-token logit entropy ───────────────────────────────────────
def run_logit_entropy(
    model,
    tok: AutoTokenizer,
    prompt: str,
    n_samples: int,
    seed: int,
    set_seed_fn,
) -> tuple[list[str], dict]:
    """
    Generate n_samples responses one at a time with output_scores=True.
    Compute Shannon entropy of the softmax distribution at each generated token.

    Per-sample seed = seed + sample_index; caller supplies
    seed = SEED + run_id*10000 + prompt_id*100.

    Parameters
    ----------
    set_seed_fn : callable
        qxc_main.set_seed — passed in to avoid a circular import.

    Returns
    -------
    responses : list[str]
    result : dict
        mean_token_entropy    – mean H across token positions, averaged over samples
        first_token_entropy   – None (invalidated by 8-bit NaN artifact)
        peak_entropy_position – argmax of H, averaged (rounded) across samples
        per_sample            – per-sample mean_token_entropy values
        per_sample_seeds      – seed used for each sample
    """
    input_ids = tok.encode(prompt, return_tensors="pt").to(model.device)
    attn_mask = torch.ones_like(input_ids)
    input_len = input_ids.shape[1]

    responses: list[str] = []
    per_sample_means: list[float] = []
    peak_positions: list[int] = []
    per_sample_seeds: list[int] = []
    token_entropy_sequences: list[list[float]] = []  # one list[float] per sample

    with torch.no_grad():
        for i in range(n_samples):
            sample_seed = seed + i
            set_seed_fn(sample_seed)
            per_sample_seeds.append(sample_seed)
            out = model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tok.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                logits_processor=_SANITIZE,
            )
            text = tok.decode(out.sequences[0][input_len:], skip_special_tokens=True)
            responses.append(text)

            step_H: list[float] = []
            for score in out.scores:
                probs = F.softmax(score[0].float(), dim=-1)
                H = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
                # Clamp tiny negative float artefacts (sign-flip from near-zero
                # entropy tokens, e.g. -0.000000 from certain Qwen outputs).
                step_H.append(max(0.0, H))
            del out
            torch.cuda.empty_cache()

            # Store full per-token sequence for EAS and downstream analysis
            token_entropy_sequences.append(step_H)

            if step_H:
                per_sample_means.append(float(np.mean(step_H)))
                peak_positions.append(int(np.argmax(step_H)))

    result = {
        "mean_token_entropy": float(np.mean(per_sample_means)) if per_sample_means else 0.0,
        "first_token_entropy": None,  # invalidated by 8-bit NaN artifact
        "peak_entropy_position": int(round(float(np.mean(peak_positions)))) if peak_positions else 0,
        "per_sample": per_sample_means,
        "per_sample_seeds": per_sample_seeds,
        "token_entropy_sequence": token_entropy_sequences,  # list[N] of list[T] floats
    }
    return responses, result


# ── EAS temporal shape features ───────────────────────────────────────────────
def eas_shape_features(seq: list[float]) -> dict:
    """
    Extract temporal shape features from a per-token entropy sequence.

    seq : list of floats, one per generated token. Values should already be
          clamped >= 0 (use max(0.0, H) at collection time).

    Returns a dict of scalar features, or an empty dict if seq is too short
    (< 4 tokens) to produce meaningful shape statistics.

    Keys returned (all prefixed eas_ except the base eas scalar):
        eas                  – normalised area (simple mean; trapezoid equiv)
        eas_early/mid/late   – mean H in temporal thirds of the sequence
        eas_final_quarter    – mean H in the final 25% of tokens
        eas_slope            – linear regression slope (nats per token)
        eas_mean_rate        – mean first-difference (nats per step)
        eas_peak_rate        – max |first-difference|
        eas_peak_rate_position – token index of max |first-difference|
        eas_n_spikes         – count of tokens where H > mean + 1.5 SD
        eas_skew             – centre-of-mass / n - 0.5  (neg = front-loaded)
        eas_sparsity         – fraction of tokens with H > 0.01
    """
    s = np.array(seq, dtype=float)
    n = len(s)
    if n < 4:
        return {}

    # Drop position 0 if it is an anomalous spike (BOS / chat-template artefact):
    # anomalous = more than 3 SD above the rest of the sequence.
    if n > 2:
        rest = s[1:]
        if s[0] > rest.mean() + 3 * rest.std():
            s = s[1:]
            n = n - 1

    # 1. EAS (normalised area = simple mean over the cleaned sequence)
    eas = float(np.sum(s) / n)

    # 2. Temporal thirds
    t1, t2, t3 = np.array_split(s, 3)
    eas_early = float(np.sum(t1) / max(len(t1), 1))
    eas_mid   = float(np.sum(t2) / max(len(t2), 1))
    eas_late  = float(np.sum(t3) / max(len(t3), 1))

    # 3. Final quarter
    final_q = s[int(0.75 * n):]
    eas_final_quarter = float(np.mean(final_q)) if len(final_q) else 0.0

    # 4. Entropy trajectory slope (linear regression over token index)
    t_idx = np.arange(n)
    slope = float(np.polyfit(t_idx, s, 1)[0])  # nats per token

    # 5. Rate-of-change statistics (first differences)
    diffs = np.diff(s)
    mean_rate = float(np.mean(diffs))
    if len(diffs):
        peak_rate     = float(np.max(np.abs(diffs)))
        peak_rate_pos = int(np.argmax(np.abs(diffs)))
    else:
        peak_rate     = 0.0
        peak_rate_pos = 0

    # 6. Entropy spikes (tokens where H > mean + 1.5 SD of non-zero positions)
    nonzero = s[s > 0.01]
    if len(nonzero) > 2:
        spike_thresh = nonzero.mean() + 1.5 * nonzero.std()
        n_spikes = int(np.sum(s > spike_thresh))
    else:
        n_spikes = 0

    # 7. Skewness of entropy over time
    # Positive = uncertainty weighted toward end; negative = toward start.
    weights = s / (s.sum() + 1e-10)
    centre_of_mass = float(np.sum(t_idx * weights))
    skew_direction = float((centre_of_mass / n) - 0.5)

    # 8. Density of non-zero positions
    sparsity = float(np.mean(s > 0.01))

    return dict(
        eas=eas,
        eas_early=eas_early,
        eas_mid=eas_mid,
        eas_late=eas_late,
        eas_final_quarter=eas_final_quarter,
        eas_slope=slope,
        eas_mean_rate=mean_rate,
        eas_peak_rate=peak_rate,
        eas_peak_rate_position=peak_rate_pos,
        eas_n_spikes=n_spikes,
        eas_skew=skew_direction,
        eas_sparsity=sparsity,
    )


# ── Analysis 2: Beam search divergence ────────────────────────────────────────
def _mean_pool_embedding(
    model, tok: AutoTokenizer, prompt: str, text: str
) -> tuple[list[np.ndarray], int]:
    """
    Embed prompt + response and return mean-pooled hidden states for EVERY
    transformer layer (plus the embedding-layer output at index 0).

    This supports layer-wise beam divergence analysis. The forward pass already
    computes all layers (output_hidden_states=True); previously everything but
    the final layer was discarded. Now we retain all of them at negligible
    additional cost since pooling collapses the sequence dimension immediately.

    Returns
    -------
    per_layer_pooled : list[np.ndarray]
        One pooled vector per layer, each of shape (hidden_dim,).
        Length = n_layers + 1 (index 0 is the embedding layer output).
    nan_count : int
        Total token-layer positions containing NaN/Inf before sanitization,
        summed across all layers. Used only as a diagnostic flag.
    """
    combined = prompt + text
    enc = tok(combined, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(model.device)
    attn_mask = enc["attention_mask"].to(model.device)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,   # no autoregressive generation, cache is wasted allocation
        )

    per_layer_pooled: list[np.ndarray] = []
    total_nan_positions = 0

    for layer_h in out.hidden_states:
        h = layer_h[0].float()  # (seq_len, hidden_dim)

        # Per-layer NaN detection & sanitisation — NaN positions can differ
        # across layers under quantisation, so we don't reuse a single mask.
        nan_mask = ~torch.isfinite(h).all(dim=-1)
        total_nan_positions += int(nan_mask.sum().item())
        h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)

        # Masked mean pool: exclude sanitised positions from the denominator
        valid_mask = (~nan_mask).float().unsqueeze(-1)
        valid_count = valid_mask.sum().clamp(min=1.0)
        pooled = (h * valid_mask).sum(dim=0) / valid_count
        per_layer_pooled.append(pooled.cpu().numpy())

    return per_layer_pooled, total_nan_positions


def run_beam_divergence(
    model,
    tok: AutoTokenizer,
    prompt: str,
    num_beams: int,
    nli_tok,
    nli_clf,
    seed: int,
    set_seed_fn,
    sbert_model=None,
) -> dict:
    """
    Run deterministic beam search and measure diversity among the beam outputs.

    Parameters
    ----------
    sbert_model : SentenceTransformer | None
        If provided, computes SBERT pairwise cosine similarity across beams
        (CPU, no-prompt mode). Pass None to skip.

    Returns
    -------
    dict with keys:
        mean_pairwise_cosine_similarity  – hidden-state mean pairwise cosine
                                           at the FINAL transformer layer only
                                           (backward-compatible scalar)
        beam_per_layer_cosine            – list[float], mean pairwise cosine
                                           across beams at EACH layer (index 0
                                           is the embedding layer, index -1 is
                                           the final layer). Length = n_layers+1.
                                           Empty if <2 valid beams.
        semantic_cluster_entropy         – NLI cluster entropy (existing)
        embed_nan_positions_sanitized    – total NaN/Inf token-layer positions
                                           sanitised across all layers & beams
        beam_score_entropy               – Shannon entropy over beam sequence scores
        beam_score_gap                   – log-prob gap between beam 0 and beam 1
        beam_score_raw_std               – std of raw sequence log-probs across beams
        beam_score_raw_range             – range (max-min) of raw sequence log-probs
        beam_first_divergence_position   – first token position where beams differ
        beam_length_mean                 – mean word count across beam responses
        beam_length_sd                   – std of word count across beam responses
        beam_sbert_cosine                – SBERT mean pairwise cosine (None if no model)
    """
    set_seed_fn(seed)
    input_ids = tok.encode(prompt, return_tensors="pt").to(model.device)
    attn_mask = torch.ones_like(input_ids)
    input_len = input_ids.shape[1]

    # ── Diverse generation via seeded independent sampling ───────────────────
    # transformers >= 4.57 deprecated native group-beam-search (num_beam_groups).
    # We replace it with num_beams separate sampling runs, each with a unique seed
    # and temperature=1.0, which gives:
    #   • genuinely different output sequences (real diversity)
    #   • non-uniform sequence log-probabilities (meaningful score entropy)
    #   • fully reproducible outputs (seed is deterministic)
    # Sequence log-probability is accumulated from per-step log P(t_i | t_{<i}).
    all_sequences: list[torch.Tensor] = []
    beam_texts: list[str] = []
    seq_log_probs: list[float] = []

    with torch.no_grad():
        for b in range(num_beams):
            set_seed_fn(seed + b)
            out_b = model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=1.0,           # unmodified distribution → valid log-probs
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                logits_processor=_SANITIZE,
            )
            seq = out_b.sequences[0]       # (input_len + gen_len,)
            all_sequences.append(seq)
            text = tok.decode(seq[input_len:], skip_special_tokens=True)
            beam_texts.append(text)

            # Accumulate length-normalised log-probability (nats per token).
            # Normalising by generation length matches the convention used by
            # beam search `sequences_scores` and prevents long responses from
            # always dominating the score distribution merely due to length.
            log_prob = 0.0
            n_tokens = 0
            for step_idx, step_score in enumerate(out_b.scores):
                token_id = int(seq[input_len + step_idx].item())
                log_p = float(
                    F.log_softmax(step_score[0].float(), dim=-1)[token_id].item()
                )
                log_prob += log_p
                n_tokens += 1
                if token_id == tok.eos_token_id:
                    break
            seq_log_probs.append(log_prob / max(n_tokens, 1))
            del out_b
            torch.cuda.empty_cache()

    beam_texts = [t for t in beam_texts if t.strip()]

    # ── 3a. Beam score metrics ────────────────────────────────────────────────
    if seq_log_probs:
        seq_scores = np.array(seq_log_probs, dtype=float)

        # Raw score spread — more sensitive than entropy to small score differences
        # because softmax compresses them into near-uniform distributions.
        beam_score_raw_std   = float(np.std(seq_scores))
        beam_score_raw_range = float(seq_scores.max() - seq_scores.min())

        # Softmax entropy over sequence scores
        probs = np.exp(seq_scores - seq_scores.max())
        probs = probs / probs.sum()
        beam_score_entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        score_gap = float(seq_scores[0] - seq_scores[1]) if len(seq_scores) > 1 else 0.0
    else:
        beam_score_raw_std   = None
        beam_score_raw_range = None
        beam_score_entropy   = None
        score_gap            = None

    # ── 3b. First-divergence token position ───────────────────────────────────
    # Stack generated token ids (pad shorter sequences to the longest length)
    gen_len = max(s.shape[0] - input_len for s in all_sequences) if all_sequences else 0
    if gen_len > 0 and len(all_sequences) > 1:
        padded = torch.full(
            (len(all_sequences), gen_len),
            tok.eos_token_id,
            dtype=torch.long,
        )
        for i, s in enumerate(all_sequences):
            g = s[input_len:]
            padded[i, : g.shape[0]] = g
        first_divergence = gen_len  # default: sequences never diverged
        for pos in range(gen_len):
            if not (padded[:, pos] == padded[0, pos]).all():
                first_divergence = pos
                break
    else:
        first_divergence = 0

    # ── 3c. Beam response length distribution ─────────────────────────────────
    beam_lengths = [len(t.split()) for t in beam_texts]
    beam_length_mean = float(np.mean(beam_lengths)) if beam_lengths else 0.0
    beam_length_sd   = float(np.std(beam_lengths))  if beam_lengths else 0.0

    # ── 3d. SBERT pairwise beam similarity ────────────────────────────────────
    if sbert_model is not None and len(beam_texts) > 1:
        embs = sbert_model.encode(beam_texts, convert_to_numpy=True, show_progress_bar=False)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
        sbert_cos_mat = embs @ embs.T
        n_s = len(embs)
        sbert_mean_cosine = float((sbert_cos_mat.sum() - n_s) / (n_s * (n_s - 1)))
    else:
        sbert_mean_cosine = None

    # ── Layer-wise hidden-state pairwise beam cosine ──────────────────────────
    # `_mean_pool_embedding` now returns one pooled vector per layer per beam.
    # We compute the mean pairwise cosine across beams at EACH layer, giving a
    # trajectory showing how representational convergence/divergence evolves
    # with depth. The final-layer value is kept as `mean_pairwise_cosine_similarity`
    # so downstream code referencing that scalar still works.
    embeddings_per_beam: list[list[np.ndarray]] = []
    total_nan_positions = 0
    for t in beam_texts:
        per_layer_vecs, nan_count = _mean_pool_embedding(model, tok, prompt, t)
        embeddings_per_beam.append(per_layer_vecs)
        total_nan_positions += nan_count
    if total_nan_positions > 0:
        print(
            f"  [beam embed] {total_nan_positions} NaN/Inf token-layer positions "
            f"sanitized across {len(beam_texts)} beams × all layers"
        )

    per_layer_cos: list[float] = []
    if len(embeddings_per_beam) >= 2 and embeddings_per_beam[0]:
        n_layers = len(embeddings_per_beam[0])
        n_beams = len(embeddings_per_beam)
        for L in range(n_layers):
            mat = np.stack([embeddings_per_beam[b][L] for b in range(n_beams)])
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
            normed = mat / norms
            cos_mat = normed @ normed.T
            per_layer_cos.append(
                float((cos_mat.sum() - n_beams) / (n_beams * (n_beams - 1)))
            )
        mean_cos_sim = per_layer_cos[-1]   # backward-compatible scalar = final layer
    else:
        mean_cos_sim = 1.0

    cluster_H = _cluster_entropy(beam_texts, nli_tok, nli_clf)

    return {
        "mean_pairwise_cosine_similarity": mean_cos_sim,
        "beam_per_layer_cosine": per_layer_cos,
        "semantic_cluster_entropy": cluster_H,
        "embed_nan_positions_sanitized": total_nan_positions,
        "beam_score_entropy": beam_score_entropy,
        "beam_score_gap": score_gap,
        "beam_score_raw_std": beam_score_raw_std,
        "beam_score_raw_range": beam_score_raw_range,
        "beam_first_divergence_position": first_divergence,
        "beam_length_mean": beam_length_mean,
        "beam_length_sd": beam_length_sd,
        "beam_sbert_cosine": sbert_mean_cosine,
    }


# ── Analysis 3: CAA hidden state displacement ─────────────────────────────────
def get_layerwise_hidden_states(
    model, tok: AutoTokenizer, prompt: str
) -> tuple[list[np.ndarray], list[int]]:
    """
    Forward pass over `prompt`; extract the last-token hidden state from every
    transformer layer (including the embedding layer output at index 0).

    Returns
    -------
    layer_vecs : list[np.ndarray]
        One vector per layer, shape (hidden_dim,). NaN/Inf positions are
        sanitized in-place and the layer index is recorded in `nan_layers`.
    nan_layers : list[int]
        Indices of layers where NaN/Inf values were detected and sanitized.
        Useful for flagging prompt_id rows with quantisation artifacts.
    """
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    attn_mask = enc["attention_mask"].to(model.device)
    last_pos = input_ids.shape[1] - 1

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    layer_vecs: list[np.ndarray] = []
    nan_layers: list[int] = []

    for i, h in enumerate(out.hidden_states):
        vec = h[0, last_pos, :].cpu().float().numpy()
        if np.any(~np.isfinite(vec)):
            nan_layers.append(i)
            vec = np.nan_to_num(vec, nan=0.0, posinf=1e4, neginf=-1e4)
        layer_vecs.append(vec)

    return layer_vecs, nan_layers


def compute_caa_for_question_group(
    hs_by_condition: dict[str, list[np.ndarray]],
) -> dict[str, dict]:
    """
    Per-layer displacement of each condition's hidden states relative to the
    no_context baseline, with a directional-coherence score against
    direct_information (when present).

    Returns a dict keyed by condition; each entry contains:
        per_layer_l2_vs_no_context      – L2 norm of (cond_hs[l] - nc_hs[l])
        per_layer_cosine_vs_no_context  – cosine(nc_hs[l], cond_hs[l])
        displacement_cosine_vs_direct   – mean cosine of this condition's
                                          displacement vector with DI's
                                          displacement vector, across layers
                                          (None if DI absent for this question)
    """
    nc_hs = hs_by_condition.get("no_context")
    if nc_hs is None:
        return {}

    n_layers = len(nc_hs)
    dir_hs = hs_by_condition.get("direct_information")

    # Pre-compute the direct-information displacement vector for coherence metric
    dir_disp = None
    if dir_hs is not None:
        dir_disp = [dir_hs[l] - nc_hs[l] for l in range(n_layers)]

    caa_results: dict[str, dict] = {}

    for cond in (
        "no_context",
        "direct_information",
        "implicature_information",
        "stochastic_information",
    ):
        if cond not in hs_by_condition:
            continue

        cond_hs = hs_by_condition[cond]
        l2_list, cos_list, cos_vs_dir = [], [], []

        for l in range(n_layers):
            diff = cond_hs[l] - nc_hs[l]
            l2_list.append(float(np.linalg.norm(diff)))

            # Cosine vs No_Context
            norm_a, norm_b = np.linalg.norm(nc_hs[l]), np.linalg.norm(cond_hs[l])
            if norm_a > 1e-9 and norm_b > 1e-9:
                cos_list.append(float(np.dot(nc_hs[l], cond_hs[l]) / (norm_a * norm_b)))
            else:
                cos_list.append(1.0)

            # Cosine vs Direct (directional coherence)
            if dir_disp is not None:
                norm_d, norm_diff = np.linalg.norm(dir_disp[l]), np.linalg.norm(diff)
                if norm_d > 1e-9 and norm_diff > 1e-9:
                    cos_vs_dir.append(
                        float(np.dot(dir_disp[l], diff) / (norm_d * norm_diff))
                    )
                else:
                    cos_vs_dir.append(1.0)

        caa_results[cond] = {
            "per_layer_l2_vs_no_context": l2_list,
            "per_layer_cosine_vs_no_context": cos_list,
            "displacement_cosine_vs_direct": (
                _nanmean_or_none(cos_vs_dir) if cos_vs_dir else None
            ),
        }

    return caa_results


# ── Analysis 4: Semantic entropy ──────────────────────────────────────────────
def compute_semantic_entropy(responses: list[str], nli_tok, nli_clf) -> float:
    """Shannon entropy (nats) over NLI-derived semantic cluster distribution."""
    return _cluster_entropy(responses, nli_tok, nli_clf)
