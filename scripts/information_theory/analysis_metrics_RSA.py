"""
analysis_metrics_RSA.py — Hidden-state extraction for RSA data collection.

Stripped-down counterpart to analysis_metrics.py. Provides a single function
that runs one forward pass per prompt and returns three pooling variants of
the per-layer hidden states, along with NaN-sanitisation metadata.

Pooling variants saved per layer
--------------------------------
  last_token   — hidden state at the final input token position
  mean_pool    — mean over all prompt token positions
  last_n       — full hidden states for the final LAST_N positions
                 (zero-padded on the left if prompt shorter than LAST_N)

Why three variants?
  RSA on chat-templated prompts is sensitive to where in the prompt the
  representation is sampled. last_token weights the question end heavily;
  mean_pool spreads weight across system + role + content tokens; last_n
  preserves enough information to re-derive other pools offline (e.g.,
  excluding the final assistant marker, mean over question-only tokens, etc.)
  without re-running the model.

NaN handling
------------
  Any layer containing NaN or Inf values has those values replaced with 0
  before pooling, and the layer index is recorded. This mirrors the
  caa_nan_layers_sanitized field produced by the original pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Number of final-token positions to retain per layer (full vectors, not pooled).
# Chosen to comfortably cover chat-template assistant-marker tokens plus a few
# question-final content tokens for offline re-pooling experiments.
LAST_N = 8


def extract_layer_representations(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt_str: str,
    last_n: int = LAST_N,
) -> dict:
    """
    Run a single forward pass on prompt_str with output_hidden_states=True
    and extract three per-layer pooling variants.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The loaded causal LM (any quantisation; eval mode).
    tok : AutoTokenizer
        Matching tokenizer.
    prompt_str : str
        Chat-templated prompt string (already passed through
        apply_chat_template).
    last_n : int
        Number of trailing token positions to retain unpooled per layer.

    Returns
    -------
    dict with keys:
      last_token_hs    : (n_layers, hidden_dim) float16
                         h[L-1] per layer.
      mean_pool_hs     : (n_layers, hidden_dim) float16
                         mean over all L positions per layer.
      last_n_hs        : (n_layers, last_n, hidden_dim) float16
                         h[L-last_n : L] per layer; left-zero-padded if
                         L < last_n.
      prompt_token_len : int    — L (full chat-templated tokenisation)
      last_pos_idx     : int    — L - 1 (explicit for clarity)
      nan_layer_indices: list[int] — layer indices where NaN/Inf was
                                     detected and zeroed.

    Notes
    -----
    - n_layers = num_hidden_layers + 1; index 0 is the embedding layer,
      indices 1..num_hidden_layers are transformer block outputs.
    - use_cache=False to avoid retaining KV state we never reuse.
    - Tensors are moved to CPU and cast to float32 before pooling for
      stable arithmetic; final storage is float16 (sufficient for
      cosine geometry; consumers can up-cast offline).
    """
    inputs = tok(prompt_str, return_tensors="pt").to(model.device)
    L = int(inputs["input_ids"].shape[1])

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)

    hs_tuple = out.hidden_states  # length n_layers + 1, each (1, L, hidden_dim)
    n_layers = len(hs_tuple)
    hidden_dim = int(hs_tuple[0].shape[-1])

    last_token_hs = np.zeros((n_layers, hidden_dim), dtype=np.float16)
    mean_pool_hs = np.zeros((n_layers, hidden_dim), dtype=np.float16)
    last_n_hs = np.zeros((n_layers, last_n, hidden_dim), dtype=np.float16)
    n_keep = min(last_n, L)

    nan_layer_indices: list[int] = []
    for li, h in enumerate(hs_tuple):
        # float32 on CPU for stable NaN detection and pooling
        h_np = h[0].float().cpu().numpy()  # (L, hidden_dim)

        if np.isnan(h_np).any() or np.isinf(h_np).any():
            nan_layer_indices.append(li)
            h_np = np.nan_to_num(h_np, nan=0.0, posinf=0.0, neginf=0.0)

        last_token_hs[li] = h_np[-1].astype(np.float16)
        mean_pool_hs[li] = h_np.mean(axis=0).astype(np.float16)
        last_n_hs[li, -n_keep:] = h_np[-n_keep:].astype(np.float16)

    # Free intermediates before returning
    del out, hs_tuple, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "last_token_hs": last_token_hs,
        "mean_pool_hs": mean_pool_hs,
        "last_n_hs": last_n_hs,
        "prompt_token_len": L,
        "last_pos_idx": L - 1,
        "nan_layer_indices": nan_layer_indices,
    }
