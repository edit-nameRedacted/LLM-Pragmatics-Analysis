"""
qxc_main_RSA_base.py — RSA hidden-state collection for BASE (non-instruct) models.

Identical pipeline to qxc_main_RSA.py but strips all chat-template logic.
Prompts are built by plain string concatenation:  "<context> <question>"
(or just "<question>" when context is empty).

Two models supported:
    --qwen    Qwen2.5-7B          (base, 8-bit)
    --llama   Meta-Llama-3.1-8B   (base, 8-bit, gated)

Outputs (into RESULTS_DIR, labelled with *_base suffix)
-------------------------------------------------------
    hidden_states_{model_label}.npz
        prompt_ids        : (N,) int32
        last_token_hs     : (N, n_layers, hidden_dim) float16
        mean_pool_hs      : (N, n_layers, hidden_dim) float16
        last_n_hs         : (N, n_layers, LAST_N, hidden_dim) float16
        prompt_token_lens : (N,) int32
        nan_mask          : (N, n_layers) bool
        layer_indices     : (n_layers,) int32

    hidden_states_meta_{model_label}.csv
        prompt_id, question_id, condition, domain, pairwise, question,
        context, prompt_token_len, n_nan_layers, timing_s

Usage (from llm_entropy_study/)
-------------------------------
    python questions_x_context/qxc_main_RSA_base.py --qwen
    python questions_x_context/qxc_main_RSA_base.py --llama
    python questions_x_context/qxc_main_RSA_base.py --qwen --dry_run
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── HF auth ───────────────────────────────────────────────────────────────────
def _setup_hf_auth() -> None:
    cache_token_file = Path.home() / ".cache" / "huggingface" / "token"
    if cache_token_file.exists():
        cached = cache_token_file.read_text().strip()
        if cached:
            os.environ["HF_TOKEN"] = cached
            print(f"[INFO] HF_TOKEN set from ~/.cache/huggingface/token (last 4: ...{cached[-4:]})")
            return
    raw = os.environ.get("HF_TOKEN", "")
    clean = raw.strip().strip('"').strip("'")
    if raw and not clean:
        del os.environ["HF_TOKEN"]
        print("[INFO] HF_TOKEN empty after stripping — gated models will fail.")
    elif clean != raw:
        os.environ["HF_TOKEN"] = clean
        print(f"[INFO] HF_TOKEN sanitised from env var (last 4: ...{clean[-4:]})")
    elif clean:
        print(f"[INFO] HF_TOKEN from env var (last 4: ...{clean[-4:]})")
    else:
        print("[INFO] No HF credentials found. Gated models will fail.")


_setup_hf_auth()

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from analysis_metrics_RSA import LAST_N, extract_layer_representations


# ── Paths & constants ─────────────────────────────────────────────────────────
SEED = 42
_MODELS_DIR = Path(r"C:/Users/Watcher/Documents/models")
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
PROMPTS_CSV = _ROOT / "data" / "human" / "prompts+SBERTsim_scores.csv"
RESULTS_DIR = _HERE / "results"


# ── Model registry — base models only ─────────────────────────────────────────
_MODEL_REGISTRY: dict[str, dict] = {
    "qwen_base": {
        "path": _MODELS_DIR / "Qwen2.5-7B",
        "hub_id": "Qwen/Qwen2.5-7B",
        "label": "qwen_base",
        "display": "Qwen2.5-7B (base)",
        "gated": False,
        "quantization": "8bit",
    },
    "llama_base": {
        "path": _MODELS_DIR / "Meta-Llama-3.1-8B",
        "hub_id": "meta-llama/Llama-3.1-8B",
        "label": "llama_base",
        "display": "Llama-3.1-8B (base)",
        "gated": True,
        "quantization": "8bit",
    },
}


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RSA hidden-state collection for base (non-instruct) models"
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--qwen", action="store_true", default=False,
                   help="Run Qwen2.5-7B base")
    g.add_argument("--llama", action="store_true", default=False,
                   help="Run Llama-3.1-8B base (gated — needs HF token)")
    p.add_argument("--dry_run", action="store_true",
                   help="Run on the first 2 prompts only — smoke test.")
    p.add_argument("--run_id", type=int, default=0,
                   help="Integer run ID for non-overlapping seed spaces across runs.")
    return p.parse_args()


# ── Prompt construction — plain concatenation, no chat template ───────────────
def build_prompt(ctx: str, question: str) -> str:
    """Concatenate context and question with a single space separator.

    No BOS/EOS injection here — the tokenizer adds BOS automatically via
    add_special_tokens=True (the default).  Keeping the surface form plain
    means the model sees the same token distribution it was trained on.
    """
    return f"{ctx} {question}".strip() if ctx else question.strip()


# ── Model loading ─────────────────────────────────────────────────────────────
def _resolve_model_source(model_cfg: dict) -> tuple[str, bool]:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    local_path: Path = model_cfg["path"]
    hub_id: str = model_cfg["hub_id"]

    def _is_complete(path: Path) -> bool:
        if not (path / "config.json").exists():
            return False
        return (path / "tokenizer.json").exists() or (path / "tokenizer.model").exists()

    if local_path.exists() and _is_complete(local_path):
        return str(local_path), True
    elif local_path.exists() and (local_path / "config.json").exists():
        print(f"  Local path {local_path} is incomplete — checking HF cache.")

    try:
        cached = snapshot_download(hub_id, local_files_only=True)
        if _is_complete(Path(cached)):
            print(f"  Found complete snapshot in HF cache: {cached}")
            return cached, True
    except (LocalEntryNotFoundError, Exception):
        pass

    print(f"  Downloading/completing model from Hub: {hub_id}")
    return hub_id, False


def load_base_model(model_cfg: dict) -> tuple:
    """Load a base (non-instruct) causal LM in 8-bit on GPU."""
    source, is_local = _resolve_model_source(model_cfg)
    source_tag = "local" if is_local else "Hub"
    print(f"Model    : {model_cfg['display']}  [{source_tag}]  {source}")

    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    print("Quant    : 8-bit")

    tok = AutoTokenizer.from_pretrained(source, use_fast=True)

    # Base models (especially LLaMA) often ship without a pad token.
    # Set it to eos_token so batched calls don't error; RSA uses single
    # sequences so padding is never actually applied here.
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
        print(f"[INFO] pad_token_id set to eos_token_id ({tok.eos_token_id})")

    model = AutoModelForCausalLM.from_pretrained(
        source,
        quantization_config=bnb_cfg,
        device_map="auto",
        attn_implementation="eager",  # required for output_hidden_states
    )
    model.eval()

    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU      : {gpu_name} ({total_gb:.1f} GB total)")
        print(f"VRAM used: {vram_gb:.2f} GB after model load")
    return tok, model


# ── Data loading ──────────────────────────────────────────────────────────────
def load_pilot_prompts(dry_run: bool) -> pd.DataFrame:
    df = pd.read_csv(PROMPTS_CSV)
    df.columns = [c.strip() for c in df.columns]
    df["context"] = (
        df["context"].fillna("").astype(str).str.strip().replace("nan", "")
    )
    df["pairwise"] = df["pairwise"].astype(str).str.strip()
    df["condition"] = (
        df["condition"].astype(str).str.strip().str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    df = df.sort_values("prompt_id").reset_index(drop=True)
    if dry_run:
        df = df.head(2)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    if args.llama:
        model_key = "llama_base"
    else:
        model_key = "qwen_base"  # default

    model_cfg = _MODEL_REGISTRY[model_key]
    model_label = model_cfg["label"]

    set_seed(SEED + args.run_id * 10000)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("RSA hidden-state collection  [BASE MODELS — no chat template]")
    print(f"  model={model_cfg['display']}  |  dry_run={args.dry_run}  |  "
          f"run_id={args.run_id}")
    print(f"  pooling: last_token + mean_pool + last_{LAST_N}")
    print("=" * 60)

    print("\nLoading model...")
    tok, model = load_base_model(model_cfg)

    print("\nLoading prompts...")
    df = load_pilot_prompts(dry_run=args.dry_run)
    N = len(df)
    print(f"Loaded {N} prompts")

    # ── Probe pass to discover (n_layers, hidden_dim) ────────────────────────
    print("\nProbing model dimensions (first prompt)...")
    probe_row = df.iloc[0]
    probe_prompt = build_prompt(str(probe_row["context"]), str(probe_row["question"]))
    t0 = time.perf_counter()
    probe_out = extract_layer_representations(model, tok, probe_prompt)
    probe_time = time.perf_counter() - t0
    n_layers, hidden_dim = probe_out["last_token_hs"].shape
    print(f"  n_layers={n_layers} (incl. embedding)  hidden_dim={hidden_dim}")
    print(f"  probe forward pass: {probe_time:.2f}s")

    # ── Pre-allocate accumulator arrays ───────────────────────────────────────
    last_token_hs = np.zeros((N, n_layers, hidden_dim), dtype=np.float16)
    mean_pool_hs = np.zeros((N, n_layers, hidden_dim), dtype=np.float16)
    last_n_hs = np.zeros((N, n_layers, LAST_N, hidden_dim), dtype=np.float16)
    prompt_token_lens = np.zeros(N, dtype=np.int32)
    nan_mask = np.zeros((N, n_layers), dtype=bool)
    timings = np.zeros(N, dtype=np.float32)

    # Slot probe result
    last_token_hs[0] = probe_out["last_token_hs"]
    mean_pool_hs[0] = probe_out["mean_pool_hs"]
    last_n_hs[0] = probe_out["last_n_hs"]
    prompt_token_lens[0] = probe_out["prompt_token_len"]
    timings[0] = probe_time
    for li in probe_out["nan_layer_indices"]:
        nan_mask[0, li] = True

    # ── Process remaining prompts ─────────────────────────────────────────────
    for i in tqdm(range(1, N), desc="Forward passes"):
        row = df.iloc[i]
        ctx = str(row["context"]) if row["context"] else ""
        question = str(row["question"])
        prompt_str = build_prompt(ctx, question)

        t0 = time.perf_counter()
        out = extract_layer_representations(model, tok, prompt_str)
        timings[i] = time.perf_counter() - t0

        last_token_hs[i] = out["last_token_hs"]
        mean_pool_hs[i] = out["mean_pool_hs"]
        last_n_hs[i] = out["last_n_hs"]
        prompt_token_lens[i] = out["prompt_token_len"]
        for li in out["nan_layer_indices"]:
            nan_mask[i, li] = True

    # ── Save numerical arrays ─────────────────────────────────────────────────
    npz_path = RESULTS_DIR / f"hidden_states_{model_label}.npz"
    np.savez_compressed(
        npz_path,
        prompt_ids=df["prompt_id"].astype(np.int32).values,
        last_token_hs=last_token_hs,
        mean_pool_hs=mean_pool_hs,
        last_n_hs=last_n_hs,
        prompt_token_lens=prompt_token_lens,
        nan_mask=nan_mask,
        layer_indices=np.arange(n_layers, dtype=np.int32),
    )
    print(f"\nHidden states saved -> {npz_path}")
    print(f"  Compressed size   : {npz_path.stat().st_size / 1e6:.1f} MB")

    # ── Save metadata CSV ─────────────────────────────────────────────────────
    df["question_id"] = (df["prompt_id"] - 1) // 4 + 1
    meta = pd.DataFrame({
        "prompt_id": df["prompt_id"].values,
        "question_id": df["question_id"].values,
        "condition": df["condition"].values,
        "domain": df["domain"].values,
        "pairwise": df["pairwise"].values,
        "question": df["question"].values,
        "context": df["context"].values,
        "prompt_token_len": prompt_token_lens,
        "n_nan_layers": nan_mask.sum(axis=1),
        "timing_s": timings,
    })
    meta_path = RESULTS_DIR / f"hidden_states_meta_{model_label}.csv"
    meta.to_csv(meta_path, index=False)
    print(f"Metadata saved      -> {meta_path}")

    # ── Sanity report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Prompts processed   : {N}")
    print(f"  Mean per-prompt time: {timings.mean():.2f}s "
          f"(probe: {timings[0]:.2f}s)")
    print(f"  Total wall-clock    : {timings.sum():.1f}s")
    print(f"  Layers with any NaN : "
          f"{int((nan_mask.sum(axis=0) > 0).sum())} / {n_layers}")
    if nan_mask.any():
        bad_layers = np.where(nan_mask.sum(axis=0) > 0)[0]
        print(f"  NaN-affected layer indices: {bad_layers.tolist()[:15]}"
              + (" ..." if len(bad_layers) > 15 else ""))
        prompts_with_any_nan = int((nan_mask.sum(axis=1) > 0).sum())
        print(f"  Prompts with any NaN layer: {prompts_with_any_nan} / {N}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
