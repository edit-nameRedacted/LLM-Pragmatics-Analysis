"""
qxc_collect_responses.py — Collect raw text responses for all 10 seeds.
=======================================================================
Stripped-down pipeline that removes all heavy processing (CAA, beam
divergence, EAS, NLI, SBERT, semantic entropy) and stores only the
generated response texts.

Outputs
-------
results/responses_{model_label}.json
    {prompt_id: {condition, domain, question, context, pairwise,
                 responses: [str × 10], seeds: [int × 10]}}

results/responses_{model_label}.csv
    Long-form: one row per response.
    Columns: prompt_id, condition, domain, question, context, pairwise,
             response_idx, response_text
    Useful for lexical diversity analysis and the listener (L(m|u)) study.

Usage (from llm_entropy_study/):
    python questions_x_context/qxc_collect_responses.py --qwen
    python questions_x_context/qxc_collect_responses.py --deepseek
    python questions_x_context/qxc_collect_responses.py --llama
    python questions_x_context/qxc_collect_responses.py --qwen --dry_run

Seeds use the same formula as the main inference scripts, so responses
are directly comparable to existing entropy measurements.

Dependencies
------------
    transformers, bitsandbytes, torch, pandas, tqdm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _setup_hf_auth() -> None:
    """Set HF_TOKEN from cache or environment."""
    from pathlib import Path as _Path
    cache_token_file = _Path.home() / ".cache" / "huggingface" / "token"
    if cache_token_file.exists():
        cached = cache_token_file.read_text().strip()
        if cached:
            os.environ["HF_TOKEN"] = cached
            print(f"[INFO] HF_TOKEN set from HF cache (last 4: ...{cached[-4:]})")
            return
    raw = os.environ.get("HF_TOKEN", "")
    clean = raw.strip().strip('"').strip("'")
    if clean:
        os.environ["HF_TOKEN"] = clean
        print(f"[INFO] HF_TOKEN from env var (last 4: ...{clean[-4:]})")
    else:
        print("[INFO] No HF credentials found.")


_setup_hf_auth()

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
_MODELS_DIR = Path(os.environ.get("QXC_MODELS_DIR", Path.home() / ".cache" / "huggingface" / "hub"))
_HERE       = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
PROMPTS_CSV = _ROOT / "data" / "human" / "prompts+SBERTsim_scores.csv"
RESULTS_DIR = _HERE / "results"

# Generation parameters — match original run_logit_entropy settings
MAX_NEW_TOKENS = 200
TEMPERATURE    = 0.8    # same stochastic sampling as original
TOP_P          = 0.95
N_SAMPLES      = 10     # reduced to 2 in dry_run

_MODEL_REGISTRY: dict[str, dict] = {
    "qwen": {
        "path": _MODELS_DIR / "Qwen2.5-7B-Instruct",
        "hub_id": "Qwen/Qwen2.5-7B-Instruct",
        "label": "qwen",
        "display": "Qwen2.5-7B-Instruct",
        "gated": False,
        "quantization": "8bit",
    },
    "llama": {
        "path": _MODELS_DIR / "Meta-Llama-3.1-8B-Instruct",
        "hub_id": "meta-llama/Llama-3.1-8B-Instruct",
        "label": "llama",
        "display": "Llama-3.1-8B-Instruct",
        "gated": True,
        "quantization": "8bit",
    },
    "mistral": {
        "path": _MODELS_DIR / "Mistral-7B-Instruct-v0.3",
        "hub_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "label": "mistral",
        "display": "Mistral-7B-Instruct-v0.3",
        "gated": False,
        "quantization": "8bit",
    },
    "deepseek": {
        "path": _MODELS_DIR / "deepseek-llm-7b-chat",
        "hub_id": "deepseek-ai/deepseek-llm-7b-chat",
        "label": "deepseek",
        "display": "DeepSeek-LLM-7B-Chat",
        "gated": False,
        "quantization": "8bit",
    },
    "deepseek_v2_lite": {
        "path": _MODELS_DIR / "DeepSeek-V2-Lite-Chat",
        "hub_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "label": "deepseek_v2_lite",
        "display": "DeepSeek-V2-Lite-Chat",
        "gated": False,
        "quantization": "4bit",
    },
}

_SIM_COLUMNS = (
    "context_question_similarity",
    "similarity_variance_flag",
    "q_tokens",
    "ctx_tokens",
    "combined_tokens",
    "token_limit_flag",
)


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect raw text responses (no heavy metrics)"
    )
    model_group = p.add_mutually_exclusive_group()
    model_group.add_argument("--qwen",            action="store_true")
    model_group.add_argument("--llama",           action="store_true")
    model_group.add_argument("--mistral",         action="store_true")
    model_group.add_argument("--deepseek",        action="store_true")
    model_group.add_argument("--deepseek_v2_lite",action="store_true")

    p.add_argument(
        "--dry_run", action="store_true",
        help="Run on 2 prompts with N=2 samples — smoke test only",
    )
    p.add_argument(
        "--run_id", type=int, default=0,
        help="Run ID for seed isolation across independent runs (default 0).",
    )
    p.add_argument(
        "--n_samples", type=int, default=N_SAMPLES,
        help=f"Number of responses to generate per prompt (default {N_SAMPLES}).",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_model_source(model_cfg: dict) -> tuple[str, bool]:
    """Resolve model source: local path → HF cache → live download."""
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
        print(f"  Local path incomplete — checking HF cache.")

    try:
        cached = snapshot_download(hub_id, local_files_only=True)
        if _is_complete(Path(cached)):
            return cached, True
    except (LocalEntryNotFoundError, Exception):
        pass

    print(f"  Downloading from Hub: {hub_id}")
    return hub_id, False


def load_model(model_cfg: dict) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load model without attn_implementation='eager' — we don't need hidden states."""
    source, is_local = _resolve_model_source(model_cfg)
    print(f"Model : {model_cfg['display']}  [{'local' if is_local else 'Hub'}]")

    quant = model_cfg.get("quantization", "8bit")
    if quant == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("Quant : 4-bit NF4")
    else:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        print("Quant : 8-bit")

    tok = AutoTokenizer.from_pretrained(source, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # No attn_implementation="eager" needed — we're not extracting hidden states
    model = AutoModelForCausalLM.from_pretrained(
        source,
        quantization_config=bnb_cfg,
        device_map="auto",
    )
    model.eval()

    if torch.cuda.is_available():
        vram_gb   = torch.cuda.memory_allocated() / 1e9
        total_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name  = torch.cuda.get_device_name(0)
        print(f"GPU   : {gpu_name} ({total_gb:.1f} GB)  VRAM used: {vram_gb:.2f} GB")

    return tok, model


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_prompts(dry_run: bool) -> pd.DataFrame:
    df = pd.read_csv(PROMPTS_CSV)
    df.columns = [c.strip() for c in df.columns]
    df["context"] = df["context"].fillna("").astype(str).str.strip().replace("nan", "")
    df["pairwise"] = df["pairwise"].astype(str).str.strip()
    df["condition"] = (
        df["condition"].astype(str).str.strip().str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    df = df.sort_values("prompt_id").reset_index(drop=True)
    if dry_run:
        # One NC + one DI prompt to verify output structure
        df = df.head(2)
    return df


def build_prompt(tok: AutoTokenizer, ctx: str, question: str) -> str:
    user_content = f"{ctx} {question}".strip() if ctx else question
    messages = [{"role": "user", "content": user_content}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_responses(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    n_samples: int,
    base_seed: int,
) -> tuple[list[str], list[int]]:
    """
    Generate n_samples responses with deterministic per-sample seeds.

    Seed formula: base_seed + sample_index
    Seed formula: base_seed + sample_index, matching the seed space used
    in the main inference scripts so responses align with entropy measurements.

    Returns
    -------
    responses : list[str]
        Decoded response texts, one per sample.
    seeds : list[int]
        The exact seed used for each sample, for reproducibility records.
    """
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = input_ids.shape[1]

    responses: list[str] = []
    seeds: list[int] = []

    for i in range(n_samples):
        sample_seed = base_seed + i
        set_seed(sample_seed)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tok.pad_token_id,
            )

        # Decode only the newly generated tokens (strip the prompt)
        new_tokens = output[0][prompt_len:]
        text = tok.decode(new_tokens, skip_special_tokens=True).strip()
        responses.append(text)
        seeds.append(sample_seed)

    return responses, seeds


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    model_key = (
        "llama"           if args.llama
        else "mistral"    if args.mistral
        else "deepseek"   if args.deepseek
        else "deepseek_v2_lite" if args.deepseek_v2_lite
        else "qwen"
    )
    model_cfg   = _MODEL_REGISTRY[model_key]
    model_label = model_cfg["label"]

    n_samples = 2 if args.dry_run else args.n_samples

    set_seed(SEED + args.run_id * 10000)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Response collection — lightweight generation pass")
    print(f"  model={model_cfg['display']}  |  n_samples={n_samples}  |"
          f"  run_id={args.run_id}  |  dry_run={args.dry_run}")
    print("=" * 60)

    print("\nLoading model...")
    tok, model = load_model(model_cfg)

    print("\nLoading prompts...")
    pilot_df = load_prompts(dry_run=args.dry_run)
    print(f"Prompts: {len(pilot_df)} rows")
    print()

    # ── Collection loop ───────────────────────────────────────────────────────
    results: dict[str, dict] = {}
    csv_rows: list[dict] = []

    for row in tqdm(pilot_df.itertuples(), total=len(pilot_df), desc="Prompts"):
        pid       = str(row.prompt_id)
        ctx       = str(row.context) if row.context else ""
        question  = str(row.question)
        condition = str(row.condition)
        domain    = str(row.domain)
        pairwise  = str(row.pairwise)

        # Seed formula: prompt_seed = SEED + run_id * 10000 + prompt_id * 100
        # per-sample seed = prompt_seed + sample_index
        prompt_seed = SEED + args.run_id * 10000 + int(row.prompt_id) * 100

        prompt = build_prompt(tok, ctx, question)

        t_start = time.perf_counter()
        tqdm.write(f"[{pid}] {condition:<28s} | generating {n_samples} responses...")

        responses, seeds = generate_responses(model, tok, prompt, n_samples, prompt_seed)

        elapsed = time.perf_counter() - t_start
        tqdm.write(f"[{pid}] done in {elapsed:.1f}s  "
                   f"| len(responses)={len(responses)}"
                   f"  | r[0][:60]: '{responses[0][:60]}'")

        # Pass-through sim columns
        sim_data = {
            col: (float(getattr(row, col))
                  if hasattr(row, col) and pd.notna(getattr(row, col))
                  else None)
            for col in _SIM_COLUMNS
        }

        # JSON record — one per prompt
        results[pid] = {
            "condition":   condition,
            "domain":      domain,
            "question":    question,
            "context":     ctx,
            "pairwise":    pairwise,
            **sim_data,
            "responses":   responses,   # list[str], length = n_samples
            "seeds":       seeds,        # list[int], same length
            "timing_s":    round(elapsed, 2),
        }

        # CSV rows — one per response (long form)
        for idx, (resp, seed) in enumerate(zip(responses, seeds)):
            csv_rows.append({
                "prompt_id":    pid,
                "condition":    condition,
                "domain":       domain,
                "question":     question,
                "context":      ctx,
                "pairwise":     pairwise,
                **sim_data,
                "response_idx": idx,
                "seed":         seed,
                "response_text": resp,
            })

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = RESULTS_DIR / f"responses_{model_label}.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nJSON saved  -> {out_json}")
    print(f"  Prompts: {len(results)}  |  Responses per prompt: {n_samples}")
    print(f"  Total responses: {len(results) * n_samples}")

    # ── Save CSV (long form) ──────────────────────────────────────────────────
    out_csv = RESULTS_DIR / f"responses_{model_label}.csv"
    csv_df  = pd.DataFrame(csv_rows)
    csv_df.to_csv(out_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"CSV saved   -> {out_csv}")
    print(f"  Rows: {len(csv_df)}  |  Columns: {list(csv_df.columns)}")

    # ── Quick sanity check ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Response length summary by condition:")
    csv_df["response_len"] = csv_df["response_text"].str.split().str.len()
    for cond in ["no_context", "direct_information",
                 "implicature_information", "stochastic_information"]:
        sub = csv_df[csv_df["condition"] == cond]["response_len"]
        if not sub.empty:
            print(f"  {cond:<30s}: mean={sub.mean():.1f}  "
                  f"std={sub.std():.1f}  "
                  f"min={sub.min()}  max={sub.max()}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
