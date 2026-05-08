"""
qxc_main.py — Questions × Context experiment: main pipeline.

Orchestrates model loading, prompt iteration, and output for the four-condition
information-theoretic pilot (no_context, stochastic_information,
implicature_information, direct_information).

Reads prompts_v5_sim.csv (output of sbert_similarity.py) and passes through the
pre-computed SBERT columns alongside the metrics computed here.

Usage (from llm_entropy_study/):
    python questions_x_context/qxc_main.py --qwen                  # Qwen2.5-7B-Instruct (default)
    python questions_x_context/qxc_main.py --llama                 # Llama 3.1 8B Instruct
    python questions_x_context/qxc_main.py --deepseek_v2_lite      # DeepSeek-V2-Lite-Chat
    python questions_x_context/qxc_main.py --qwen --dry_run        # smoke test, 2 prompts
    python questions_x_context/qxc_main.py --llama --run_id 1      # independent replication

Output files are suffixed with the model label (e.g. pilot_summary_qwen.csv) so
results from different models never overwrite each other.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

# Load .env file if python-dotenv is installed (optional convenience)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def _setup_hf_auth() -> None:
    """Set HF_TOKEN from the best available source with no network calls.

    Priority:
      1. ~/.cache/huggingface/token  (written by `hf auth login`) — always
         preferred because it is the most recently validated credential.
      2. HF_TOKEN env var — used only if the cache file is absent, after
         stripping stray quotes that PowerShell sometimes adds.

    We intentionally do NOT call login() or whoami() because both may make
    blocking network requests that hang on slow or firewalled connections.
    Setting os.environ["HF_TOKEN"] is enough — huggingface_hub and transformers
    both read this variable for every authenticated request.
    """
    from pathlib import Path as _Path

    cache_token_file = _Path.home() / ".cache" / "huggingface" / "token"

    # Prefer the cached token written by `hf auth login` — it is always the
    # most recently validated credential and avoids the stale-env-var problem.
    if cache_token_file.exists():
        cached = cache_token_file.read_text().strip()
        if cached:
            os.environ["HF_TOKEN"] = cached
            print(f"[INFO] HF_TOKEN set from ~/.cache/huggingface/token (last 4: ...{cached[-4:]})")
            return

    # Fallback: sanitise the env var (strip stray quotes from PowerShell)
    raw = os.environ.get("HF_TOKEN", "")
    clean = raw.strip().strip('"').strip("'")

    if raw and not clean:
        del os.environ["HF_TOKEN"]
        print("[INFO] HF_TOKEN empty after stripping — no credentials available.")
        print("       Run: huggingface-cli login")
    elif clean != raw:
        os.environ["HF_TOKEN"] = clean
        print(f"[INFO] HF_TOKEN sanitised from env var (last 4: ...{clean[-4:]})")
    elif clean:
        print(f"[INFO] HF_TOKEN from env var (last 4: ...{clean[-4:]})")
    else:
        print("[INFO] No HF credentials found. Gated models will fail.")
        print("       Run: huggingface-cli login")


_setup_hf_auth()

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from analysis_metrics import (
    _nanmean_or_none,
    compute_caa_for_question_group,
    compute_semantic_entropy,
    eas_shape_features,
    get_layerwise_hidden_states,
    run_beam_divergence,
    run_logit_entropy,
)

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False

# ── Paths & constants ──────────────────────────────────────────────────────────
SEED = 42
_MODELS_DIR = Path(os.environ.get("QXC_MODELS_DIR", Path.home() / ".cache" / "huggingface" / "hub"))
NLI_MODEL_ID = "microsoft/deberta-base-mnli"
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
PROMPTS_CSV = _ROOT / "data" / "human" / "prompts+SBERTsim_scores.csv"
RESULTS_DIR = _HERE / "results"

# Model registry — add entries here to support additional models.
# "label"        : suffix for all output filenames so runs never overwrite each other.
# "path"         : local directory to load from (preferred — no download needed).
# "hub_id"       : HuggingFace Hub model ID, used as fallback when the local path is
#                  absent or has an incomplete config.json.
# "gated"        : if True, HF_TOKEN is required and a clear error is raised if missing.
# "quantization" : "8bit" (default) or "4bit" (NF4). Use "4bit" for MoE models whose
#                  total parameter count exceeds available VRAM at 8-bit precision.
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
    # DeepSeek-V2-Lite is a 16B-parameter MoE model (only ~2.4B active per token,
    # but all expert weights must reside in VRAM). 8-bit would require ~16 GB;
    # 4-bit NF4 brings this to ~8 GB, fitting within a 12 GB card.
    "deepseek_v2_lite": {
        "path": _MODELS_DIR / "DeepSeek-V2-Lite-Chat",
        "hub_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "label": "deepseek_v2_lite",
        "display": "DeepSeek-V2-Lite-Chat",
        "gated": False,
        "quantization": "4bit",
    },
}

# Pre-computed columns written by sbert_similarity.py — passed through if present
_SIM_COLUMNS = (
    "context_question_similarity",
    "similarity_variance_flag",
    "q_tokens",
    "ctx_tokens",
    "combined_tokens",
    "token_limit_flag",
)


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM context-sensitivity pilot")

    # ── Model selection (mutually exclusive; default: qwen) ───────────────────
    model_group = p.add_mutually_exclusive_group()
    model_group.add_argument(
        "--qwen",
        action="store_true",
        default=False,
        help="Use Qwen2.5-7B-Instruct (default if neither flag is given)",
    )
    model_group.add_argument(
        "--llama",
        action="store_true",
        default=False,
        help="Use Llama 3.1 8B Instruct",
    )
    model_group.add_argument(
        "--mistral",
        action="store_true",
        default=False,
        help="Use Mistral-7B-Instruct-v0.3",
    )
    model_group.add_argument(
        "--deepseek",
        action="store_true",
        default=False,
        help="Use DeepSeek-LLM-7B-Chat",
    )
    model_group.add_argument(
        "--deepseek_v2_lite",
        action="store_true",
        default=False,
        help="Use DeepSeek-V2-Lite-Chat",
    )

    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Run on 2 prompts only with N=2 samples and num_beams=3",
    )
    p.add_argument(
        "--run_id",
        type=int,
        default=0,
        help=(
            "Integer run ID. Per-sample seed = SEED + run_id*10000 + prompt_id*100 "
            "+ sample_index, guaranteeing non-overlapping seed spaces across runs, "
            "prompts, and samples (assumes <100 prompts, <100 samples)."
        ),
    )
    p.add_argument(
        "--no_store_sequences",
        action="store_true",
        help=(
            "Do not write raw token_entropy_sequence arrays to pilot_results.json. "
            "EAS and extended_metrics.csv are still computed from the in-memory sequences. "
            "Use this flag when the full sequence data would make the JSON unmanageably large."
        ),
    )
    return p.parse_args()


# ── Model loading ──────────────────────────────────────────────────────────────
def _resolve_model_source(model_cfg: dict) -> tuple[str, bool]:
    """Return (source, is_local) for the given registry entry.

    Resolution order:
      1. Local path (e.g. models/Meta-Llama-3.1-8B-Instruct/) — used when a
         valid config.json is present there.
      2. HuggingFace cache — snapshot_download with local_files_only=True finds
         a previously downloaded snapshot without any network request. This is
         the common case when the model was downloaded via `hf auth login` /
         from_pretrained on a previous run.
      3. Live Hub download — only reached if the model is genuinely absent from
         both the local path and the HF cache.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    local_path: Path = model_cfg["path"]
    hub_id: str = model_cfg["hub_id"]

    def _is_complete(path: Path) -> bool:
        """True if a model directory has the minimum files needed to load."""
        if not (path / "config.json").exists():
            return False
        # Must have either a fast tokenizer or a SentencePiece vocab file
        has_tokenizer = (
            (path / "tokenizer.json").exists()
            or (path / "tokenizer.model").exists()
        )
        return has_tokenizer

    # 1. Explicit local directory — only use if it has a complete set of files
    if local_path.exists() and _is_complete(local_path):
        return str(local_path), True
    elif local_path.exists() and (local_path / "config.json").exists():
        print(f"  Local path {local_path} is incomplete (missing tokenizer files) — checking HF cache.")

    # 2. HF cache — only use if the cached snapshot is complete
    try:
        cached = snapshot_download(hub_id, local_files_only=True)
        cached_path = Path(cached)
        if _is_complete(cached_path):
            print(f"  Found complete snapshot in HF cache: {cached}")
            return cached, True
        else:
            print(f"  HF cache snapshot is incomplete (only {list(cached_path.iterdir())[:5]}...) — will download missing files.")
    except (LocalEntryNotFoundError, Exception):
        pass

    # 3. Download from Hub (downloads missing files; existing cache is reused)
    print(f"  Downloading/completing model from Hub: {hub_id}")
    print(f"  (This may take several minutes for an 8B model.)")
    return hub_id, False


def load_main_model(model_cfg: dict) -> tuple:
    """Load the selected model in 8-bit on GPU.

    Parameters
    ----------
    model_cfg : dict
        Entry from _MODEL_REGISTRY.

    Notes
    -----
    Loads from local path when a valid config.json is present there; falls back
    to the HuggingFace Hub otherwise (downloading on first use).

    Both models share the same loading logic:
      - 8-bit quantisation via bitsandbytes
      - attn_implementation="eager"  (required for output_hidden_states)
      - pad_token_id fallback to eos_token_id (Llama has no pad token by default)
      - apply_chat_template for prompt construction (both tokenizers have built-in templates)
    """
    source, is_local = _resolve_model_source(model_cfg)
    source_tag = "local" if is_local else "Hub"
    print(f"Model    : {model_cfg['display']}  [{source_tag}]  {source}")

    quant = model_cfg.get("quantization", "8bit")
    if quant == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"Quant    : 4-bit NF4 (double quant, compute fp16)")
    else:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        print(f"Quant    : 8-bit")

    # use_fast=True forces the fast tokenizer (tokenizer.json) path.
    # Llama 3.1 uses a tiktoken-based fast tokenizer and has no tokenizer.model
    # (SentencePiece) file, so AutoTokenizer must not try to load the slow
    # LlamaTokenizer fallback — which crashes with TypeError if the file is absent.
    tok = AutoTokenizer.from_pretrained(source, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

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


def load_nli_model() -> tuple:
    """Load DeBERTa-base-MNLI on GPU for accelerated inference."""
    nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
    nli_clf = (
        AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL_ID, 
            torch_dtype=torch.float32 # Keep float32 for classification accuracy
        )
        .to("cuda") # MOVE TO GPU
        .eval()
    )
    print(f"NLI model: {NLI_MODEL_ID} on CUDA")
    return nli_tok, nli_clf


def load_sbert_model():
    """Load Qwen3-Embedding-0.6B via SentenceTransformer on GPU."""
    if not _SBERT_AVAILABLE:
        print("[WARNING] sentence_transformers not installed.")
        return None
    # CHANGE device="cpu" to device="cuda"
    sbert = _SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")
    print("SBERT model: Qwen/Qwen3-Embedding-0.6B on CUDA")
    return sbert


# ── Data loading & subsetting ──────────────────────────────────────────────────
def load_pilot_prompts(dry_run: bool) -> pd.DataFrame:
    """
    Load all prompts from prompts_v5_sim.csv.

    Normalises column whitespace and condition strings.
    --dry_run takes the first 2 rows to smoke-test the pipeline.
    """
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


# ── Prompt construction ────────────────────────────────────────────────────────
def build_prompt(tok: AutoTokenizer, ctx: str, question: str) -> str:
    """
    Prepend context to question in the user turn of the chat template.
    Empty ctx -> user content = question only.
    """
    user_content = f"{ctx} {question}".strip() if ctx else question
    messages = [{"role": "user", "content": user_content}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Main pipeline ──────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── Resolve model ──────────────────────────────────────────────────────────
    model_key = (
        "llama" if args.llama
        else "mistral" if args.mistral
        else "deepseek" if args.deepseek
        else "deepseek_v2_lite" if args.deepseek_v2_lite
        else "qwen"
    )
    model_cfg = _MODEL_REGISTRY[model_key]
    model_label = model_cfg["label"]          # used to suffix all output files

    set_seed(SEED + args.run_id * 10000)
    RESULTS_DIR.mkdir(exist_ok=True)

    n_samples = 2 if args.dry_run else 10
    num_beams = 3 if args.dry_run else 10

    print("=" * 60)
    print("LLM context-sensitivity pilot experiment")
    print(f"  model={model_cfg['display']}  |  dry_run={args.dry_run}  |  "
          f"N={n_samples}  |  beams={num_beams}  |  run_id={args.run_id}")
    print("=" * 60)

    # ── Load models ────────────────────────────────────────────────────────────
    print("\nLoading main model (8-bit)...")
    tok, model = load_main_model(model_cfg)

    print("\nLoading NLI model (CPU)...")
    nli_tok, nli_clf = load_nli_model()

    print("\nLoading SBERT model (CPU)...")
    sbert_model = load_sbert_model()

    # ── Load & display pilot prompt set ───────────────────────────────────────
    print("\nLoading pilot prompts...")
    pilot_df = load_pilot_prompts(dry_run=args.dry_run)
    print(f"Pilot prompts: {len(pilot_df)} rows")

    present_sim_cols = [c for c in _SIM_COLUMNS if c in pilot_df.columns]
    missing_sim_cols = [c for c in _SIM_COLUMNS if c not in pilot_df.columns]
    if present_sim_cols:
        print(f"Pre-computed columns found : {present_sim_cols}")
    if missing_sim_cols:
        print(f"Pre-computed columns absent: {missing_sim_cols}")

    print()
    print(
        pilot_df[["prompt_id", "condition", "domain", "question", "context"]]
        .to_string(index=False)
    )
    print()

    # ── Processing loop ────────────────────────────────────────────────────────
    results: dict[str, dict] = {}
    results["_meta"] = {"run_id": args.run_id, "seed_base": SEED}

    hs_store: dict[str, list[np.ndarray]] = {}
    nan_layer_store: dict[str, list[int]] = {}
    q_to_cond_pid: dict[str, dict[str, str]] = defaultdict(dict)

    for row in tqdm(pilot_df.itertuples(), total=len(pilot_df), desc="Prompts"):
        pid = str(row.prompt_id)
        ctx = str(row.context) if row.context else ""
        question = str(row.question)
        condition = str(row.condition)
        domain = str(row.domain)
        pairwise = str(row.pairwise)

        # Read pre-computed sbert_similarity columns (None if column absent)
        sim_data: dict[str, float | int | None] = {
            col: (
                float(getattr(row, col))
                if hasattr(row, col) and pd.notna(getattr(row, col))
                else None
            )
            for col in _SIM_COLUMNS
        }

        prompt = build_prompt(tok, ctx, question)
        prompt_seed = SEED + args.run_id * 10000 + int(row.prompt_id) * 100
        t_start = time.perf_counter()

        # ── 1. Logit entropy ──────────────────────────────────────────────────
        tqdm.write(f"[{pid}] {condition:<28s} | logit entropy ({n_samples} samples)...")
        responses, logit_data = run_logit_entropy(
            model, tok, prompt, n_samples=n_samples, seed=prompt_seed,
            set_seed_fn=set_seed,
        )

        # ── 2. EAS + temporal shape features ─────────────────────────────────
        # Computed in-memory from token_entropy_sequence; written to
        # extended_metrics.csv only — pilot_summary.csv and pilot_results.json
        # are not modified.
        seqs = logit_data["token_entropy_sequence"]  # list[N] of list[T] floats

        # Per-sample shape feature dicts (empty dict for sequences that are too short)
        sample_shapes: list[dict] = [eas_shape_features(s) for s in seqs]

        # Scalar EAS per sample for backward-compatible eas_mean / eas_sd columns
        eas_per_sample = [sh.get("eas", 0.0) for sh in sample_shapes]
        logit_data["eas_mean"] = float(np.mean(eas_per_sample)) if eas_per_sample else 0.0
        logit_data["eas_sd"]   = float(np.std(eas_per_sample))  if eas_per_sample else 0.0
        logit_data["eas_per_sample"] = eas_per_sample

        # Aggregate temporal shape features across samples (mean; SD for two
        # reliability indicators: eas_skew and eas_sparsity).
        _shape_keys = [
            "eas_early", "eas_mid", "eas_late", "eas_final_quarter",
            "eas_slope", "eas_mean_rate", "eas_peak_rate",
            "eas_peak_rate_position", "eas_n_spikes", "eas_skew", "eas_sparsity",
        ]
        shape_agg: dict[str, float | None] = {}
        for k in _shape_keys:
            vals = [sh[k] for sh in sample_shapes if k in sh]
            shape_agg[k] = float(np.mean(vals)) if vals else None
        # Cross-sample SD for the two reliability indicators
        for k in ("eas_skew", "eas_sparsity"):
            vals = [sh[k] for sh in sample_shapes if k in sh]
            shape_agg[f"{k}_sd"] = float(np.std(vals)) if len(vals) > 1 else None

        # Cross-sample variance profile: var of H_t across samples at each position
        # (up to the shortest sequence length so all samples contribute at every t).
        min_len = min((len(s) for s in seqs), default=0)
        if min_len >= 2 and len(seqs) >= 2:
            arr = np.array([s[:min_len] for s in seqs], dtype=float)  # (N, T)
            cross_var = arr.var(axis=0).tolist()  # variance across samples per position
        else:
            cross_var = []

        logit_data["shape_agg"] = shape_agg
        logit_data["eas_cross_sample_variance_profile"] = cross_var

        # ── 3. Beam search divergence ─────────────────────────────────────────
        tqdm.write(f"[{pid}] {condition:<28s} | beam divergence ({num_beams} beams)...")
        beam_data = run_beam_divergence(
            model, tok, prompt, num_beams=num_beams,
            nli_tok=nli_tok, nli_clf=nli_clf, seed=prompt_seed,
            set_seed_fn=set_seed,
            sbert_model=sbert_model,
        )

        # ── 4. CAA forward pass ───────────────────────────────────────────────
        tqdm.write(f"[{pid}] {condition:<28s} | CAA forward pass...")
        with torch.no_grad():
            layer_vecs, nan_layers = get_layerwise_hidden_states(model, tok, prompt)
        hs_store[pid] = layer_vecs
        nan_layer_store[pid] = nan_layers

        # ── 5. Semantic entropy ───────────────────────────────────────────────
        tqdm.write(f"[{pid}] {condition:<28s} | semantic entropy...")
        sem_H = compute_semantic_entropy(responses, nli_tok, nli_clf)

        elapsed = time.perf_counter() - t_start

        results[pid] = {
            "condition": condition,
            "domain": domain,
            "question": question,
            "context": ctx,
            "pairwise": pairwise,
            **sim_data,
            "logit_entropy": logit_data,
            "beam_divergence": beam_data,
            "caa": None,
            "semantic_entropy": sem_H,
            "timing": elapsed,
        }
        q_to_cond_pid[question][condition] = pid

        tqdm.write(
            f"[{pid}] done in {elapsed:.1f}s "
            f"| mean_H={logit_data['mean_token_entropy']:.3f} "
            f"| sem_H={sem_H:.3f}"
        )

    # ── CAA: compute per-layer distances ──────────────────────────────────────
    print("\nComputing CAA hidden-state displacements...")
    for question, cond_to_pid in q_to_cond_pid.items():
        hs_by_cond = {
            cond: hs_store[pid]
            for cond, pid in cond_to_pid.items()
            if pid in hs_store
        }
        caa_results = compute_caa_for_question_group(hs_by_cond)
        for cond, caa_dict in caa_results.items():
            if cond in cond_to_pid:
                pid_for_cond = cond_to_pid[cond]
                caa_dict["nan_layers_sanitized"] = nan_layer_store.get(pid_for_cond, [])
                results[pid_for_cond]["caa"] = caa_dict
        print(f"  Q: \"{question[:60]}\" — {len(caa_results)} conditions processed")

    # ── Save extended_metrics.csv ─────────────────────────────────────────────
    # New measurements only — does not modify pilot_summary.csv or pilot_results.json.
    ext_rows: list[dict] = []
    for pid, r in results.items():
        if pid == "_meta":
            continue
        le = r["logit_entropy"]
        bd = r["beam_divergence"]

        # Serialise list columns as semicolon-separated strings (flat CSV, no JSON parser needed)
        eas_str = ";".join(f"{v:.6f}" for v in le.get("eas_per_sample", []))
        seq_cols: dict = {}
        for i, seq in enumerate(le.get("token_entropy_sequence", [])):
            seq_cols[f"token_entropy_sequence_sample_{i}"] = ";".join(f"{v:.6f}" for v in seq)

        sa = le.get("shape_agg", {}) or {}
        cross_var = le.get("eas_cross_sample_variance_profile", [])
        cross_var_str = ";".join(f"{v:.8f}" for v in cross_var)

        caa = r.get("caa") or {}
        l2_vals  = caa.get("per_layer_l2_vs_no_context") or []
        cos_vals = caa.get("per_layer_cosine_vs_no_context") or []
        caa_l2_str  = ";".join(f"{v:.6f}" for v in l2_vals)
        caa_cos_str = ";".join(f"{v:.6f}" for v in cos_vals)

        ext_rows.append({
            # ── identity ──────────────────────────────────────────────────────
            "prompt_id": pid,
            "condition": r["condition"],
            "domain": r["domain"],
            "pairwise": r["pairwise"],
            # ── EAS scalar (backward-compatible) ──────────────────────────────
            "eas_mean": le.get("eas_mean"),
            "eas_sd": le.get("eas_sd"),
            "eas_per_sample": eas_str,
            # ── EAS temporal shape features (aggregated across samples) ───────
            "eas_early": sa.get("eas_early"),
            "eas_mid": sa.get("eas_mid"),
            "eas_late": sa.get("eas_late"),
            "eas_final_quarter": sa.get("eas_final_quarter"),
            "eas_slope": sa.get("eas_slope"),
            "eas_mean_rate": sa.get("eas_mean_rate"),
            "eas_peak_rate": sa.get("eas_peak_rate"),
            "eas_peak_rate_position": sa.get("eas_peak_rate_position"),
            "eas_n_spikes": sa.get("eas_n_spikes"),
            "eas_skew": sa.get("eas_skew"),
            "eas_skew_sd": sa.get("eas_skew_sd"),
            "eas_sparsity": sa.get("eas_sparsity"),
            "eas_sparsity_sd": sa.get("eas_sparsity_sd"),
            "eas_cross_sample_variance_profile": cross_var_str,
            # ── raw token entropy sequences ───────────────────────────────────
            **seq_cols,
            # ── CAA hidden-state displacement ─────────────────────────────────
            "caa_mean_l2": _nanmean_or_none(l2_vals),
            "caa_mean_cosine": _nanmean_or_none(cos_vals),
            "caa_per_layer_l2": caa_l2_str,
            "caa_per_layer_cosine": caa_cos_str,
            "displacement_cosine_vs_direct": caa.get("displacement_cosine_vs_direct"),
            "caa_nan_layers_sanitized": ";".join(
                str(x) for x in caa.get("nan_layers_sanitized", [])
            ),

            # ── beam divergence ───────────────────────────────────────────────
            "beam_score_entropy": bd.get("beam_score_entropy"),
            "beam_score_gap": bd.get("beam_score_gap"),
            "beam_score_raw_std": bd.get("beam_score_raw_std"),
            "beam_score_raw_range": bd.get("beam_score_raw_range"),
            "beam_first_divergence_position": bd.get("beam_first_divergence_position"),
            "beam_length_mean": bd.get("beam_length_mean"),
            "beam_length_sd": bd.get("beam_length_sd"),
            "beam_sbert_cosine": bd.get("beam_sbert_cosine"),
            # Per-layer hidden-state pairwise cosine across beams (trajectory)
            "beam_per_layer_cosine": ";".join(
                f"{v:.6f}" for v in bd.get("beam_per_layer_cosine", []) or []
            ),
        })

    ext_df = pd.DataFrame(ext_rows)
    ext_csv = RESULTS_DIR / f"extended_metrics_{model_label}.csv"
    ext_df.to_csv(ext_csv, index=False)
    print(f"Extended metrics saved -> {ext_csv}")

    # ── Correlation check: logit entropy vs semantic entropy ──────────────────
    print("\n" + "=" * 60)
    print("Pilot correlation: mean token logit entropy vs semantic entropy")
    logit_vals = [
        results[p]["logit_entropy"]["mean_token_entropy"]
        for p in results
        if p != "_meta" and results[p]["logit_entropy"]["per_sample"]
    ]
    sem_vals = [
        results[p]["semantic_entropy"]
        for p in results
        if p != "_meta" and results[p]["logit_entropy"]["per_sample"]
    ]
    if len(logit_vals) >= 2:
        r, p_val = pearsonr(logit_vals, sem_vals)
        print(f"  Pearson r = {r:.4f}  (p = {p_val:.4f}  n = {len(logit_vals)})")
        print(
            "  Interpretation: "
            + (
                "high correlation -> semantic entropy may be redundant with logit entropy"
                if abs(r) > 0.7
                else "low correlation -> both measures capture distinct signal"
            )
        )
    else:
        print("  Not enough data for correlation (need >= 2 prompts).")

    # ── Save pilot_results.json ────────────────────────────────────────────────
    # Optionally strip raw token_entropy_sequence before serialising to keep the
    # file size manageable. EAS and extended_metrics.csv are already written above.
    if args.no_store_sequences:
        for pid, r in results.items():
            if pid != "_meta":
                r["logit_entropy"].pop("token_entropy_sequence", None)

    out_json = RESULTS_DIR / f"pilot_results_{model_label}.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved -> {out_json}")

    # ── Save pilot_summary.csv ─────────────────────────────────────────────────
    csv_rows: list[dict] = []
    for pid, r in results.items():
        if pid == "_meta":
            continue
        caa = r.get("caa") or {}
        l2_vals = caa.get("per_layer_l2_vs_no_context") or []
        cos_vals = caa.get("per_layer_cosine_vs_no_context") or []
        csv_rows.append(
            {
                "prompt_id": pid,
                "condition": r["condition"],
                "domain": r["domain"],
                "question": r["question"][:70],
                "context": r["context"][:60],
                "pairwise": r["pairwise"],
                # Pre-computed sbert_similarity columns
                "context_question_similarity": r.get("context_question_similarity"),
                "similarity_variance_flag": r.get("similarity_variance_flag"),
                "q_tokens": r.get("q_tokens"),
                "ctx_tokens": r.get("ctx_tokens"),
                "combined_tokens": r.get("combined_tokens"),
                "token_limit_flag": r.get("token_limit_flag"),
                # Computed metrics
                "mean_token_entropy": r["logit_entropy"]["mean_token_entropy"],
                "first_token_entropy": r["logit_entropy"]["first_token_entropy"],
                "peak_entropy_position": r["logit_entropy"]["peak_entropy_position"],
                "beam_mean_cosine_sim": r["beam_divergence"]["mean_pairwise_cosine_similarity"],
                "beam_cluster_entropy": r["beam_divergence"]["semantic_cluster_entropy"],
                "beam_embed_nan_positions": r["beam_divergence"].get(
                    "embed_nan_positions_sanitized", None
                ),
                "caa_mean_l2": _nanmean_or_none(l2_vals),
                "caa_mean_cosine": _nanmean_or_none(cos_vals),
                "displacement_cosine_vs_direct": caa.get("displacement_cosine_vs_direct"),
                "nan_layers_sanitized": ";".join(
                    str(x) for x in caa.get("nan_layers_sanitized", [])
                ),
                "semantic_entropy": r["semantic_entropy"],
                "timing_s": round(r["timing"], 2),
            }
        )

    summary_df = pd.DataFrame(csv_rows)
    out_csv = RESULTS_DIR / f"pilot_summary_{model_label}.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"Summary saved -> {out_csv}")

    # ── Per-condition summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Per-condition mean token entropy:")
    for cond in ("no_context", "stochastic_information", "implicature_information", "direct_information"):
        rows = summary_df[summary_df["condition"] == cond]
        if not rows.empty:
            print(f"  {cond:<28s}: {rows['mean_token_entropy'].mean():.4f} nats")
    print("=" * 60)

    # ── Between-sample variance check ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Between-sample variance of mean_token_entropy per condition:")
    RELIABILITY_THRESHOLD = 0.05
    reliability_warnings: list[str] = []
    for cond in ("no_context", "stochastic_information", "implicature_information", "direct_information"):
        per_sample_vals = [
            results[p]["logit_entropy"]["per_sample"]
            for p in results
            if p != "_meta"
            and results[p]["condition"] == cond
            and results[p]["logit_entropy"]["per_sample"]
        ]
        if not per_sample_vals:
            continue
        cond_variances = [float(np.var(ps)) for ps in per_sample_vals]
        cond_stds = [float(np.std(ps)) for ps in per_sample_vals]
        print(f"  {cond:<28s}: between-sample var = {float(np.mean(cond_variances)):.6f}")
        pids_for_cond = [
            p for p in results
            if p != "_meta"
            and results[p]["condition"] == cond
            and results[p]["logit_entropy"]["per_sample"]
        ]
        for pid, ps, std in zip(pids_for_cond, per_sample_vals, cond_stds):
            if std > RELIABILITY_THRESHOLD:
                msg = (
                    f"  [RELIABILITY WARNING] prompt_id={pid} ({cond}): "
                    f"per-sample std={std:.4f} nats > {RELIABILITY_THRESHOLD} threshold"
                )
                reliability_warnings.append(msg)
                print(msg)
    if not reliability_warnings:
        print("  No reliability warnings (all per-sample std <= 0.05 nats).")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
