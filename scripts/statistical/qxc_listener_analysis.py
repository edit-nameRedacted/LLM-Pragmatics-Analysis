"""
qxc_listener_analysis.py — Listener study and lexical diversity analysis
=========================================================================
Two analyses in one script, designed to run after qxc_collect_responses.py.

Analysis 1: Listener study — L(m|u) and V_L(m,u)
--------------------------------------------------
Uses DeepSeek-LLM-7B-Chat as a fixed listener to estimate
P(condition | response, question). This operationalises the RSA utility
V_L(m,u) = log L(m|u) in the G_α framework (Zaslavsky, Hu & Levy, 2020).

For each speaker response r, the listener receives:
    - The question q
    - The response r (context hidden)
    - The four possible contexts in randomised order
    - A request to assign a probability distribution over conditions

Outputs:
    results/listener_results_{speaker_model}.csv
        One row per response. Columns: prompt_id, condition, response_idx,
        P_NC, P_DI, P_II, P_SI, log_P_correct, V_L, parse_failed
    results/listener_summary_{speaker_model}.csv
        One row per prompt (averaged over 10 responses).

Analysis 2: Lexical diversity
------------------------------
Tests whether DI context reduces within-condition lexical diversity
(convergence on domain register) while SI produces the broadest scatter.

Measures per prompt x condition:
    type_token_ratio     -- unique_tokens / total_tokens
    mean_response_len    -- mean word count across samples
    unique_bigram_ratio  -- unique bigrams / total bigrams
    mean_pairwise_jaccard -- mean Jaccard similarity of response pairs
    distinct_openings_4w  -- number of distinct 4-word opening phrases

Outputs:
    results/lexical_diversity_{speaker_model}.csv
        One row per prompt x condition.

Analysis 3: Alpha estimation
-----------------------------
After running the listener study, merges V_L with DEAS from
extended_metrics_{speaker}.csv to estimate alpha-hat in:
    G_alpha[S,L] = H_S(U|M) + alpha * E_S[V_L]
alpha-hat = -r(DEAS, V_L). alpha-hat > 0 is consistent with G_alpha optimisation.

Usage
-----
    python qxc_listener_analysis.py --speaker deepseek
    python qxc_listener_analysis.py --speaker qwen
    python qxc_listener_analysis.py --speaker llama
    python qxc_listener_analysis.py --speaker deepseek --dry_run
    python qxc_listener_analysis.py --speaker deepseek --skip_listener
    python qxc_listener_analysis.py --compare   # after all speakers run

Dependencies
------------
    pip install transformers bitsandbytes torch pandas numpy scipy tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
_HERE       = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / "results"
MODELS_DIR  = Path(r"C:/Users/Watcher/Documents/models")

CONDITIONS = [
    "no_context",
    "direct_information",
    "implicature_information",
    "stochastic_information",
]

COND_LABELS = {
    "no_context":              "A",
    "direct_information":      "B",
    "implicature_information": "C",
    "stochastic_information":  "D",
}

# Listener model -- fixed across all speaker evaluations for comparability
LISTENER_MODEL_PATH = MODELS_DIR / "deepseek-llm-7b-chat"
LISTENER_HUB_ID     = "deepseek-ai/deepseek-llm-7b-chat"
LISTENER_DISPLAY    = "DeepSeek-LLM-7B-Chat"

# Generation parameters for listener responses
LISTENER_MAX_NEW_TOKENS = 60   # only needs to output a short JSON object
# Greedy decoding (do_sample=False) — temperature and top_p are not used


# -----------------------------------------------------------------------------
# HF AUTH -- identical to qxc_collect_responses.py
# -----------------------------------------------------------------------------
def _setup_hf_auth() -> None:
    cache_token_file = Path.home() / ".cache" / "huggingface" / "token"
    if cache_token_file.exists():
        cached = cache_token_file.read_text().strip()
        if cached:
            os.environ["HF_TOKEN"] = cached
            print(f"[INFO] HF_TOKEN set from HF cache (last 4: ...{cached[-4:]})")
            return
    raw   = os.environ.get("HF_TOKEN", "")
    clean = raw.strip().strip('"').strip("'")
    if clean:
        os.environ["HF_TOKEN"] = clean
        print(f"[INFO] HF_TOKEN from env var (last 4: ...{clean[-4:]})")
    else:
        print("[INFO] No HF credentials found.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Listener study + lexical diversity")
    p.add_argument("--speaker", type=str, default=None,
                   help="Speaker model label (e.g. deepseek, qwen, llama). "
                        "Loads results/responses_{speaker}.csv.")
    p.add_argument("--dry_run", action="store_true",
                   help="Process first 2 unique questions only.")
    p.add_argument("--skip_listener", action="store_true",
                   help="Skip listener inference; run lexical analysis only.")
    p.add_argument("--compare", action="store_true",
                   help="Load all listener_results_*.csv and compare V_L "
                        "across speaker models. Does not require --speaker.")
    return p.parse_args()


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
def load_responses(speaker: str) -> pd.DataFrame:
    """Load speaker responses from results/responses_{speaker}.csv."""
    path = RESULTS_DIR / f"responses_{speaker}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"responses_{speaker}.csv not found in {RESULTS_DIR}. "
            f"Run qxc_collect_responses.py --{speaker} first."
        )
    df = pd.read_csv(path)
    df["context"]       = df["context"].fillna("").astype(str).str.strip()
    df["response_text"] = df["response_text"].fillna("").astype(str).str.strip()
    return df


def build_context_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a question x condition -> context lookup directly from the responses
    dataframe. The responses CSV already contains question, condition, and context
    columns, so there is no need to re-read the original prompts file.

    Returns a DataFrame with columns: question, condition, context.
    One row per unique (question, condition) pair.
    """
    return (
        df[["question", "condition", "context"]]
        .drop_duplicates(subset=["question", "condition"])
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------
# LISTENER MODEL LOADING
# -----------------------------------------------------------------------------
def _is_complete(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    return (path / "tokenizer.json").exists() or (path / "tokenizer.model").exists()


def load_listener_model() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load DeepSeek-LLM-7B-Chat as the listener in 8-bit quantisation.
    Resolution order: local path -> HF cache -> Hub download.
    """
    if LISTENER_MODEL_PATH.exists() and _is_complete(LISTENER_MODEL_PATH):
        source = str(LISTENER_MODEL_PATH)
        print(f"Listener: {LISTENER_DISPLAY}  [local]")
    else:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError
        try:
            cached = snapshot_download(LISTENER_HUB_ID, local_files_only=True)
            source = cached
            print(f"Listener: {LISTENER_DISPLAY}  [HF cache]")
        except (LocalEntryNotFoundError, Exception):
            source = LISTENER_HUB_ID
            print(f"Listener: {LISTENER_DISPLAY}  [Hub download]")

    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

    tok = AutoTokenizer.from_pretrained(source, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        source,
        quantization_config=bnb_cfg,
        device_map="auto",
    )
    model.eval()

    # Override the model's generation_config to suppress spurious
    # temperature/top_p warnings when do_sample=False
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    if torch.cuda.is_available():
        vram_gb  = torch.cuda.memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB)  "
              f"VRAM used: {vram_gb:.2f} GB")

    return tok, model


# -----------------------------------------------------------------------------
# LISTENER PROMPT CONSTRUCTION AND INFERENCE
# -----------------------------------------------------------------------------
def build_listener_prompt(
    question: str,
    response: str,
    contexts: dict[str, str],
    randomised_order: list[str],
) -> str:
    """
    Build the listener prompt.

    The listener sees the question and response (context hidden) and must
    assign a probability distribution over the four possible contexts.
    The same randomised presentation order is used for all 10 responses
    per prompt, so results are directly comparable within a prompt.

    The prompt is kept terse and template-like to help DeepSeek-LLM-7B-Chat
    comply with the JSON-only output requirement.
    """
    options = "\n".join(
        f"{COND_LABELS[cond]}: "
        + (f'"{contexts[cond]}"' if contexts[cond] else "[No context was given]")
        for cond in randomised_order
    )

    return (
        f"QUESTION: {question}\n\n"
        f"RESPONSE: {response}\n\n"
        f"The response was generated after seeing exactly one of these contexts:\n"
        f"{options}\n\n"
        f"Assign probabilities (must sum to 1.0) reflecting how likely each "
        f"context produced this response. Output ONLY this JSON, no other text:\n"
        f"{{\"A\": 0.25, \"B\": 0.25, \"C\": 0.25, \"D\": 0.25}}\n"
        f"Replace each 0.25 with your actual probability estimate."
    )


def parse_listener_response(raw: str) -> dict[str, float] | None:
    """
    Parse the listener model's JSON probability distribution.
    Returns None on parse failure (row is flagged as parse_failed in output).
    Handles markdown code fences and JSON embedded in surrounding text.
    """
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        probs = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{[^{}]+\}', raw)
        if not m:
            return None
        try:
            probs = json.loads(m.group())
        except json.JSONDecodeError:
            return None

    if not all(k in probs for k in ["A", "B", "C", "D"]):
        return None

    total = sum(float(v) for v in probs.values())
    if total <= 0:
        return None
    # Reject degenerate all-zero outputs (model copied the template literally)
    if max(float(v) for v in probs.values()) == 0.0:
        return None
    return {k: float(v) / total for k, v in probs.items()}


def query_listener(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt_text: str,
    max_retries: int = 3,
) -> dict[str, float] | None:
    """
    Run the listener model and parse the probability distribution.
    Uses the DeepSeek chat template with a system prompt enforcing JSON output.
    Uses greedy decoding (do_sample=False) for stable, reproducible estimates.
    Passes attention_mask explicitly to suppress pad==eos warnings.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise analytical assistant. "
                "You MUST respond with ONLY a JSON object containing exactly "
                "four keys A, B, C, D with float values that sum to 1.0. "
                "Do not write any other text, explanation, or preamble."
            ),
        },
        {"role": "user", "content": prompt_text},
    ]

    for attempt in range(max_retries):
        try:
            formatted = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            encoding = tok(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            input_ids      = encoding["input_ids"].to(model.device)
            attention_mask = encoding["attention_mask"].to(model.device)
            prompt_len     = input_ids.shape[1]

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=LISTENER_MAX_NEW_TOKENS,
                    do_sample=False,          # greedy — no temperature/top_p needed
                    pad_token_id=tok.pad_token_id,
                )

            new_tokens = output[0][prompt_len:]
            raw = tok.decode(new_tokens, skip_special_tokens=True).strip()

            result = parse_listener_response(raw)
            if result is not None:
                return result

            print(f"  [WARNING] Parse failed (attempt {attempt+1}): {raw[:100]}")

        except Exception as e:
            print(f"  [WARNING] Inference error (attempt {attempt+1}): {e}")

    return None


# -----------------------------------------------------------------------------
# LISTENER ANALYSIS -- main loop
# -----------------------------------------------------------------------------
def run_listener_study(
    df: pd.DataFrame,
    speaker: str,
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    dry_run: bool,
) -> pd.DataFrame:
    """
    Evaluate every speaker response through the listener model.

    Returns a DataFrame with one row per response:
        prompt_id, question, condition, response_idx,
        P_NC, P_DI, P_II, P_SI,
        log_P_correct, V_L  (= log L(m|u), C(u) = 0)
        parse_failed
    """
    context_lookup = build_context_lookup(df)

    questions = df["question"].unique()
    if dry_run:
        questions = questions[:2]
        df = df[df["question"].isin(questions)]

    print(f"\nListener study: {len(df)} responses  |  {len(questions)} questions")
    print(f"Listener: {LISTENER_DISPLAY}")

    rows = []

    for pid, prompt_group in tqdm(
        df.groupby("prompt_id", sort=True), desc="Prompts"
    ):
        question  = prompt_group["question"].iloc[0]
        condition = prompt_group["condition"].iloc[0]   # true condition

        # Build context dict from lookup (derived from responses df itself)
        q_ctx = context_lookup[context_lookup["question"] == question]
        contexts = {row.condition: row.context for _, row in q_ctx.iterrows()}
        for cond in CONDITIONS:
            contexts.setdefault(cond, "")

        # Fix randomised presentation order per prompt (seeded by prompt_id)
        rng = np.random.default_rng(
            int(pid) if str(pid).isdigit() else hash(str(pid)) % 2**31
        )
        randomised_order = rng.permutation(CONDITIONS).tolist()

        for _, resp_row in prompt_group.iterrows():
            response_text = resp_row["response_text"]
            response_idx  = int(resp_row["response_idx"])

            listener_prompt = build_listener_prompt(
                question, response_text, contexts, randomised_order
            )

            probs = query_listener(tok, model, listener_prompt)

            if probs is None:
                rows.append({
                    "prompt_id":     pid,
                    "question":      question,
                    "condition":     condition,
                    "response_idx":  response_idx,
                    "P_NC":          np.nan,
                    "P_DI":          np.nan,
                    "P_II":          np.nan,
                    "P_SI":          np.nan,
                    "log_P_correct": np.nan,
                    "V_L":           np.nan,
                    "parse_failed":  True,
                })
                continue

            # Map option labels back to condition names using the local order
            label_to_cond = {COND_LABELS[c]: c for c in randomised_order}
            p_by_cond = {label_to_cond[lbl]: p for lbl, p in probs.items()}

            p_correct = p_by_cond.get(condition, 1e-10)
            log_p     = float(np.log(max(p_correct, 1e-10)))

            rows.append({
                "prompt_id":     pid,
                "question":      question,
                "condition":     condition,
                "response_idx":  response_idx,
                "P_NC":  p_by_cond.get("no_context",              0.0),
                "P_DI":  p_by_cond.get("direct_information",      0.0),
                "P_II":  p_by_cond.get("implicature_information",  0.0),
                "P_SI":  p_by_cond.get("stochastic_information",   0.0),
                "log_P_correct": log_p,
                "V_L":           log_p,   # V_L(m,u) = log L(m|u), C(u) = 0
                "parse_failed":  False,
            })

    result_df = pd.DataFrame(rows)

    out_path = RESULTS_DIR / f"listener_results_{speaker}.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nListener results  -> {out_path}")
    n_failed = int(result_df["parse_failed"].sum())
    print(f"  Parse failures: {n_failed} / {len(result_df)}")

    # Per-prompt summary (mean V_L across 10 responses)
    summary = (
        result_df[~result_df["parse_failed"]]
        .groupby(["prompt_id", "question", "condition"])
        .agg(
            V_L_mean       = ("V_L", "mean"),
            V_L_sd         = ("V_L", "std"),
            P_correct_mean = ("log_P_correct", lambda x: float(np.exp(x.mean()))),
            n_responses    = ("V_L", "count"),
        )
        .reset_index()
    )
    sum_path = RESULTS_DIR / f"listener_summary_{speaker}.csv"
    summary.to_csv(sum_path, index=False)
    print(f"Listener summary  -> {sum_path}")

    # Condition-level V_L summary
    print("\nMean V_L by condition  (chance = log(0.25) = -1.386):")
    for cond in CONDITIONS:
        sub = result_df[
            (result_df["condition"] == cond) & ~result_df["parse_failed"]
        ]
        if not sub.empty:
            print(f"  {cond:<30s}: {sub['V_L'].mean():.4f}  "
                  f"(SD={sub['V_L'].std():.4f})")

    return result_df


# -----------------------------------------------------------------------------
# LEXICAL DIVERSITY ANALYSIS
# -----------------------------------------------------------------------------
def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def type_token_ratio(texts: list[str]) -> float:
    """TTR across the pooled response set. Higher = more diverse vocabulary."""
    all_tokens = [t for text in texts for t in _tokenise(text)]
    return len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0


def mean_response_len(texts: list[str]) -> float:
    return float(np.mean([len(_tokenise(t)) for t in texts]))


def unique_bigram_ratio(texts: list[str]) -> float:
    """Proportion of unique bigrams across the pooled response set."""
    all_bigrams = []
    for text in texts:
        tokens = _tokenise(text)
        all_bigrams.extend(zip(tokens, tokens[1:]))
    return len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0


def mean_pairwise_jaccard(texts: list[str]) -> float:
    """Mean pairwise Jaccard similarity. Low = diverse, high = similar wording."""
    token_sets = [set(_tokenise(t)) for t in texts]
    n = len(token_sets)
    if n < 2:
        return 0.0
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = token_sets[i], token_sets[j]
            union = len(a | b)
            sims.append(len(a & b) / union if union > 0 else 0.0)
    return float(np.mean(sims))


def distinct_opening_phrases(texts: list[str], n: int = 4) -> int:
    """Number of distinct n-word opening phrases across responses."""
    openings = set()
    for text in texts:
        tokens = _tokenise(text)
        if len(tokens) >= n:
            openings.add(" ".join(tokens[:n]))
    return len(openings)


def run_lexical_analysis(
    df: pd.DataFrame, speaker: str, dry_run: bool
) -> pd.DataFrame:
    """Compute lexical diversity metrics per prompt x condition."""
    questions = df["question"].unique()
    if dry_run:
        questions = questions[:2]
        df = df[df["question"].isin(questions)]

    print(f"\nLexical diversity: {df['question'].nunique()} questions")

    rows = []
    for (pid, cond), group in df.groupby(["prompt_id", "condition"]):
        texts = group["response_text"].tolist()
        rows.append({
            "prompt_id":             pid,
            "question":              group["question"].iloc[0],
            "condition":             cond,
            "n_responses":           len(texts),
            "type_token_ratio":      type_token_ratio(texts),
            "mean_response_len":     mean_response_len(texts),
            "unique_bigram_ratio":   unique_bigram_ratio(texts),
            "mean_pairwise_jaccard": mean_pairwise_jaccard(texts),
            "distinct_openings_4w":  distinct_opening_phrases(texts, n=4),
            "context_similarity":    (
                group["context_question_similarity"].iloc[0]
                if "context_question_similarity" in group.columns
                else np.nan
            ),
        })

    result = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"lexical_diversity_{speaker}.csv"
    result.to_csv(out_path, index=False)
    print(f"Lexical diversity -> {out_path}")

    cols = ["type_token_ratio", "mean_response_len", "unique_bigram_ratio",
            "mean_pairwise_jaccard", "distinct_openings_4w"]
    summary = result.groupby("condition")[cols].mean().round(4)
    order = [c for c in CONDITIONS if c in summary.index]
    print("\nMeans by condition:")
    print(summary.loc[order].to_string())

    return result


# -----------------------------------------------------------------------------
# CROSS-SPEAKER COMPARISON (--compare mode)
# -----------------------------------------------------------------------------
def run_comparison() -> None:
    """
    Load all listener_results_*.csv files and compare V_L across speakers.
    Saves a combined CSV and prints a summary table.
    """
    import glob
    files = sorted(glob.glob(str(RESULTS_DIR / "listener_results_*.csv")))
    if not files:
        print(f"No listener_results_*.csv files found in {RESULTS_DIR}.")
        return

    frames = []
    for f in files:
        sp = Path(f).stem.replace("listener_results_", "")
        d = pd.read_csv(f)
        d["speaker"] = sp
        frames.append(d)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[~combined["parse_failed"].fillna(False)]

    out = RESULTS_DIR / "listener_results_all_speakers.csv"
    combined.to_csv(out, index=False)
    print(f"Combined results -> {out}")

    print("\n=== V_L by speaker x condition  (chance = log(0.25) = -1.386) ===")
    pivot = (
        combined.groupby(["speaker", "condition"])["V_L"]
        .mean()
        .unstack("condition")
    )
    cond_order = [c for c in CONDITIONS if c in pivot.columns]
    print(pivot[cond_order].round(4).to_string())


# -----------------------------------------------------------------------------
# ALPHA ESTIMATION -- merge V_L with DEAS
# -----------------------------------------------------------------------------
def estimate_alpha_from_VL(speaker: str) -> None:
    """
    Merge listener V_L with DEAS from extended_metrics_{speaker}.csv.
    Estimates alpha-hat in G_alpha[S,L] = H_S(U|M) + alpha * E_S[V_L].
    alpha-hat = -r(DEAS, V_L) in standardised units.
    alpha-hat > 0 means entropy falls as listener-recoverable utility rises,
    consistent with G_alpha optimisation.
    """
    from scipy.stats import pearsonr

    listener_path = RESULTS_DIR / f"listener_results_{speaker}.csv"
    metrics_path  = RESULTS_DIR / f"extended_metrics_{speaker}.csv"

    if not listener_path.exists():
        print(f"  listener_results_{speaker}.csv not available -- skipping alpha.")
        return
    if not metrics_path.exists():
        print(f"  extended_metrics_{speaker}.csv not found -- skipping alpha.")
        return

    listener = pd.read_csv(listener_path)
    listener = listener[~listener["parse_failed"].fillna(False)]
    metrics  = pd.read_csv(metrics_path)

    # Average V_L across responses per prompt
    vl_prompt = (
        listener.groupby("prompt_id")["V_L"]
        .mean().reset_index()
        .rename(columns={"V_L": "V_L_mean"})
    )

    # Compute DEAS vs NC baseline per question
    metrics["question"] = (metrics["prompt_id"] - 1) // 4 + 1
    nc = (
        metrics[metrics["condition"] == "no_context"]
        [["question", "eas_mean"]]
        .rename(columns={"eas_mean": "nc_eas_mean"})
    )
    m = metrics[metrics["condition"] != "no_context"].merge(nc, on="question")
    m["delta_eas"] = m["eas_mean"] - m["nc_eas_mean"]

    merged = (
        m.merge(vl_prompt, on="prompt_id", how="inner")
        .dropna(subset=["delta_eas", "V_L_mean"])
    )

    if len(merged) < 10:
        print("  Insufficient data to estimate alpha.")
        return

    r, p = pearsonr(merged["V_L_mean"], merged["delta_eas"])
    alpha_hat = -r

    print(f"\n=== alpha-hat for speaker={speaker} using empirical V_L ===")
    print(f"  r(DEAS, V_L) = {r:+.3f}  p={p:.4f}{'*' if p < 0.05 else ''}")
    print(f"  alpha-hat = {alpha_hat:+.3f}  "
          f"({'consistent with G_alpha' if alpha_hat > 0 else 'inconsistent with G_alpha'})")
    print(f"  n = {len(merged)} prompt-condition pairs")
    print(f"  Note: alpha-hat > 0 means H_S falls as V_L rises.")
    print(f"  The Zaslavsky critical value alpha=1 is not directly comparable "
          f"(V_L is in log-probability units; alpha=1 was derived in utility units).")

    out = RESULTS_DIR / f"alpha_estimate_{speaker}.csv"
    merged[["prompt_id", "condition", "delta_eas", "V_L_mean"]].to_csv(out, index=False)
    print(f"  Saved -> {out}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    _setup_hf_auth()
    args = parse_args()

    if args.compare:
        run_comparison()
        return

    if args.speaker is None:
        print("Error: --speaker required unless --compare is set.")
        print("  e.g. python qxc_listener_analysis.py --speaker deepseek")
        return

    speaker = args.speaker
    print("=" * 60)
    print("Listener study + Lexical diversity")
    print(f"  Speaker: {speaker}  |  dry_run: {args.dry_run}")
    print("=" * 60)

    df = load_responses(speaker)
    print(f"Loaded {len(df)} responses ({df['prompt_id'].nunique()} prompts, "
          f"{df['response_idx'].nunique()} samples each)")

    # Analysis 1: Lexical diversity (no model needed)
    run_lexical_analysis(df, speaker, dry_run=args.dry_run)

    # Analysis 2: Listener study
    if not args.skip_listener:
        print("\nLoading listener model...")
        tok, model = load_listener_model()
        run_listener_study(df, speaker, tok, model, dry_run=args.dry_run)

        # Analysis 3: alpha estimation
        estimate_alpha_from_VL(speaker)

    print("\nDone.")


if __name__ == "__main__":
    main()