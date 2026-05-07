"""
compute_sbert_sim.py — Add context-question SBERT similarity to a pilot CSV.

Computes cosine similarity between each (question, context) pair using
all-MiniLM-L6-v2. This is a pre-model measure — computed from raw text only,
independent of any LLM outputs — so it can be used as a predictor without
introducing circular logic.

For no_context rows (empty context), similarity is set to 0.0.

Usage:
    python questions_x_context/scripts/compute_sbert_sim.py \\
        --input  questions_x_context/data/pilot_summary.csv \\
        --output questions_x_context/data/pilot_summary_enriched.csv

Requirements: sentence-transformers, numpy
    pip install sentence-transformers numpy
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="questions_x_context/data/pilot_summary.csv")
    parser.add_argument("--output", default="questions_x_context/data/pilot_summary_enriched.csv")
    parser.add_argument("--model",  default="all-MiniLM-L6-v2",
                        help="sentence-transformers model name")
    args = parser.parse_args()

    rows = []
    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows in {args.input}")

    print(f"Loading SBERT model: {args.model}")
    model = SentenceTransformer(args.model)

    questions = [r["question"] for r in rows]
    contexts  = [r.get("context", "") or "" for r in rows]

    print(f"Encoding {len(rows)} question–context pairs...")
    q_embs = model.encode(questions, show_progress_bar=False)
    c_embs = model.encode(contexts,  show_progress_bar=False)

    out_rows = []
    for i, r in enumerate(rows):
        r2 = dict(r)
        r2["context_question_sbert_sim"] = round(cosine(q_embs[i], c_embs[i]), 6)
        r2["context_word_count"]  = len(contexts[i].split()) if contexts[i].strip() else 0
        r2["prompt_word_count"]   = len((questions[i] + " " + contexts[i]).split())
        out_rows.append(r2)

    new_cols = ["context_question_sbert_sim", "context_word_count", "prompt_word_count"]
    fieldnames = [k for k in rows[0].keys() if k not in new_cols] + new_cols

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nWritten: {args.output}")
    print("\nprompt_id  condition                       sbert_sim  ctx_words")
    for r2 in out_rows:
        print(f"  {r2['prompt_id']:>3s}  {r2['condition']:<32s}  "
              f"{float(r2['context_question_sbert_sim']):.4f}     "
              f"{r2['context_word_count']}")


if __name__ == "__main__":
    main()
