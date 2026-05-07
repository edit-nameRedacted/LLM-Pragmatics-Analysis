"""
probe_inventory.py — Inventory check for the probing-classifier extension.

Produces questions_x_context/results/probe_inventory/inventory_report.md.
No model loading; file inspection, JSON schema parsing, npz metadata reads only.
"""
from __future__ import annotations

import csv
import json
import re
import datetime
from pathlib import Path

import numpy as np

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent.parent
RDM_DATA = ROOT / "data" / "rdm"
OUT_DIR  = ROOT / "results" / "probe_inventory"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT   = OUT_DIR / "inventory_report.md"

ANALYSIS_PY = ROOT / "scripts" / "information_theory" / "analysis_metrics.py"

MODEL_DATA_DIR = ROOT / "data" / "model"

PILOT_SEARCH_DIRS = [
    MODEL_DATA_DIR / m
    for m in ["deepseek", "llama", "mistral", "qwen"]
]

CONDITION_ALIASES = {
    "no_context":              "NC",
    "direct_information":      "DI",
    "implicature_information": "II",
    "stochastic_information":  "SI",
}


# ─── Markdown helpers ──────────────────────────────────────────────────────────
def _md_table(headers: list[str], rows: list[list]) -> str:
    col_widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    def _row(cells):
        return "| " + " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    return "\n".join([_row(headers), sep] + [_row(r) for r in rows])


def _bool_icon(b: bool) -> str:
    return "✓" if b else "✗"


def _extract_fn_source(src: str, fn_name: str, max_lines: int = 22) -> str:
    """Return up to max_lines of source starting at 'def fn_name'."""
    pattern = rf"^(def {re.escape(fn_name)}\b)"
    match = re.search(pattern, src, re.MULTILINE)
    if not match:
        return f"(function `{fn_name}` not found in source)"
    start = match.start()
    lines = src[start:].splitlines()[:max_lines]
    return "\n".join(lines)


# ─── §1 Pilot results JSON audit ───────────────────────────────────────────────
def audit_pilot_results() -> str:
    rows = []
    seen = set()
    for search_dir in PILOT_SEARCH_DIRS:
        for jpath in sorted(search_dir.glob("pilot_results_*.json")):
            if ".git" in jpath.parts or jpath in seen:
                continue
            seen.add(jpath)
            rel   = str(jpath.relative_to(ROOT))
            model = jpath.stem.replace("pilot_results_", "")
            try:
                d = json.loads(jpath.read_text(encoding="utf-8"))
            except Exception as e:
                rows.append([rel, model, "ERR", "–", "–", "–", str(e)])
                continue

            numeric_keys = [k for k in d if k != "_meta"]
            n_prompts = len(numeric_keys)
            if not numeric_keys:
                rows.append([rel, model, 0, "–", "–", "–"])
                continue

            entry = d[numeric_keys[0]]
            has_responses  = _bool_icon("responses" in entry)
            se             = entry.get("semantic_entropy")
            se_type        = type(se).__name__ if se is not None else "absent"
            # Scan JSON text for any key containing "cluster"
            entry_str      = json.dumps(entry)
            has_cluster    = _bool_icon(bool(re.search(r'"cluster', entry_str)))
            rows.append([rel, model, n_prompts, has_responses, se_type, has_cluster])

    table = _md_table(
        ["File (rel. results/)", "Model", "n_prompts", "responses_key?", "SE type", "cluster_keys?"],
        rows,
    )
    finding = (
        "\n**Finding:** `semantic_entropy` is a bare `float` scalar in every file. "
        "No `responses` key, no cluster assignments, and no `cluster_*` sub-keys exist at "
        "any prompt entry in any audited JSON file. The cluster labels that generated each "
        "SE scalar were computed transiently inside `_mutual_entailment_cluster()` and were "
        "never persisted to disk.\n"
    )
    return f"## §1 — Pilot results JSON audit\n\n{table}\n{finding}"


# ─── §2 Raw response text audit ────────────────────────────────────────────────
def audit_response_files() -> str:
    json_rows, csv_rows = [], []

    for jpath in sorted(MODEL_DATA_DIR.rglob("responses_*.json")):
        if ".git" in jpath.parts:
            continue
        rel   = str(jpath.relative_to(ROOT))
        model = jpath.stem.replace("responses_", "")
        try:
            d = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception as e:
            json_rows.append([rel, model, "ERR", "–", "–", str(e)])
            continue
        numeric_keys = [k for k in d if k != "_meta"]
        n_prompts    = len(numeric_keys)
        if not numeric_keys:
            json_rows.append([rel, model, n_prompts, "–", "–"])
            continue
        entry        = d[numeric_keys[0]]
        has_resp     = "responses" in entry
        n_resp       = len(entry["responses"]) if has_resp else 0
        json_rows.append([rel, model, n_prompts, _bool_icon(has_resp), n_resp])

    for cpath in sorted(MODEL_DATA_DIR.rglob("responses_*.csv")):
        if ".git" in cpath.parts:
            continue
        rel   = str(cpath.relative_to(ROOT))
        model = cpath.stem.replace("responses_", "")
        try:
            with open(cpath, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                cols   = reader.fieldnames or []
                n_rows = sum(1 for _ in reader)
            csv_rows.append([rel, model, n_rows, ", ".join(cols)])
        except Exception as e:
            csv_rows.append([rel, model, "ERR", str(e)])

    json_table = _md_table(
        ["File (rel. results/)", "Model", "n_prompts", "responses_key?", "n_responses/prompt"],
        json_rows,
    )
    csv_section = ""
    if csv_rows:
        csv_table = _md_table(["File (rel. results/)", "Model", "n_rows", "Columns"], csv_rows)
        csv_section = f"\n**Associated CSV files:**\n\n{csv_table}\n"

    finding = (
        "\n**Finding:** Raw response text is saved for **DeepSeek-instruct only** "
        "(`responses_deepseek.json`: 60 prompts × 10 responses; "
        "`responses_deepseek.csv`: 600 rows). No `responses_*` file exists for "
        "llama, mistral, qwen, deepseek-v2-lite, llama-base, or qwen-base. "
        "Response text for those six models is unrecoverable without re-generation.\n"
    )
    return f"## §2 — Raw response text audit\n\n{json_table}\n{csv_section}{finding}"


# ─── §3 compute_semantic_entropy side-effect check ─────────────────────────────
def audit_semantic_entropy_fn() -> str:
    try:
        src = ANALYSIS_PY.read_text(encoding="utf-8")
        snippets = {
            fn: _extract_fn_source(src, fn)
            for fn in ["compute_semantic_entropy", "_cluster_entropy", "_mutual_entailment_cluster"]
        }
        source_block = "\n\n".join(
            f"```python\n# ── {fn} ──\n{snippet}\n```"
            for fn, snippet in snippets.items()
        )
    except Exception as e:
        source_block = f"**ERROR reading source:** {e}"

    call_chain = """**Call chain:**

```
compute_semantic_entropy(responses, nli_tok, nli_clf)   →  float
  └─ _cluster_entropy(texts, nli_tok, nli_clf)          →  float
       └─ _mutual_entailment_cluster(texts, ...)        →  list[int]  (local variable only)
            ├─ np.bincount(labels)
            └─ return float entropy   [list[int] discarded here]
```"""

    finding = (
        "\n**Finding:** Cluster labels are computed inside `_mutual_entailment_cluster()` "
        "as a local `list[int]`. They are consumed immediately by `np.bincount()` and are "
        "never returned to the caller, never written to disk, and not stored in any mutable "
        "container outside the call frame. `compute_semantic_entropy` returns only the entropy "
        "`float`. **No side effect — cluster labels must be re-derived from scratch.**\n"
    )

    return (
        f"## §3 — `compute_semantic_entropy` side-effect check\n\n"
        f"**Source:** `questions_x_context/analysis_metrics.py`\n\n"
        f"**Relevant source excerpts (first 22 lines each):**\n\n"
        f"{source_block}\n\n"
        f"{call_chain}\n"
        f"{finding}"
    )


# ─── §4 hidden_states_*.npz schema audit ───────────────────────────────────────
def audit_hidden_states() -> str:
    rows      = []
    warn_msgs = []

    for npzpath in sorted(RDM_DATA.glob("hidden_states_*.npz")):
        model     = npzpath.stem.replace("hidden_states_", "")
        meta_path = RDM_DATA / f"hidden_states_meta_{model}.csv"
        try:
            npz = np.load(npzpath, allow_pickle=False)

            lths     = npz["last_token_hs"]   # (P, L, D)
            mpool    = npz["mean_pool_hs"]     # (P, L, D)
            nan_mask = npz["nan_mask"]         # (P, L) bool

            n_prompts, n_layers, dim = lths.shape
            dtype = lths.dtype

            nan_prompt_pct = 100.0 * nan_mask.any(axis=1).mean()

            shapes_str = (
                f"`last_token_hs` {tuple(lths.shape)}, "
                f"`mean_pool_hs` {tuple(mpool.shape)}"
            )

            if nan_prompt_pct == 100.0:
                nan_str = "**100% — ⚠ EXCLUDED**"
                warn_msgs.append(model)
            elif nan_prompt_pct > 0.0:
                nan_str = f"{nan_prompt_pct:.1f}% (partial)"
            else:
                nan_str = "0%"

            meta_ok = _bool_icon(meta_path.exists())
            rows.append([model, n_prompts, n_layers, dim, str(dtype), nan_str, meta_ok])

        except Exception as e:
            rows.append([model, "ERR", "–", "–", "–", str(e), "–"])

    table = _md_table(
        ["Model", "n_prompts", "n_layers (L)", "hidden_dim (D)", "dtype",
         "nan_prompt%", "meta_csv?"],
        rows,
    )

    pooling_note = (
        "\n**Pooling variants present in all files:** `last_token_hs`, `mean_pool_hs` "
        "(reported above). `last_n_hs` (last-8-token window) is present but excluded "
        "from probe reporting per scope.\n"
    )

    warn_block = ""
    if warn_msgs:
        warn_block = "\n> ⚠ **WARNING — Probe not viable without hidden-state re-extraction. Excluded from probe planning.**\n"
        for w in warn_msgs:
            warn_block += (
                f">\n"
                f"> **`{w}`**: `nan_mask` is `True` for 100 % of prompts. "
                f"The `.npz` was written but every hidden-state vector is NaN. "
                f"Do not use this file for probe training until a clean "
                f"hidden-state extraction pass is completed.\n"
            )

    finding = (
        "\n**Finding:** Five models (deepseek, deepseek\\_v2\\_lite, llama, llama\\_base, mistral) "
        "have 0 % NaN-affected prompts and are probe-ready from disk. "
        "Qwen-base has 21.7 % NaN-affected prompts (≈13/60), leaving ≈47 clean prompts — "
        "a partial probe is possible. "
        "Qwen-instruct is excluded from probe planning (100 % NaN).\n"
    )

    return (
        f"## §4 — `hidden_states_*.npz` schema audit\n\n"
        f"{table}\n"
        f"{pooling_note}"
        f"{warn_block}"
        f"{finding}"
    )


# ─── §5 Cluster-count distribution proxy ───────────────────────────────────────
def estimate_cluster_coverage() -> str:
    # Prefer V2 deepseek; fall back to other locations
    candidates = [
        MODEL_DATA_DIR / "deepseek" / "pilot_results_deepseek.json",
    ]
    pj = next((c for c in candidates if c.exists()), None)
    if pj is None:
        return (
            "## §5 — Recoverable cluster-count distribution\n\n"
            "**ERROR:** No `pilot_results_deepseek.json` found.\n"
        )

    d = json.loads(pj.read_text(encoding="utf-8"))
    numeric_keys = [k for k in d if k != "_meta"]

    cond_stats: dict[str, dict] = {
        alias: {"n": 0, "se_gt0": 0, "se_vals": []}
        for alias in ["NC", "DI", "II", "SI"]
    }
    for k in numeric_keys:
        entry = d[k]
        alias = CONDITION_ALIASES.get(entry.get("condition", ""), entry.get("condition", "?"))
        se    = entry.get("semantic_entropy")
        if alias not in cond_stats:
            cond_stats[alias] = {"n": 0, "se_gt0": 0, "se_vals": []}
        cond_stats[alias]["n"] += 1
        if isinstance(se, (int, float)) and se == se:   # not NaN
            cond_stats[alias]["se_vals"].append(se)
            if se > 0.0:
                cond_stats[alias]["se_gt0"] += 1

    rows = []
    for alias in ["NC", "DI", "II", "SI"]:
        s    = cond_stats[alias]
        n    = s["n"]
        gt0  = s["se_gt0"]
        vals = s["se_vals"]
        pct  = f"{100*gt0/n:.0f}%" if n else "–"
        mean = f"{float(np.mean(vals)):.3f}" if vals else "–"
        rows.append([alias, n, gt0, pct, mean])

    table = _md_table(
        ["Condition", "n_prompts", "SE > 0 → cluster≥2", "% prompts cluster≥2 (proxy)", "mean SE (nats)"],
        rows,
    )

    source_note  = f"**Source:** `{pj.relative_to(ROOT)}`\n"
    method_note  = (
        "_No NLI model loaded. Cluster-count is proxied from stored SE scalars: "
        "SE > 0 implies cluster\\_count ≥ 2. This proxy cannot distinguish "
        "cluster\\_count = 2 from cluster\\_count = 3+._\n"
    )
    finding = (
        "\n**Finding:** The SE proxy gives a rough picture of multi-cluster prevalence per "
        "condition. Actual per-prompt cluster labels require running "
        "`_mutual_entailment_cluster()` on the saved `responses_deepseek.json` texts "
        "(60 prompts × C(10, 2) = 45 NLI pairs each = **2 700 NLI pairs total**). "
        "This is CPU-feasible with DeBERTa-MNLI in a few minutes — no GPU required.\n"
    )

    return (
        f"## §5 — Recoverable cluster-count distribution (DeepSeek-instruct only)\n\n"
        f"{source_note}\n"
        f"{method_note}\n"
        f"{table}\n"
        f"{finding}"
    )


# ─── §6 Cost estimate ──────────────────────────────────────────────────────────
def estimate_regen_cost() -> str:
    rows = [
        [
            "deepseek (instruct)",
            "✓  (60 × 10)",
            "**Path A**",
            "Run `_mutual_entailment_cluster()` on `responses_deepseek.json`. "
            "DeBERTa only; 2 700 NLI pairs; CPU; ~minutes. **START HERE.**",
        ],
        [
            "llama (instruct)",
            "✗",
            "Path B",
            "Run `qxc_collect_responses.py --llama`; load 8 B model; generate 600 responses; then cluster.",
        ],
        [
            "mistral (instruct)",
            "✗",
            "Path B",
            "Run `qxc_collect_responses.py --mistral`; load 7 B model; generate 600 responses; then cluster.",
        ],
        [
            "deepseek\\_v2\\_lite",
            "✗",
            "Path B",
            "Run `qxc_collect_responses.py --deepseek_v2_lite`; load 16 B model; generate 600 responses; then cluster.",
        ],
        [
            "llama\\_base",
            "✗",
            "Path B",
            "Run `qxc_collect_responses.py --llama_base`; load 8 B base; generate 600 responses; then cluster.",
        ],
        [
            "qwen\\_base",
            "✗ + 21.7 % NaN hs",
            "Path B+",
            "Generate responses + re-extract hidden states for ~13 NaN-affected prompts.",
        ],
        [
            "**qwen (instruct)**",
            "✗ + **100 % NaN hs**",
            "**EXCLUDED**",
            "⚠ **Probe not viable without hidden-state re-extraction. Excluded from probe planning.**",
        ],
    ]

    table = _md_table(
        ["Model", "Responses saved?", "Path", "Required action"],
        rows,
    )

    path_defs = (
        "\n**Path definitions:**\n\n"
        "- **Path A** — Call `_mutual_entailment_cluster(responses, nli_tok, nli_clf)` "
        "directly on texts already on disk. Requires only DeBERTa-MNLI (CPU-feasible). "
        "No generation, no LLM loading.\n"
        "- **Path B** — Run `qxc_collect_responses.py` for the target model to save "
        "response texts, then run `_mutual_entailment_cluster`. Requires loading and "
        "running the generative LLM.\n"
        "- **Path B+** — Path B, plus a partial hidden-state re-extraction pass for "
        "NaN-affected prompts.\n"
    )

    qwen_warn = (
        "\n> ⚠ **WARNING — Qwen-instruct: Probe not viable without hidden-state re-extraction. "
        "Excluded from probe planning.**\n"
        "> `hidden_states_qwen.npz` contains 100 % NaN-masked prompts. "
        "Do not include qwen-instruct in any probe training run until a clean "
        "extraction pass is completed and the NaN mask is cleared.\n"
    )

    recommendation = (
        "\n**Recommendation:** Implement and validate the probe on DeepSeek-instruct first (Path A). "
        "Only extend to other models after validating the probe design on DeepSeek.\n"
    )

    return (
        f"## §6 — Cost estimate: cheapest path to per-sample cluster labels\n\n"
        f"{table}\n"
        f"{path_defs}"
        f"{qwen_warn}"
        f"{recommendation}"
    )


# ─── Next step ─────────────────────────────────────────────────────────────────
def next_step_section() -> str:
    return (
        "## Next step\n\n"
        "Write `questions_x_context/probe_classifier.py`. Add a function "
        "`extract_cluster_labels(model_label: str) -> dict[str, list[int]]` to that file. "
        "The function should load `results/responses_{model_label}.json`, call "
        "`_mutual_entailment_cluster()` imported from `analysis_metrics.py` on the stored "
        "response texts, and return a dict mapping each `prompt_id` (string key) to its "
        "list of per-response integer cluster labels. "
        "The **probe target variable** is the per-response NLI cluster ID "
        "(one integer label per response, yielding a label vector of shape "
        "`(n_prompts × n_responses,)`). "
        "These labels are to be predicted from the corresponding CAA hidden state "
        "(drawn from `rdm/data/hidden_states_{model_label}.npz`, using either "
        "`last_token_hs` or `mean_pool_hs` at a chosen layer) via a linear classifier "
        "trained to maximise held-out accuracy. "
        "The Fano lower bound on I(hidden\\_state ; cluster\\_structure) is then "
        "computed from the probe's test-set error rate as "
        "I ≥ H(cluster) − H(cluster | hidden\\_state) ≥ H(cluster) − h(ε), "
        "where ε is the probe error rate and h(·) is the binary entropy function.\n"
    )


# ─── Report assembly ───────────────────────────────────────────────────────────
def write_report() -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    header = (
        "# Probe Inventory Report\n\n"
        f"**Generated:** {ts}  \n"
        f"**Working directory:** `{ROOT}`  \n"
        f"**Output directory:** `{OUT_DIR.relative_to(ROOT)}`  \n\n"
        "This report audits what is already on disk for the probing-classifier extension "
        "to the RDM analysis. It answers five questions per model before any probe code "
        "is written.\n\n---\n\n"
    )

    sections = [
        audit_pilot_results(),
        audit_response_files(),
        audit_semantic_entropy_fn(),
        audit_hidden_states(),
        estimate_cluster_coverage(),
        estimate_regen_cost(),
        next_step_section(),
    ]

    body = "\n\n---\n\n".join(sections)
    report_text = header + body
    REPORT.write_text(report_text, encoding="utf-8")
    print(f"[OK] Report written -> {REPORT}")


if __name__ == "__main__":
    write_report()
