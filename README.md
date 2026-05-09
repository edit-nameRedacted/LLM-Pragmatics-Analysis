# LLM Detection of Polysemous Message Encoding

Reproduction package for: *LLM Detection of Polysemous Message Encoding: Pragmatic Context Modulates LLM Processing Uncertainty* (Allahverdiyeva & Harkness, 2026).

---

## Overview

This repository contains all code and data needed to reproduce the paper's results. The pipeline has four stages:

| Stage | What it does | Scripts |
|---|---|---|
| **0. Setup** | Install dependencies, set HF token | — |
| **1. Model inference** | Run 5 LLMs on 60 prompts, collect EAS / CAA / beam metrics | `scripts/model_prompting/` |
| **2. RSA data collection** | Extract per-layer hidden states for RDM analysis | `scripts/model_prompting/qxc_main_RSA.py` |
| **3. Statistical analysis** | Wilcoxon tests, trajectory metrics, MLMs, RSA | `scripts/information_theory/`, `scripts/statistical/` |
| **4. Figures** | Reproduce all paper plots | `scripts/statistical/plot_buckets.py`, `entropy_trajectory_analysis.py` |

**If you only want to reproduce the statistical results and figures** (without re-running inference), all pre-computed model outputs are in `data/model/` and `data/rdm/`. Skip to [Step 3](#step-3-statistical-analysis).

---

## Repository layout

```
├── data/
│   ├── human/                  # Human rater scores (context relevance, question diversity)
│   │   ├── QuestionContext_Scores.csv
│   │   ├── QuestionDiversity_scores.csv
│   │   ├── data_compiled_sim.csv
│   │   └── prompts+SBERTsim_scores.csv
│   ├── model/                  # Pre-computed model outputs (canonical V2 data)
│   │   ├── deepseek/           # extended_metrics_deepseek.csv, pilot_summary_deepseek.csv
│   │   ├── llama/
│   │   ├── mistral/
│   │   └── qwen/
│   └── rdm/                    # Hidden-state metadata CSVs (NPZ files regenerated separately)
│       └── hidden_states_meta_*.csv
├── prompts/
│   └── prompts_full.csv        # 60 prompts (15 questions × 4 conditions)
├── scripts/
│   ├── model_prompting/        # LLM inference and hidden-state extraction
│   │   ├── qxc_main.py         # Exp 1 & 2.1–2.3: EAS, CAA, beam diversity
│   │   ├── qxc_main_CAA.py     # CAA-only variant
│   │   ├── qxc_main_RSA.py     # Exp 2.4: extract layer hidden states
│   │   ├── qxc_main_RSA_base.py # Same for base (non-instruct) models
│   │   └── qxc_collect_responses.py  # Lightweight response-only collection
│   ├── information_theory/
│   │   ├── entropy_trajectory_analysis.py  # Exp 2.1: HL_rel, H_fb, Wilcoxon tests + plots
│   │   └── run_mlm_analysis.py             # Exp 2.2: multilevel models (MLM)
│   └── statistical/
│       ├── full_analysis.py     # Exp 1: EAS Wilcoxon tests + report
│       ├── plot_buckets.py      # Exp 2.3: CAA trajectory + beam diversity figures
│       └── qxc_listener_analysis.py  # Lexical diversity analysis
├── results/                    # All output files written here
│   ├── plots/                  # Entropy trajectory figures (Figs 1–2)
│   ├── MLM/                    # MLM outputs: analysis_base.csv, mlm_scalar_extended.csv
│   ├── probe_inventory/        # Effort–utility scatter plots
│   └── rdm/rdm_sbert/          # RSA outputs: rdm_results.csv, rdm_trajectory.png
└── notebooks/
    └── rdm_analysis.ipynb      # RSA analysis notebook (Exp 2.4)
```

---

## Step 0: Setup

### Requirements

- Python 3.10 or 3.11
- CUDA-capable GPU with at least 16 GB VRAM (for model inference steps only)
- ~50 GB disk space for model weights (downloaded automatically on first run)

### Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes
pip install numpy pandas scipy scikit-learn statsmodels
pip install sentence-transformers
pip install matplotlib tqdm python-dotenv
```

### HuggingFace token

Model inference requires a HuggingFace account with access to gated models. Accept the licence agreements on the HuggingFace pages for each model before running (links in [Models](#models) below), then set your token:

```bash
export HF_TOKEN=hf_your_token_here
```

Or create a `.env` file in the repo root (it is gitignored):

```
HF_TOKEN=hf_your_token_here
```

### Model weights path

By default, models are loaded from the HuggingFace hub cache (`~/.cache/huggingface/hub`). To use a custom local directory:

```bash
export QXC_MODELS_DIR=/path/to/your/models
```

---

## Step 1: Model inference (Experiments 1, 2.1–2.3)

> **Skip this step if using pre-computed data in `data/model/`.**

This runs all five models on the 60 prompts and collects EAS, CAA, and beam diversity metrics. Each call takes 2–6 hours on a single GPU.

```bash
cd scripts/model_prompting

python qxc_main.py --qwen
python qxc_main.py --llama
python qxc_main.py --mistral
python qxc_main.py --deepseek
python qxc_main.py --deepseek_v2_lite
```

Output files (`extended_metrics_<model>.csv`, `pilot_summary_<model>.csv`) are written to the current directory. Move them to `data/model/<model>/` before running analyses.

**Reproducibility note:** Inference uses a fixed seed (`SEED=42`). Per-sample seed is `42 + run_id×10000 + prompt_id×100 + sample_index`. With `--run_id 0` (default) results are exactly reproducible across identical hardware and software versions; minor floating-point differences may occur across GPU types.

### Models

| Flag | Model | HuggingFace page |
|---|---|---|
| `--qwen` | Qwen2.5-7B-Instruct | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| `--llama` | LLaMA-3.1-8B-Instruct | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |
| `--mistral` | Mistral-7B-Instruct-v0.3 | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| `--deepseek` | DeepSeek-LLM-7B-Chat | [deepseek-ai/deepseek-llm-7b-chat](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat) |
| `--deepseek_v2_lite` | DeepSeek-V2-Lite-Chat | [deepseek-ai/DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat) |

---

## Step 2: RSA hidden-state extraction (Experiment 2.4)

> **Skip this step if using pre-computed metadata in `data/rdm/`.**

Extracts per-layer hidden states for all 60 prompts. Output NPZ files are ~63–150 MB each and are excluded from git; only the companion `hidden_states_meta_*.csv` files are committed.

```bash
cd scripts/model_prompting

# Instruct models
python qxc_main_RSA.py --qwen
python qxc_main_RSA.py --llama
python qxc_main_RSA.py --mistral
python qxc_main_RSA.py --deepseek
python qxc_main_RSA.py --deepseek_v2_lite

# Base models (Exp 2.4 comparison)
python qxc_main_RSA_base.py --qwen
python qxc_main_RSA_base.py --llama
```

Output files (`hidden_states_<model>.npz`, `hidden_states_meta_<model>.csv`) land in `data/rdm/`.

---

## Step 3: Statistical analysis

All analysis scripts read from `data/` and write to `results/`. Run from the **repo root**.

### Experiment 1 — Output entropy (EAS)

Wilcoxon tests for EAS vs. NC baseline. Produces `results/full_analysis_wilcoxon.csv` and `results/full_analysis_report.txt`.

```bash
python scripts/statistical/full_analysis.py
```

### Experiment 2.1 — Generation dynamics (HL_rel, H_fb)

Trajectory metrics, Wilcoxon tests, and EAS/logit-scale trajectory plots (Figures 1–2 in paper). Produces files in `results/` and `results/plots/`.

Run once per model, pointing at its `extended_metrics_*.csv`:

```bash
python scripts/information_theory/entropy_trajectory_analysis.py \
    --csv data/model/deepseek/extended_metrics_deepseek.csv --out results

python scripts/information_theory/entropy_trajectory_analysis.py \
    --csv data/model/llama/extended_metrics_llama.csv --out results

python scripts/information_theory/entropy_trajectory_analysis.py \
    --csv data/model/mistral/extended_metrics_mistral.csv --out results

python scripts/information_theory/entropy_trajectory_analysis.py \
    --csv data/model/qwen/extended_metrics_qwen.csv --out results

python scripts/information_theory/entropy_trajectory_analysis.py \
    --csv data/model/deepseek/extended_metrics_deepseek_v2_lite.csv --out results
```

### Experiment 2.2 — Multilevel models (MLM)

Fits scalar MLMs for all DVs × models. Produces `results/MLM/analysis_base.csv`, `results/MLM/mlm_scalar_extended.csv`, and `results/MLM/mlm_per_layer.csv` (Table 3 in paper).

```bash
python scripts/information_theory/run_mlm_analysis.py \
    --data-dir data/model \
    --out-dir results/MLM
```

### Experiment 2.3 — CAA trajectories and beam diversity figures

Produces the bucket plots (Figure 3 in paper) in `results/plots/`.

```bash
python scripts/statistical/plot_buckets.py \
    --data-dir data/model \
    --out-dir results/plots
```

### Experiment 2.4 — RSA

Run the notebook `notebooks/rdm_analysis.ipynb`. It reads `data/rdm/hidden_states_*.npz` and `data/rdm/hidden_states_meta_*.csv` and writes RSA outputs to `results/rdm/rdm_sbert/`, including `rdm_results.csv` and `rdm_trajectory.png` (Figure 4 in paper).

The notebook can be run locally or on Google Colab (GPU recommended for the bootstrap step).

---

## Expected outputs

After running all steps, the following files should exist:

```
results/
├── full_analysis_wilcoxon.csv          # Exp 1: EAS Wilcoxon p-values (Table 1)
├── entropy_trajectory_wilcoxon.csv     # Exp 2.1: HL_rel / H_fb Wilcoxon (Table 2)
├── plots/
│   ├── entropy_trajectory_eas_*.png    # Fig 1 (one per model)
│   └── entropy_trajectory_logit_*.png  # Fig 2 (one per model)
├── MLM/
│   ├── mlm_scalar_extended.csv         # Exp 2.2: MLM coefficients (Table 3)
│   └── analysis_base.csv               # Per-prompt merged data used for MLMs
├── plots/buckets_*.png                 # Fig 3: CAA + beam diversity (one per model)
└── rdm/rdm_sbert/
    ├── rdm_results.csv                 # Exp 2.4: per-layer RSA ρ values (Table 4)
    └── rdm_trajectory.png              # Fig 4: partial Spearman by layer
```

---

## Hardware used

All model inference was run on a single NVIDIA GPU (≥16 GB VRAM). Statistical analyses (Steps 3–4) run on CPU and complete in under 10 minutes total. The RSA bootstrap (1000 iterations) takes approximately 20–40 minutes on CPU or 5–10 minutes on GPU.

---

## Citation

If you use this code or data, please cite the paper:

> Allahverdiyeva, A. & Harkness, A.C. (2026). LLM Detection of Polysemous Message Encoding: Pragmatic Context Modulates LLM Processing Uncertainty.
