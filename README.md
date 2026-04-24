# Antahkarana: Cognitively-Inspired Adaptive Reasoning for LLMs and VLMs

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Patent-IN%20202641043041-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Model-Qwen2.5--7B--Instruct-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/VLM-BLIP--2%20Flan--T5--XL-red?style=flat-square" />
  <img src="https://img.shields.io/badge/GPU-NVIDIA%20L4%2024GB-76b900?style=flat-square&logo=nvidia" />
</p>

> **IEEE Conference Submission** | SRM Institute of Science and Technology, Kattankulathur, India
> 
> *System design protected under Indian Patent Application No. 202641043041, filed April 3, 2026.*

---

## Abstract

Monolithic prompting strategies impose uniform computational budgets across all queries, irrespective of task difficulty or domain. **Antahkarana** is a five-module architecture that selectively activates verification and consistency passes based on question type and output-quality heuristics — reducing inference cost while preserving accuracy on tasks where expensive ensemble methods add value.

**Key results:**
- **+59.61%** accuracy on TruthfulQA over direct prompting (`p<0.001`)
- **+40.06%** exact match on SVAMP over direct prompting (`p<0.001`)
- **+250.82%** F1 over CoT on FEVER (`p<0.001`)
- **26% fewer** model calls than uniform Self-Consistency (2,219 vs 3,000 on VQA benchmarks)
- VQA hallucination reduced from **26.6% → 20.9%** at 79.1% exact match

---

## Architecture

The framework is named after a structured cognitive model from **Advaita Vedanta philosophy**, mapping five mental faculties to discrete computational modules:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│  MANAS  — 6-way Question-Type Router             │  ← simple / multihop / math /
│           keyword heuristics + dataset overrides │    verification / mchoice / comparison
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  CHITTA — In-Context Dense Retrieval             │  ← all-MiniLM-L6-v2 embeddings
│           cosine sim + entity overlap (λ=0.15)  │    top-k=5 passages
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  BUDDHI — Three-Pass Conditional Reasoning       │
│   Pass 1 (Tarka)   : base LLM inference          │  ← always active
│   Pass 2 (Pramana) : fact verification           │  ← FEVER only (~8% of samples)
│   Pass 3 (Samsaya) : self-consistency n=5        │  ← low-quality outputs (~12%)
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  AHAMKARA — Structured Output Layer              │  ← typed dict: predicted, confidence,
│             zero computational overhead          │    evidence_spans, latency, q_type
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  SAKSHI   — Non-Interfering Logger               │  ← zero accuracy impact (confirmed
│             writes to disk after inference       │    by ablation)
└─────────────────────────────────────────────────┘
```

For **VQA tasks**, Chitta extends to a dual-stream scorer combining textual similarity, ViT-G/14 visual embeddings (β=0.20), and entity overlap — with BLIP-2 Flan-T5-XL as the multimodal Tarka engine.

---

## Repository Structure

```
antahkarana-reasoning-framework/
│
├── LLM_Qwen_code_500_samples/          # NLP pipeline (Qwen2.5-7B-Instruct)
│   ├── antahkarana/
│   │   └── system.py                   # Core: Manas, Chitta, Buddhi, Ahamkara, Sakshi
│   ├── baselines/
│   │   ├── prompts.py                  # Direct, CoT, Self-Consistency, ToT prompts
│   │   └── runner.py                   # Baseline execution engine
│   ├── evaluation/
│   │   ├── metrics.py                  # Token-level F1, EM, accuracy, significance tests
│   │   ├── ablation.py                 # 5-module ablation framework
│   │   └── visualize.py                # Result plots (Figures 4–7 in paper)
│   ├── hf_datasets/
│   │   └── loader.py                   # HotpotQA, MMLU, FEVER, SVAMP, TruthfulQA
│   ├── main.py                         # Entry point: full pipeline + baselines
│   ├── vllm_engine.py                  # vLLM singleton (BF16, PagedAttention)
│   └── Antahkarana_v11.ipynb           # Jupyter notebook version
│
├── VQA_code_1000_samples/              # VQA pipeline (BLIP-2 Flan-T5-XL)
│   ├── main.py                         # Entry point: VQA full pipeline
│   ├── model.py                        # BLIP-2 + dual-stream Chitta + 3-pass Buddhi
│   ├── dataset.py                      # VQAv2, GQA, OK-VQA, TextVQA, ScienceQA loaders
│   ├── evaluator.py                    # Soft VQA accuracy, EM, hallucination rate
│   ├── utils.py                        # Shared utilities
│   └── requirements.txt                # VQA-specific dependencies
│
├── VLM_VQA_2500_samples/               # Large-scale VQA pipeline with V9/V10 fixes (2500 samples)
│   ├── main.py                         # Entry point: Enhanced IEEE metrics (95% CI, Pareto)
│   ├── src/                            # Refactored modular framework (Manas, Chitta, Buddhi)
│   ├── outputs/                        # Generated IEEE figures and efficiency Pareto tables
│   ├── experiments/                    # Structured experiment logs and configuration
│   └── notebooks/                      # Original research notebooks
│
├── .gitignore
└── README.md
```

---

## Datasets

### NLP Benchmarks (500 samples each)
| Dataset | Task | Metric |
|---------|------|--------|
| [HotpotQA](https://hotpotqa.github.io/) | Multi-hop reasoning | Token-level F1 |
| [MMLU](https://github.com/hendrycks/test) | 57-subject multiple choice | Accuracy |
| [TruthfulQA](https://github.com/sylinrl/TruthfulQA) | Factual calibration | Accuracy |
| [FEVER](https://fever.ai/) | Fact verification | F1 (SUPPORTS / REFUTES / NEI) |
| [SVAMP](https://github.com/arkilpatel/SVAMP) | Arithmetic word problems | Exact Match |

### VQA Benchmarks (Original 1000 samples & New 2500-sample scale-up)
| Dataset | Task | Samples (Orig) | Samples (New) |
|---------|------|----------------|---------------|
| [VQAv2](https://visualqa.org/) | General VQA (soft accuracy + EM) | 200 | 500 |
| [GQA](https://cs.stanford.edu/people/dorarad/gqa/) | Compositional reasoning | 200 | 500 |
| [OK-VQA](https://okvqa.allenai.org/) | Knowledge-grounded VQA | 200 | 500 |
| [TextVQA](https://textvqa.org/) | OCR + reasoning | 200 | 500 |
| [ScienceQA](https://scienceqa.github.io/) | Multimodal science QA | 200 | 500 |

---

## Results

### NLP Results (Table I in paper)
| Dataset | Antahkarana | Direct | CoT | Self-Cons. | ToT |
|---------|-------------|--------|-----|------------|-----|
| HotpotQA (F1) | 0.588 | 0.568 | 0.602 | **0.631** | 0.492 |
| MMLU (Acc) | 0.668 | 0.652 | 0.706 | **0.736** | 0.714 |
| TruthfulQA (Acc) | **0.267** | 0.167 | 0.264 | 0.259 | 0.327 |
| FEVER (F1) | **0.428** | 0.494 | 0.122 | 0.100 | 0.166 |
| SVAMP (EM) | **0.930** | 0.664 | 0.900 | 0.916 | 0.922 |

### VQA Results (Table II in paper - 1000 samples)
| Method | VQA Acc | Exact Match | Hallucination% | Latency (s) | Model Calls |
|--------|---------|-------------|----------------|-------------|-------------|
| **Full Antahkarana** | 72.6% | **79.1%** | **20.9%** | 0.690 | **2,219** |
| Single-Pass | 73.4% | 73.4% | 26.6% | **0.351** | 1,000 |
| CoT Baseline | 68.4% | 68.4% | 31.6% | 0.525 | 1,000 |
| Self-Consistency (3×) | **74.5%** | 74.5% | 15.6% | 0.900 | 3,000 |

### VLM VQA Scale-up Results (V9/V10 Architecture)
*Recent scale-up experiments using the updated V9/V10 architecture (featuring strict P2 junk-guards, MCQ hallucination resolution, and VQAv2 retrieval skip) yielded the following aggregate metrics:*
- **Overall Exact Match (EM):** ~43.3% 
- **VQAv2 EM:** ~67.2% (+5.6pp recovery over baseline)
- **ScienceQA Hallucination Rate:** Reduced from 94% → 48% (True false positives)
- All new results include **95% Bootstrap Confidence Intervals** and **Efficiency Pareto Trade-off Analysis**.

---

## Hardware & Software Requirements

```
GPU  : NVIDIA L4 24GB (Ampere, CUDA 12.1)  — minimum 16GB VRAM recommended
CPU  : 32 vCPUs
RAM  : 128GB
OS   : Ubuntu 20.04+
```

```
Python      >= 3.8
PyTorch     == 2.1.0
CUDA        == 12.1
vLLM        == 0.5.4   (NLP pipeline)
transformers>= 4.37.0
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/rajaganaa/antahkarana-reasoning-framework.git
cd antahkarana-reasoning-framework

# Install NLP dependencies
pip install vllm==0.5.4
pip install datasets transformers accelerate
pip install sentence-transformers scipy matplotlib numpy spacy
python -m spacy download en_core_web_sm

# Install VQA dependencies (Original)
cd VQA_code_1000_samples
pip install -r requirements.txt

# Install VQA dependencies (2500 Samples Scale-up)
cd ../VLM_VQA_2500_samples
pip install -r requirements.txt
```

> **Note:** No HuggingFace token required. `Qwen/Qwen2.5-7B-Instruct` and `Salesforce/blip2-flan-t5-xl` are publicly available.

---

## Usage

### NLP Pipeline (Qwen2.5-7B-Instruct)

```bash
cd LLM_Qwen_code_500_samples

# Full experiment: 500 samples + 100-sample ablation
python main.py --n-main 500 --n-ablation 100

# Quick test: 100 samples
python main.py --n-main 100 --n-ablation 50

# Override model at runtime
MODEL_ID=Qwen/Qwen2.5-14B-Instruct python main.py --n-main 500 --n-ablation 100
```

### VQA Pipeline (BLIP-2 Flan-T5-XL)

```bash
cd VQA_code_1000_samples

# Full VQA experiment: 1000 samples (200 per dataset)
python main.py --n-samples 200

# Run specific dataset only
python main.py --dataset vqav2 --n-samples 200
```

### VQA 2500-Sample IEEE Pipeline (V9/V10 Fixes)

```bash
cd VLM_VQA_2500_samples

# Full scale-up VQA experiment: 2500 samples (500 per dataset)
python main.py --samples_per_dataset 500

# Results and 95% bootstrap CIs will be saved to outputs/results/
```

### Jupyter Notebook

```bash
cd LLM_Qwen_code_500_samples
jupyter lab Antahkarana_v11.ipynb
```

### Expected Runtime (NVIDIA L4, 7B model)

| Method | Throughput |
|--------|-----------|
| Direct | ~13–16 samples/s |
| CoT | ~7–9 samples/s |
| Self-Consistency | ~2–3 samples/s |
| Antahkarana | ~3–5 samples/s |

> Full run (500 samples × 5 datasets × 5 methods + ablation): approximately **3–4 hours**

---

## Reproducibility

- **Random seeds:** 42 (data splits), 7 (subsets)
- **Model weights:** HuggingFace, commit `8f7a92c` (March 15, 2026)
- **No prompt tuning** on validation data
- All significance tests use **Welch's t-test** with **Bonferroni correction** (α=0.0025 for NLP, α=0.01 for VQA)

To reproduce Table I exactly:
```bash
cd LLM_Qwen_code_500_samples
python main.py --n-main 500 --n-ablation 100 --seed 42
# Results written to: results/processed/metrics_table.csv
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{rajaganapathy2026antahkarana,
  title     = {Antahkarana: Cognitively-Inspired Adaptive Reasoning for LLMs and VLMs},
  author    = {RajaGanapathy, M and Karthikeyan, H},
  journal   = {IEEE Conference Proceedings},
  year      = {2026},
  note      = {Indian Patent Application No. 202641043041}
}
```

---

## Authors

| Name | Role | Institution |
|------|------|-------------|
| **M RajaGanapathy** | Primary Author | Dept. of Computational Intelligence, SRM Institute of Science and Technology |
| **Dr. H Karthikeyan** | Advisor | Dept. of NWC, SRM Institute of Science and Technology |

---

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

The system architecture is protected under **Indian Patent Application No. 202641043041** (filed April 3, 2026, Office of the Controller General of Patents, Designs & Trade Marks, India).
