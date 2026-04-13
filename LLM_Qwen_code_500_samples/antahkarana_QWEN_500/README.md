# Antahkarana v11 — IEEE Evaluation Pipeline

**Model**: Qwen/Qwen2.5-7B-Instruct
**Hardware**: NVIDIA L4 24GB · 32 vCPUs · 128GB RAM
**Engine**: vLLM (continuous batching · BFloat16 · CUDA graphs)

> This is the **Qwen2.5-7B-Instruct** version of the Antahkarana pipeline.
> Prior versions: LLAMA (`antahkarana_v12_fixed`) · Mistral (`antahkarana_MISTRAL_500`).
> All logic, evaluation, and output formats are identical — only the model changes.

---

## Project Structure

```
antahkarana_QWEN_500/
├── main.py                  # Full pipeline entry point
├── vllm_engine.py           # Singleton vLLM engine (loads model ONCE)
├── Antahkarana_v11.ipynb    # JupyterLab version (cell-by-cell)
├── antahkarana/system.py    # Manas-Chitta-Buddhi-Ahamkara-Sakshi
├── baselines/               # direct, cot, self_consistency, tot
├── evaluation/              # metrics, ablation, visualize
├── hf_datasets/             # loader for all 5 datasets
└── datasets/cache/          # pre-cached 100/500 sample JSON files
```

---

## What Changed vs Mistral Version

| File | Change |
|------|--------|
| `vllm_engine.py` | `MODEL_ID` → `Qwen/Qwen2.5-7B-Instruct` |
| `vllm_engine.py` | `apply_chat_template` comment updated (Qwen uses ChatML with dedicated system role) |
| `vllm_engine.py` | `trust_remote_code=True` noted as required for Qwen2.5 |
| `main.py` | Report header updated: `Qwen2.5-7B-Instruct` |
| `antahkarana/system.py` | Comments cleaned (Mistral-specific notes updated); all prompts unchanged |
| `baselines/prompts.py` | Module docstring updated; all prompts identical to Mistral version |
| `README.md` | This file |

**Zero prompt changes** — Qwen2.5 has strong ChatML instruction following.
All structured-format prompts (VERIFY, PRAMANA, MULTIHOP, ToT) that were
fixed for Mistral work correctly on Qwen2.5 without further modification.

---

## Qwen2.5 Chat Format

Qwen2.5-7B-Instruct uses **ChatML**:
```
<|im_start|>system
{system prompt}
<|im_end|>
<|im_start|>user
{user message}
<|im_end|>
<|im_start|>assistant
```
Unlike Mistral (which merges system into user turn), Qwen2.5 has a fully
dedicated system token. The tokenizer's `apply_chat_template()` handles this
automatically — no code changes needed.

---

## Setup

```bash
pip install vllm
pip install datasets transformers accelerate
pip install scipy matplotlib numpy sentence-transformers
# No HF_TOKEN needed — Qwen2.5-7B-Instruct is publicly available
```

---

## Execution

### 500-sample experiment
```bash
cd antahkarana_QWEN_500
python main.py --n-main 500 --n-ablation 100
```

### 100-sample run
```bash
python main.py --n-main 100 --n-ablation 50
```

### Override model at runtime
```bash
MODEL_ID=Qwen/Qwen2.5-14B-Instruct python main.py --n-main 500 --n-ablation 100
```

---

## Expected Runtime (L4, 7B model)

| Method           | Throughput    |
|------------------|---------------|
| Direct           | ~13-16 samp/s |
| CoT              | ~7-9 samp/s   |
| Self-Consistency | ~2-3 samp/s   |
| ToT              | ~6-8 samp/s   |
| Antahkarana      | ~3-5 samp/s   |

Total (500 samples × 5 datasets × 5 methods + ablation): ~3–4 hours
