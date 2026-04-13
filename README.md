# Antahkarana Reasoning Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> A novel cognitive architecture inspired by Vedantic philosophy for LLM reasoning and evaluation.

The **Antahkarana Reasoning Framework** introduces a multi-agent or multi-component approach to structural reasoning, evaluating Large Language Models across multiple datasets and standard baselines (Direct, CoT, Self-Consistency, ToT). 

## Architecture

The framework implements a cognitive pipeline consisting of interacting modules:
* `Manas` (Mind/Sensory Processing)
* `Chitta` (Memory/Subconscious)
* `Buddhi` (Intellect/Decision Making)
* `Ahamkara` (Ego/Identity)
* `Sakshi` (Witness/Meta-Cognition)

## Project Structure

* `LLM_Qwen_code_500_samples/` - Implementation using `Qwen/Qwen2.5-7B-Instruct` via vLLM with high-throughput evaluation.
* `VQA_code_1000_samples/` - Vision-Language evaluation pipeline.

## Features

* **Multiple Baselines**: Compares the Antahkarana method against standard prompting techniques (Direct, Chain-of-Thought, Tree-of-Thoughts, Self-Consistency).
* **High-Performance Evaluation**: Uses `vLLM` for continuous batching and high-throughput inference on NVIDIA GPUs.
* **Extensive Metrics**: Generates comprehensive evaluation metrics, ablations, and visualizations for IEEE paper standards.

## Execution

For detailed execution instructions, navigate to the specific model folder. Example for Qwen execution:

```bash
cd LLM_Qwen_code_500_samples/antahkarana_QWEN_500
python main.py --n-main 500 --n-ablation 100
```
