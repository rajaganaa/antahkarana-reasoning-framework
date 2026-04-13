"""
baselines/prompts.py — Prompt construction for all baseline methods.
All methods use chat-template formatting via the shared vLLM engine.

Qwen2.5-7B-Instruct notes:
- Native ChatML system role — no merging needed (unlike Mistral).
- Strong structured-format compliance: all prompts including ToT work
  correctly. The simplified ToT format introduced for Mistral is kept
  as it is cleaner and generalises well.
"""

from typing import List, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Context helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_context_hotpot(context: List[Dict]) -> str:
    """Format HotpotQA paragraph context."""
    parts = []
    for para in context[:8]:
        title = para.get("title", "")
        sents = para.get("sentences", [])
        text  = " ".join(sents) if isinstance(sents, list) else str(sents)
        parts.append(f"[{title}]: {text}")
    return "\n".join(parts)


def _format_choices_mmlu(choices: List[str]) -> str:
    labels = ["A", "B", "C", "D"]
    return "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))


# ─────────────────────────────────────────────────────────────────────────────
# DIRECT ANSWER
# ─────────────────────────────────────────────────────────────────────────────

DIRECT_SYSTEM = (
    "You are a concise, factual assistant. "
    "Answer the question in as few words as possible. "
    "Output ONLY the answer with no preamble."
)

def direct_prompt(sample: Dict, dataset: str, engine) -> str:
    if dataset == "hotpotqa":
        ctx = _format_context_hotpot(sample.get("context", []))
        user = f"Context:\n{ctx}\n\nQuestion: {sample['question']}\n\nAnswer:"
    elif dataset == "mmlu":
        user = (
            f"Question: {sample['question']}\n"
            f"{_format_choices_mmlu(sample['choices'])}\n\n"
            f"Answer with the letter only (A/B/C/D):"
        )
    elif dataset == "fever":
        user = f"{sample['question']}\nAnswer (SUPPORTS/REFUTES/NOT ENOUGH INFO):"
    else:
        user = f"Question: {sample['question']}\nAnswer:"

    return engine.apply_chat_template(DIRECT_SYSTEM, user)


# ─────────────────────────────────────────────────────────────────────────────
# CHAIN-OF-THOUGHT
# ─────────────────────────────────────────────────────────────────────────────

COT_SYSTEM = (
    "You are a careful reasoning assistant. "
    "Think step-by-step before answering. "
    "Format your response as:\n"
    "Reasoning: <step-by-step reasoning>\n"
    "ANSWER: <final answer only>"
)

COT_FEW_SHOT_HOTPOT = (
    "Example:\n"
    "Context:\n"
    "[Leon: The Professional]: 1994 thriller film directed by Luc Besson.\n"
    "[Luc Besson]: French filmmaker born in Paris.\n"
    "Question: What is the citizenship of the director of Leon: The Professional?\n"
    "Reasoning:\n"
    "Step 1: The question asks for the citizenship of the director of 'Leon: The Professional'.\n"
    "Step 2: From [Leon: The Professional], the director is Luc Besson.\n"
    "Step 3: From [Luc Besson], he is French (born in Paris, France).\n"
    "ANSWER: French\n\n"
)

def cot_prompt(sample: Dict, dataset: str, engine) -> str:
    if dataset == "hotpotqa":
        ctx = _format_context_hotpot(sample.get("context", []))
        user = (
            COT_FEW_SHOT_HOTPOT
            + f"Context:\n{ctx}\n\n"
            + f"Question: {sample['question']}\n\n"
            + "Reasoning:\n"
        )
    elif dataset == "mmlu":
        user = (
            f"Question: {sample['question']}\n"
            f"{_format_choices_mmlu(sample['choices'])}\n\n"
            "Reasoning:\n"
        )
    elif dataset == "svamp":
        user = (
            "Solve the math problem step by step.\n"
            f"Problem: {sample['question']}\n\n"
            "Reasoning:\n"
        )
    elif dataset == "fever":
        user = (
            f"{sample['question']}\n\n"
            "Reasoning: Assess the evidence step by step.\n"
        )
    else:
        user = f"Question: {sample['question']}\n\nReasoning:\n"

    return engine.apply_chat_template(COT_SYSTEM, user)


# ─────────────────────────────────────────────────────────────────────────────
# SELF-CONSISTENCY (uses cot prompt; SC aggregation done in runner)
# ─────────────────────────────────────────────────────────────────────────────

def sc_prompt(sample: Dict, dataset: str, engine) -> str:
    """Same prompt as CoT; diversity comes from temperature=0.7, n=5 sampling."""
    return cot_prompt(sample, dataset, engine)


# ─────────────────────────────────────────────────────────────────────────────
# TREE-OF-THOUGHT
#
# Simplified 2-step deliberation format introduced for Mistral compatibility.
# Kept for Qwen2.5 — cleaner and more reliable than the original 5-section
# PATH A/B/C format even on models with strong instruction following.
# ─────────────────────────────────────────────────────────────────────────────

TOT_SYSTEM = (
    "You are an expert deliberative reasoning assistant. "
    "Before answering, consider multiple approaches, then commit to the best one.\n\n"
    "Format your response exactly as:\n"
    "Consider 1: <first approach and conclusion>\n"
    "Consider 2: <second approach and conclusion>\n"
    "Consider 3: <third approach and conclusion>\n"
    "Best approach: <which and why in one sentence>\n"
    "ANSWER: <final answer only>"
)

def tot_prompt(sample: Dict, dataset: str, engine) -> str:
    if dataset == "hotpotqa":
        ctx = _format_context_hotpot(sample.get("context", []))
        user = f"Context:\n{ctx}\n\nQuestion: {sample['question']}"
    elif dataset == "mmlu":
        user = (
            f"Question: {sample['question']}\n"
            f"{_format_choices_mmlu(sample['choices'])}"
        )
    elif dataset == "svamp":
        user = f"Math Problem: {sample['question']}"
    elif dataset == "fever":
        user = sample["question"]
    else:
        user = f"Question: {sample['question']}"

    return engine.apply_chat_template(TOT_SYSTEM, user)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt dispatch
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_BUILDERS = {
    "direct":           direct_prompt,
    "cot":              cot_prompt,
    "self_consistency": sc_prompt,
    "tot":              tot_prompt,
}


def build_prompts_for_method(
    samples: List[Dict],
    dataset: str,
    method: str,
    engine,
) -> List[str]:
    """Build a list of formatted prompts for (samples x method x dataset)."""
    builder = PROMPT_BUILDERS.get(method)
    if builder is None:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(PROMPT_BUILDERS)}")
    return [builder(s, dataset, engine) for s in samples]
