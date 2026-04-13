"""
vllm_engine.py — Singleton vLLM engine with continuous batching.
Hardware target: NVIDIA L4 24 GB, single GPU.
All Antahkarana inference routes through this module.

Model: Qwen/Qwen2.5-7B-Instruct
"""

import os
import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Model identifier ───────────────────────────────────────────────────────
# Changed from mistralai/Mistral-7B-Instruct-v0.3
# Qwen2.5-7B-Instruct is publicly available on HuggingFace — no gating.
# Native context window: 128K tokens. We cap at 4096 to match L4 budget.
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


# ── Per-method sampling configurations ────────────────────────────────────
# Identical to Mistral version. Qwen2.5 uses the same sampling space.
# One difference: Qwen2.5 responds well to temperature=0.0 (greedy) and
# does NOT need top_p clamping — kept at 0.95 for consistency.

@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 512
    n: int = 1
    stop: List[str] = field(default_factory=list)


METHOD_CONFIGS: Dict[str, SamplingConfig] = {
    "direct":           SamplingConfig(temperature=0.0, max_tokens=256),
    "cot":              SamplingConfig(temperature=0.0, max_tokens=1024),
    "self_consistency": SamplingConfig(temperature=0.7, max_tokens=512, n=5),
    "tot":              SamplingConfig(temperature=0.3, max_tokens=768),
    "antahkarana":      SamplingConfig(temperature=0.0, max_tokens=1536),
    "ablation":         SamplingConfig(temperature=0.0, max_tokens=512),
}


class VLLMEngine:
    """
    Singleton wrapper around vLLM's LLM class.
    Acquire via VLLMEngine.get(); never instantiate directly a second time.
    """

    _instance: Optional["VLLMEngine"] = None

    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self._load_time: float = 0.0
        self._init_engine()

    @classmethod
    def get(cls) -> "VLLMEngine":
        if cls._instance is None:
            logger.info("Initialising vLLM engine (first call)...")
            cls._instance = cls()
        return cls._instance

    # ── Engine initialisation ──────────────────────────────────────────────

    def _init_engine(self):
        try:
            from vllm import LLM  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "vLLM not installed. Install with:\n"
                "  pip install vllm\n"
                "Then restart the kernel."
            )

        from vllm import LLM
        from transformers import AutoTokenizer

        t0 = time.time()
        logger.info(f"Loading {MODEL_ID} via vLLM...")

        self.llm = LLM(
            model=MODEL_ID,
            tensor_parallel_size=1,          # single L4 GPU
            gpu_memory_utilization=0.92,      # ~22 GB of 24 GB for KV cache
            # Qwen2.5-7B supports 128K context natively.
            # Capped at 4096 to stay within L4 memory budget (matches prior runs).
            max_model_len=4096,
            trust_remote_code=True,           # required for Qwen2.5 custom code
            dtype="bfloat16",
            enforce_eager=False,              # CUDA graphs ON for throughput
            max_num_batched_tokens=8192,      # continuous batching window
            max_num_seqs=64,                  # max parallel sequences
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN or None,
            trust_remote_code=True,           # required for Qwen2.5 tokenizer
        )

        self._load_time = time.time() - t0
        logger.info(f"vLLM engine ready in {self._load_time:.1f}s")

    # ── Core batched generation ────────────────────────────────────────────

    def generate_batch(
        self,
        prompts: List[str],
        method: str = "direct",
        override: Optional[Dict[str, Any]] = None,
    ) -> List[List[str]]:
        """
        Generate completions for a list of raw-text prompts.

        Returns:
            List[List[str]] -- outer dim = prompt index,
                               inner dim = n samples per prompt.
        """
        from vllm import SamplingParams

        cfg = METHOD_CONFIGS.get(method, METHOD_CONFIGS["direct"])
        sp_kwargs: Dict[str, Any] = dict(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            n=cfg.n,
        )
        if cfg.stop:
            sp_kwargs["stop"] = cfg.stop
        if override:
            sp_kwargs.update(override)

        sampling_params = SamplingParams(**sp_kwargs)

        t0 = time.time()
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        elapsed = time.time() - t0

        results: List[List[str]] = []
        total_out_tokens = 0
        for req_out in outputs:
            texts = [o.text.strip() for o in req_out.outputs]
            results.append(texts)
            total_out_tokens += sum(len(o.token_ids) for o in req_out.outputs)

        n = len(prompts)
        logger.debug(
            f"Batch({method}) n={n} | {total_out_tokens} out-tokens | "
            f"{elapsed:.2f}s | {n/elapsed:.1f} samp/s"
        )
        return results

    def generate_single(
        self,
        prompt: str,
        method: str = "direct",
        override: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convenience: single prompt -> first sample string."""
        return self.generate_batch([prompt], method, override)[0][0]

    # ── Prompt formatting ──────────────────────────────────────────────────

    def apply_chat_template(
        self,
        system: str,
        user: str,
    ) -> str:
        """
        Format a system+user pair using Qwen2.5 chat template.

        Qwen2.5-7B-Instruct uses the ChatML format:
          <|im_start|>system\\n{system}<|im_end|>
          <|im_start|>user\\n{user}<|im_end|>
          <|im_start|>assistant\\n

        Unlike Mistral, Qwen2.5 has a fully dedicated system role token —
        no merging into the user turn. The tokenizer's apply_chat_template
        handles this correctly when passed:
          messages=[{"role":"system",...}, {"role":"user",...}]
        This is identical call signature to the Mistral version, so all
        prompt builders in system.py and baselines/prompts.py are unchanged.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def apply_messages_template(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


# ── Module-level helpers ───────────────────────────────────────────────────

def get_engine() -> VLLMEngine:
    return VLLMEngine.get()


def batch_infer(
    prompts: List[str],
    method: str = "direct",
    override: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return first sample per prompt (for n=1 methods)."""
    engine = get_engine()
    results = engine.generate_batch(prompts, method, override)
    return [r[0] for r in results]


def batch_infer_multi(
    prompts: List[str],
    method: str = "self_consistency",
    override: Optional[Dict[str, Any]] = None,
) -> List[List[str]]:
    """Return all n samples per prompt (for self-consistency)."""
    engine = get_engine()
    return engine.generate_batch(prompts, method, override)
