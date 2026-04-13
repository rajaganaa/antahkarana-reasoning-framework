"""
model.py — Antahkarana Cognitive Architecture
BLIP-2 (Salesforce/blip2-flan-t5-xl) VLM with TensorRT / FP16 optimization.
Handles vision + language encoding, multi-pass inference, GPU batching.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from utils import (
    ExperimentConfig,
    LOG,
    PassResult,
    Timer,
    gpu_memory_stats,
    gpu_utilization_pct,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_BLIP2_MODELS = {
    "small": "Salesforce/blip2-opt-2.7b",
    "medium": "Salesforce/blip2-flan-t5-xl",
    "large": "Salesforce/blip2-flan-t5-xxl",
}
_DEFAULT_MODEL = "Salesforce/blip2-flan-t5-xl"

# ─────────────────────────────────────────────────────────────────────────────
# TensorRT optimization wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TensorRTOptimizer:
    """
    Applies TensorRT optimisation to a PyTorch vision encoder via Torch-TensorRT.
    Falls back gracefully to FP16 if TRT is unavailable.
    """

    def __init__(
        self,
        workspace_gb: float = 6.0,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.workspace_gb = workspace_gb
        self.device = device
        self._trt_available = self._check_trt()

    def _check_trt(self) -> bool:
        # Catch both ImportError and OSError (e.g. libcudart.so.13 not found)
        try:
            import torch_tensorrt  # noqa: F401
            LOG.info("[TRT] torch-tensorrt available ✓")
            return True
        except (ImportError, OSError) as e:
            LOG.warning(f"[TRT] torch-tensorrt unavailable ({e}) — using FP16 fallback")
        try:
            import tensorrt  # noqa: F401
            LOG.info("[TRT] tensorrt (direct) available ✓")
            return True
        except (ImportError, OSError) as e:
            LOG.warning(f"[TRT] TensorRT not found ({e}) — using FP16 fallback")
        return False

    def optimize_vision_encoder(
        self,
        encoder: nn.Module,
        example_input: torch.Tensor,
    ) -> nn.Module:
        """
        Compile the vision encoder with Torch-TensorRT.
        Falls back to .half() on the original encoder if TRT unavailable.
        """
        if not self._trt_available or not torch.cuda.is_available():
            LOG.info("[TRT] FP16 path — vision encoder stays as torch.float16")
            return encoder.half().to(self.device)

        try:
            import torch_tensorrt  # noqa: F401 — may raise OSError if CUDA libs missing

            LOG.info("[TRT] Compiling vision encoder with Torch-TensorRT…")
            encoder = encoder.half().to(self.device).eval()

            # Dynamic shape: handle variable batch sizes 1–8
            trt_model = torch_tensorrt.compile(
                encoder,
                ir="torch_compile",
                inputs=[
                    torch_tensorrt.Input(
                        min_shape=[1, *example_input.shape[1:]],
                        opt_shape=[4, *example_input.shape[1:]],
                        max_shape=[8, *example_input.shape[1:]],
                        dtype=torch.float16,
                    )
                ],
                enabled_precisions={torch.float16},
                workspace_size=int(self.workspace_gb * 1024**3),
                truncate_long_and_double=True,
                require_full_compilation=False,
            )
            LOG.info("[TRT] Vision encoder compiled ✓")
            return trt_model

        except (Exception, OSError) as e:
            LOG.warning(f"[TRT] Compilation failed ({e}) — using FP16 fallback")
            return encoder.half().to(self.device)

    def optimize_via_onnx(
        self,
        encoder: nn.Module,
        example_input: torch.Tensor,
        onnx_path: str,
    ) -> Optional[Any]:
        """
        Alternative TRT path: export to ONNX then load TRT engine.
        Returns an onnxruntime InferenceSession or None on failure.
        """
        try:
            import onnx
            import onnxruntime as ort

            onnx_path = Path(onnx_path)
            onnx_path.parent.mkdir(parents=True, exist_ok=True)

            LOG.info(f"[TRT/ONNX] Exporting vision encoder to {onnx_path}…")
            encoder_cpu = encoder.float().cpu().eval()

            torch.onnx.export(
                encoder_cpu,
                example_input.float().cpu(),
                str(onnx_path),
                opset_version=17,
                input_names=["pixel_values"],
                output_names=["image_embeds"],
                dynamic_axes={"pixel_values": {0: "batch_size"}},
                do_constant_folding=True,
            )

            # Validate
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            LOG.info("[TRT/ONNX] ONNX export validated ✓")

            # ORT session with TRT or CUDA EP
            providers = []
            if torch.cuda.is_available():
                providers.append(
                    (
                        "TensorrtExecutionProvider",
                        {
                            "trt_fp16_enable": True,
                            "trt_max_workspace_size": int(self.workspace_gb * 1024**3),
                        },
                    )
                )
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            sess = ort.InferenceSession(str(onnx_path), providers=providers)
            LOG.info(
                f"[TRT/ONNX] ORT session created | EP: {sess.get_providers()[0]}"
            )
            return sess

        except Exception as e:
            LOG.warning(f"[TRT/ONNX] ONNX path failed ({e})")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(question: str, instruction: str, prior_answer: str = "") -> str:
    """Construct a VQA prompt for BLIP-2 / FlanT5 backend."""
    if prior_answer:
        return (
            f"Question: {question}\n"
            f"Previous answer: {prior_answer}\n"
            f"{instruction}"
        )
    return f"Question: {question}\n{instruction}"


def _build_pass2_prompt(question: str, prior_answer: str) -> str:
    """
    Pass 2: Force answer extraction — never ask yes/no.
    The model must return the correct short answer directly,
    not confirm or deny the previous answer.
    """
    return (
        f"Question: {question}\n"
        f"Look at the image carefully. Give the correct answer in 1-2 words only.\n"
        f"Previous answer was: {prior_answer}\n"
        f"Correct answer:"
    )


def _build_pass3_prompt(
    question: str,
    answer1: str,
    answer2: str,
    instruction: str,
) -> str:
    """Pass 3: choose better answer — forces a concrete word, not a/b/yes/no."""
    return (
        f"Question: {question}\n"
        f"Option A: {answer1}\n"
        f"Option B: {answer2}\n"
        f"Which option correctly answers the question? "
        f"Reply with only the answer word — 1-2 words."
    )


def _extract_final_answer(raw: str) -> str:
    """Extract answer after known marker prefixes, or return first line."""
    for line in raw.split("\n"):
        line = line.strip()
        if line.upper().startswith("FINAL:"):
            return line[6:].strip()
        if line.lower().startswith("correct answer:"):
            return line[15:].strip()
    return raw.strip().split("\n")[0].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Core VLM Wrapper (BLIP-2)
# ─────────────────────────────────────────────────────────────────────────────

class AntahkaranaVLM:
    """
    BLIP-2 VLM with optional TensorRT vision encoder optimisation.

    Usage:
        vlm = AntahkaranaVLM(cfg)
        vlm.load()
        answer = vlm.infer_single(image, "What is in the image?")
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[cfg.dtype]
        self.model_name = cfg.model_name
        self._loaded = False
        self.processor = None
        self.model = None
        self._trt_optimizer = TensorRTOptimizer(
            workspace_gb=cfg.trt_workspace_gb,
            device=self.device,
        )
        self.call_count = 0

    def _clear_hf_cache(self) -> None:
        """Delete cached model files for the current model_name to force re-download."""
        import shutil
        cache_root = Path(self.cfg.cache_dir) / "models"
        # HuggingFace stores models as models--org--name
        slug = self.model_name.replace("/", "--")
        for candidate in [
            cache_root / f"models--{slug}",
            Path.home() / ".cache" / "huggingface" / "hub" / f"models--{slug}",
        ]:
            if candidate.exists():
                LOG.warning(f"[VLM] Removing cache: {candidate}")
                shutil.rmtree(candidate, ignore_errors=True)

    # ── Load ────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load BLIP-2 model and processor; apply TRT optimisation."""
        if self._loaded:
            return

        LOG.info(f"[VLM] Loading {self.model_name} …")
        mem_before = gpu_memory_stats()

        try:
            self._load_blip2()
        except Exception as e:
            err_str = str(e)
            # Corrupted tokenizer.json in HF cache — delete and retry same model
            if "ModelWrapper" in err_str or "untagged enum" in err_str:
                LOG.warning(
                    f"[VLM] Corrupted tokenizer cache detected ({e}). "
                    "Clearing HF cache and retrying…"
                )
                self._clear_hf_cache()
                try:
                    self._load_blip2()
                except Exception as e2:
                    LOG.warning(f"[VLM] Retry failed ({e2}) — falling back to OPT variant")
                    self.model_name = _BLIP2_MODELS["small"]
                    self._clear_hf_cache()
                    self._load_blip2()
            else:
                LOG.warning(f"[VLM] BLIP-2 load failed ({e}) — trying OPT variant")
                self.model_name = _BLIP2_MODELS["small"]
                self._load_blip2()

        if self.cfg.use_tensorrt and torch.cuda.is_available():
            self._apply_trt()

        self._loaded = True
        mem_after = gpu_memory_stats()
        LOG.info(
            f"[VLM] Loaded | "
            f"VRAM: {mem_before['free_gb']:.1f}GB → {mem_after['free_gb']:.1f}GB free"
        )

    def _load_blip2(self) -> None:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor  # type: ignore

        cache = Path(self.cfg.cache_dir) / "models"
        cache.mkdir(parents=True, exist_ok=True)

        # transformers 4.40.x ships processor_config.json with 'num_query_tokens'
        # which Blip2Processor.__init__ does not accept — strip it out explicitly.
        try:
            self.processor = Blip2Processor.from_pretrained(
                self.model_name,
                cache_dir=str(cache),
            )
        except TypeError as e:
            if "num_query_tokens" in str(e) or "unexpected keyword argument" in str(e):
                LOG.warning(
                    f"[VLM] Processor config has unsupported keys ({e}). "
                    "Retrying with ignore_model_input_names workaround…"
                )
                # Load image processor and tokenizer separately, bypassing bad config keys
                from transformers import AutoTokenizer, AutoProcessor  # type: ignore
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        cache_dir=str(cache),
                        trust_remote_code=False,
                    )
                except Exception:
                    # Last resort: build processor manually from components
                    from transformers import Blip2Processor, BlipImageProcessor  # type: ignore
                    image_processor = BlipImageProcessor.from_pretrained(
                        self.model_name, cache_dir=str(cache)
                    )
                    # Use T5 tokenizer for flan-t5 models, GPT2 for OPT models
                    if "opt" in self.model_name.lower():
                        from transformers import GPT2Tokenizer  # type: ignore
                        tokenizer = GPT2Tokenizer.from_pretrained(
                            "facebook/opt-2.7b", cache_dir=str(cache)
                        )
                    else:
                        from transformers import T5TokenizerFast  # type: ignore
                        tokenizer = T5TokenizerFast.from_pretrained(
                            "google/flan-t5-xl", cache_dir=str(cache)
                        )
                    self.processor = Blip2Processor(
                        image_processor=image_processor,
                        tokenizer=tokenizer,
                    )
                LOG.info("[VLM] Processor loaded via fallback path ✓")
            else:
                raise

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir=str(cache),
            ignore_mismatched_sizes=True,
        )
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)

        self.model.eval()
        LOG.info(f"[VLM] Model dtype: {self.dtype} | device_map: auto")

    def _apply_trt(self) -> None:
        """Attempt to TRT-compile the BLIP-2 vision encoder (Q-Former input)."""
        try:
            vision_encoder = self.model.vision_model
            # Build example pixel tensor matching BLIP-2 processor output (224x224)
            example = torch.randn(1, 3, 224, 224, dtype=torch.float16, device=self.device)
            compiled = self._trt_optimizer.optimize_vision_encoder(vision_encoder, example)
            self.model.vision_model = compiled
            LOG.info("[VLM] TRT vision encoder applied ✓")
        except Exception as e:
            LOG.warning(f"[VLM] TRT application failed ({e}) — continuing without TRT")

    # ── Single inference ─────────────────────────────────────────────────────

    @torch.inference_mode()
    def infer_single(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 64,
        temperature_sampling: bool = False,
    ) -> Tuple[str, float]:
        """
        Run a single forward pass.

        temperature_sampling=True uses do_sample=True, temperature=0.7 for
        self-consistency (Wang et al., 2022) instead of image-noise perturbation.

        Returns:
            answer: decoded text
            confidence: approximate confidence (mean token probability)
        """
        assert self._loaded, "Call .load() first"
        self.call_count += 1

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype if self.dtype != torch.float32 else None)

        # Cast input_ids to long (they must stay int)
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].long()

        if temperature_sampling:
            # BUG 10 FIX: Standard self-consistency via temperature sampling
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                output_scores=True,
                return_dict_in_generate=True,
            )
        else:
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                num_beams=3,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=torch.cuda.is_available()):
            outputs = self.model.generate(**inputs, **gen_kwargs)

        answer = self.processor.decode(
            outputs.sequences[0], skip_special_tokens=True
        ).strip()

        # Approximate confidence from beam scores
        confidence = self._beam_confidence(outputs)
        return answer, confidence

    def _beam_confidence(self, outputs: Any) -> float:
        """Estimate confidence from generation scores (mean softmax entropy proxy)."""
        try:
            scores = outputs.scores  # Tuple[Tensor(batch, vocab)]
            if not scores:
                return 0.5
            probs = [torch.softmax(s[0].float(), dim=-1).max().item() for s in scores[:10]]
            return float(sum(probs) / len(probs))
        except Exception:
            return 0.5

    def _beam_confidence_for_sample(self, outputs: Any, sample_idx: int) -> float:
        """
        BUG 5 FIX: Extract per-sample confidence from batch generation scores.
        outputs.scores is a tuple of (n_steps,) tensors each shaped (batch, vocab).
        We index dimension 0 with sample_idx to get that sample's token probs.
        """
        try:
            scores = outputs.scores  # Tuple[Tensor(batch_size, vocab_size)]
            if not scores:
                return 0.5
            probs = [
                torch.softmax(s[sample_idx].float(), dim=-1).max().item()
                for s in scores[:10]
                if s.shape[0] > sample_idx
            ]
            return float(sum(probs) / len(probs)) if probs else 0.5
        except Exception:
            return 0.5

    # ── Batch inference ──────────────────────────────────────────────────────

    @torch.inference_mode()
    def infer_batch(
        self,
        images: List[Image.Image],
        prompts: List[str],
        max_new_tokens: int = 64,
    ) -> List[Tuple[str, float]]:
        """
        Batch inference for throughput efficiency.
        Falls back to sequential if OOM.
        """
        assert self._loaded
        if not images:
            return []

        try:
            return self._batch_forward(images, prompts, max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            LOG.warning("[VLM] OOM in batch — falling back to sequential")
            torch.cuda.empty_cache()
            gc.collect()
            return [
                self.infer_single(img, p, max_new_tokens)
                for img, p in zip(images, prompts)
            ]

    @torch.inference_mode()
    def _batch_forward(
        self,
        images: List[Image.Image],
        prompts: List[str],
        max_new_tokens: int,
    ) -> List[Tuple[str, float]]:
        self.call_count += len(images)

        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].long()

        # Cast pixel_values to model dtype
        if "pixel_values" in inputs and self.dtype != torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=torch.cuda.is_available()):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=3,
                early_stopping=True,
                output_scores=True,              # BUG 5 FIX: request generation scores
                return_dict_in_generate=True,    # BUG 5 FIX: return structured output
            )

        results = []
        for i, seq in enumerate(outputs.sequences):
            ans = self.processor.decode(seq, skip_special_tokens=True).strip()
            # BUG 5 FIX: compute per-sample confidence from beam scores instead of
            # hardcoding 0.5.  The previous code returned (ans, 0.5) for every sample,
            # meaning batch-processed samples could never be flagged as hallucinations
            # (threshold is 0.67).  We now extract the same confidence metric used
            # in infer_single() via _beam_confidence_for_sample().
            confidence = self._beam_confidence_for_sample(outputs, sample_idx=i)
            results.append((ans, confidence))
        return results

    # ── Utility ──────────────────────────────────────────────────────────────

    def warmup(self, num_steps: int = 3) -> None:
        """GPU warmup to stabilise timing benchmarks."""
        LOG.info(f"[VLM] Warming up GPU ({num_steps} steps)…")
        dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        for _ in range(num_steps):
            self.infer_single(dummy_img, "What is this?", max_new_tokens=10)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.call_count = 0   # reset after warmup
        LOG.info("[VLM] Warmup complete")

    def model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {}
        params = sum(p.numel() for p in self.model.parameters())
        return {
            "name": self.model_name,
            "parameters_M": round(params / 1e6, 1),
            "dtype": str(self.dtype),
            "device": str(self.device),
        }

    def reset_call_count(self) -> None:
        self.call_count = 0


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build a PassResult from an inference call
# ─────────────────────────────────────────────────────────────────────────────

def run_pass(
    vlm: AntahkaranaVLM,
    pass_id: int,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 64,
    temperature_sampling: bool = False,
) -> "PassResult":
    """
    Execute a single reasoning pass and return a PassResult.

    temperature_sampling=True uses do_sample=True, temperature=0.7 for
    self-consistency diversity (Wang et al., 2022).  This is the standard
    approach; image-noise perturbation is NOT used.
    """
    gpu_before = gpu_utilization_pct()
    mem_before = gpu_memory_stats().get("allocated_gb", 0.0)

    with Timer(f"pass{pass_id}") as t:
        answer, confidence = vlm.infer_single(
            image, prompt, max_new_tokens,
            temperature_sampling=temperature_sampling,
        )

    gpu_util = (gpu_utilization_pct() + gpu_before) / 2
    mem_used = gpu_memory_stats().get("allocated_gb", 0.0)

    return PassResult(
        pass_id=pass_id,
        answer=answer,
        latency_s=t.elapsed,
        gpu_util_pct=gpu_util,
        gpu_mem_gb=mem_used,
    )
