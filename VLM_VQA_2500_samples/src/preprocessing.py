"""
ANTAHKARANA v7 — Preprocessing
Model loading, visual embedding extraction, and BLIP-2 generation.
Exact logic preserved from original notebook Cell 5.
"""

import os
import torch
import numpy as np
from typing import List

import torchvision.transforms as T
if not hasattr(T.InterpolationMode, 'NEAREST_EXACT'):
    T.InterpolationMode.NEAREST_EXACT = T.InterpolationMode.NEAREST

from transformers import Blip2Processor, Blip2ForConditionalGeneration, logging as hf_logging
from sentence_transformers import SentenceTransformer
from PIL import Image

from .utils import postprocess_answer, free_gpu, gpu_mem_gb
from config import BLIP2_MODEL_ID, EMBED_MODEL_ID, DEVICE, MAX_NEW_TOKENS

hf_logging.set_verbosity_error()

# Module-level globals — initialized by init_models()
processor = None
blip2_model = None
embedder = None


def init_models():
    """Load BLIP-2 and SentenceTransformer models. Must be called before inference."""
    global processor, blip2_model, embedder

    print(f'Loading BLIP-2 ({BLIP2_MODEL_ID})...')
    processor = Blip2Processor.from_pretrained(BLIP2_MODEL_ID, token=os.environ.get('HF_TOKEN'))
    blip2_model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_MODEL_ID,
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True,
        token=os.environ.get('HF_TOKEN'),
    )
    blip2_model.eval()

    # FIX #8: torch.compile is incompatible with device_map='auto' (multi-device dispatch)
    # Only apply if model is on a single device
    try:
        if len(set(str(p.device) for p in blip2_model.parameters())) == 1:
            blip2_model = torch.compile(blip2_model, mode='reduce-overhead')
            print('  torch.compile enabled ✓')
        else:
            print('  torch.compile skipped (multi-device model — using device_map=auto)')
    except Exception as e:
        print(f'  torch.compile skipped: {e}')

    print(f'  BLIP-2 loaded | GPU mem: {gpu_mem_gb():.1f} GB')

    print(f'Loading sentence embedder ({EMBED_MODEL_ID})...')
    embedder = SentenceTransformer(EMBED_MODEL_ID)
    embedder.to(DEVICE)
    print('  Embedder loaded ✓')

    print('\n✅ Models ready.')


def get_visual_embedding(images: List[Image.Image]) -> np.ndarray:
    """Batch visual embedding extraction — no per-image overhead."""
    try:
        inputs       = processor(images=images, return_tensors='pt', padding=True)
        pixel_values = inputs['pixel_values'].to(DEVICE, torch.float16)
        with torch.no_grad():
            vis_feats = blip2_model.vision_model(pixel_values=pixel_values).last_hidden_state
            vis_emb   = vis_feats.mean(dim=1).cpu().float().numpy()
        return vis_emb
    except Exception as e:
        print(f'  visual_emb error: {e}')
        return np.zeros((len(images), 1408), dtype=np.float32)


def blip2_generate(
    images: List[Image.Image],
    prompts: List[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float  = 1.0,
    do_sample: bool     = False,
) -> List[str]:
    """
    TRUE BATCH generation — all images+prompts processed in one forward pass.
    FIX #5: all callers now pass full sub-batch lists, not single items.
    FIX #4: latency is measured at the caller level (per-sample wall clock).
    """
    if not images:
        return []
    inputs = processor(
        images=images, text=prompts, return_tensors='pt',
        padding=True, truncation=True, max_length=256,
    )
    inputs_on_device = {
        k: (v.to(DEVICE, torch.float16)
            if v.dtype in (torch.float32, torch.float64, torch.bfloat16)
            else v.to(DEVICE))
        for k, v in inputs.items()
    }
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_beams=1 if do_sample else 3,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs['temperature'] = temperature
    with torch.no_grad():
        out_ids = blip2_model.generate(**inputs_on_device, **gen_kwargs)
    raw = [o.strip() for o in processor.batch_decode(out_ids, skip_special_tokens=True)]
    return [postprocess_answer(o) for o in raw]
