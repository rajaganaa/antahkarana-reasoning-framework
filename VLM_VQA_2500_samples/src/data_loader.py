"""
ANTAHKARANA v7 — Dataset Loaders
VQAv2, GQA, OK-VQA, TextVQA, ScienceQA loaders.
Exact logic preserved from original notebook Cell 4.
"""

import os
import io
from typing import List

from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image


class VQASample:
    __slots__ = ['qid','question','image','answers','dataset_name','question_type',
                 'choices_str','choices_list']
    def __init__(self, qid, question, image, answers, dataset_name,
                 question_type='unknown', choices_str='', choices_list=None):
        self.qid           = str(qid)
        self.question      = str(question)
        self.image         = image
        self.answers       = [str(a) for a in answers if a]
        self.dataset_name  = dataset_name
        self.question_type = question_type
        self.choices_str   = choices_str
        self.choices_list  = choices_list or []


class VQADataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def pil_from_raw(raw_img):
    try:
        if isinstance(raw_img, Image.Image): return raw_img.convert('RGB')
        if isinstance(raw_img, dict):
            b = raw_img.get('bytes')
            if isinstance(b, bytes) and b: return Image.open(io.BytesIO(b)).convert('RGB')
            img_obj = raw_img.get('image')
            if isinstance(img_obj, Image.Image): return img_obj.convert('RGB')
        if isinstance(raw_img, (bytes, bytearray)) and raw_img:
            return Image.open(io.BytesIO(raw_img)).convert('RGB')
    except Exception: return None
    return None


def extract_vqav2_answers(raw_answers) -> List[str]:
    if not raw_answers: return []
    answers = []
    for item in raw_answers:
        if isinstance(item, dict):
            ans = item.get('answer', '')
            if ans: answers.append(str(ans))
        elif isinstance(item, str) and item: answers.append(item)
    return answers


def load_vqav2(n):
    print('  Loading VQAv2...', end=' ', flush=True)
    try:
        ds = load_dataset('lmms-lab/VQAv2', split='validation', streaming=True,
                          token=os.environ.get('HF_TOKEN'))
        samples = []
        for raw in ds:
            if len(samples) >= n: break
            img = pil_from_raw(raw.get('image'))
            if not img or not raw.get('question'): continue
            ans = extract_vqav2_answers(raw.get('answers', []))
            if not ans and raw.get('multiple_choice_answer'): ans = [raw['multiple_choice_answer']]
            if not ans: continue
            samples.append(VQASample(f'v2_{len(samples)}', raw['question'], img, ans, 'vqav2'))
        print(f'{len(samples)} samples ✓'); return samples
    except Exception as e:
        print(f'❌ VQAv2 failed: {e}'); return []


def load_gqa(n):
    print('  Loading GQA...', end=' ', flush=True)
    try:
        img_ds = load_dataset('lmms-lab/GQA', 'testdev_balanced_images',
                              split='testdev', streaming=True, token=os.environ.get('HF_TOKEN'))
        image_map = {}
        for row in img_ds:
            if len(image_map) >= n * 5: break
            img = pil_from_raw(row.get('image'))
            if img: image_map[str(row['id'])] = img
        if not image_map: raise ValueError('No GQA images loaded')
        qa_ds = load_dataset('lmms-lab/GQA', 'testdev_balanced_instructions',
                             split='testdev', streaming=True, token=os.environ.get('HF_TOKEN'))
        samples = []
        for row in qa_ds:
            if len(samples) >= n: break
            img = image_map.get(str(row.get('imageId', '')))
            q, a = row.get('question'), row.get('answer')
            if img and q and a:
                q_type = row['types'].get('semantic','unknown') if isinstance(row.get('types'),dict) else 'unknown'
                samples.append(VQASample(f'gqa_{len(samples)}', q, img, [a], 'gqa', q_type))
        print(f'{len(samples)} samples ✓'); return samples
    except Exception as e:
        print(f'❌ GQA failed: {e}'); return []


def load_okvqa(n):
    print('  Loading OK-VQA...', end=' ', flush=True)
    for hf_id, split in [('lmms-lab/OK-VQA','validation'), ('Multimodal-Fatima/OK-VQA_train','train')]:
        try:
            ds = load_dataset(hf_id, split=split, streaming=True, token=os.environ.get('HF_TOKEN'))
            samples = []
            for raw in ds:
                if len(samples) >= n: break
                img = pil_from_raw(raw.get('image'))
                if img and raw.get('question'):
                    raw_ans = raw.get('answers', [])
                    ans = (extract_vqav2_answers(raw_ans)
                           if raw_ans and isinstance(raw_ans[0], dict)
                           else [str(a) for a in raw_ans if a])
                    if not ans: continue
                    samples.append(VQASample(f'ok_{len(samples)}', raw['question'], img, ans, 'okvqa'))
            if samples:
                print(f'{len(samples)} samples ✓'); return samples
        except Exception:
            continue
    print('❌ OK-VQA failed'); return []


def load_textvqa(n):
    print('  Loading TextVQA...', end=' ', flush=True)
    try:
        ds = load_dataset('lmms-lab/textvqa', split='validation', streaming=True,
                          token=os.environ.get('HF_TOKEN'))
        samples = []
        for raw in ds:
            if len(samples) >= n: break
            img = pil_from_raw(raw.get('image'))
            if img and raw.get('question'):
                raw_ans = raw.get('answers', [])
                ans = (extract_vqav2_answers(raw_ans)
                       if raw_ans and isinstance(raw_ans[0], dict)
                       else [str(a) for a in raw_ans if a])
                if not ans: continue
                samples.append(VQASample(f'txt_{len(samples)}', raw['question'], img, ans, 'textvqa'))
        print(f'{len(samples)} samples ✓'); return samples
    except Exception as e:
        print(f'❌ TextVQA failed: {e}'); return []


def load_scienceqa(n):
    print('  Loading ScienceQA...', end=' ', flush=True)
    try:
        ds = load_dataset('derek-thomas/ScienceQA', split='validation', streaming=True,
                          token=os.environ.get('HF_TOKEN'))
        samples = []
        for raw in ds:
            if len(samples) >= n: break
            if not raw.get('image'): continue
            img     = pil_from_raw(raw['image'])
            choices = raw.get('choices', [])
            idx     = raw.get('answer', 0)
            ans     = [choices[idx]] if (choices and isinstance(idx, int) and 0 <= idx < len(choices)) else [str(idx)]
            if img and raw.get('question'):
                letters     = 'ABCDEFGH'
                choices_str = ' '.join(f'{letters[i]}) {c}' for i,c in enumerate(choices)) if choices else ''
                samples.append(VQASample(
                    f'sci_{len(samples)}', raw['question'], img, ans, 'scienceqa',
                    choices_str=choices_str, choices_list=list(choices)
                ))
        print(f'{len(samples)} samples ✓'); return samples
    except Exception as e:
        print(f'❌ ScienceQA failed: {e}'); return []


def load_all_datasets(samples_per_dataset: int):
    """Load all 5 datasets and return (all_samples, dataset_samples, full_dataset)."""
    dataset_list = [
        ('vqav2', load_vqav2), ('gqa', load_gqa), ('okvqa', load_okvqa),
        ('textvqa', load_textvqa), ('scienceqa', load_scienceqa),
    ]
    all_samples, dataset_samples = [], {}
    print('Loading all datasets...')
    for name, loader_fn in dataset_list:
        try:
            subset = loader_fn(samples_per_dataset)
            all_samples.extend(subset)
            dataset_samples[name] = subset
        except Exception as e:
            print(f'  ❌ {name} Error: {e}')
            dataset_samples[name] = []

    assert len(all_samples) > 0, 'No samples loaded!'
    print(f'\n✅ DATASETS LOADED: {len(all_samples)} total')
    print(f'   Per-dataset: { {k: len(v) for k,v in dataset_samples.items()} }')
    full_dataset = VQADataset(all_samples)
    return all_samples, dataset_samples, full_dataset
