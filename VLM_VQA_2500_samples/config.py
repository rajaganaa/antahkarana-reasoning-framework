"""
ANTAHKARANA v7 — Configuration
All hyperparameters, paths, and experiment settings.
"""

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

# ─── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ─── Environment Flags ──────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL']              = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']         = 'true'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM']            = 'false'

# ─── GPU Setup ──────────────────────────────────────────────────────────────────
cudnn.benchmark     = True   # cache best cuDNN kernel — ~5% speedup
cudnn.deterministic = False

DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_GPU         = torch.cuda.device_count()
TORCH_VERSION = torch.__version__

if torch.cuda.is_available():
    _props     = torch.cuda.get_device_properties(0)
    GPU_NAME   = _props.name
    GPU_MEM_GB = _props.total_memory / 1e9
    # FIX #9: Cap batch size — BLIP-2 XL fp16 ~150MB/image; 24GB → safe limit is 64
    if GPU_MEM_GB >= 22:
        BATCH_SIZE = 64
    elif GPU_MEM_GB >= 14:
        BATCH_SIZE = 32
    else:
        BATCH_SIZE = 16
else:
    GPU_NAME, GPU_MEM_GB, BATCH_SIZE = 'CPU', 0, 8

# ─── Experiment Settings ────────────────────────────────────────────────────────
NUM_DATASETS         = 5
SAMPLES_PER_DATASET  = 500   # 500/dataset × 5 = 2500 total for scaled experiments
SAMPLES_PER_DATASET_DEFAULT = 200  # Original notebook default: 200/dataset × 5 = 1000

# ─── Model IDs ──────────────────────────────────────────────────────────────────
BLIP2_MODEL_ID = 'Salesforce/blip2-flan-t5-xl'
EMBED_MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'

# ─── Hyperparameters ────────────────────────────────────────────────────────────
MAX_NEW_TOKENS  = 40    # FIX C: allow longer answers (was 30, too short)
SC_N_PASSES     = 5
SC_TEMPERATURE  = 0.45  # V8-R: lower temp → tighter SC voting clusters (was 0.55)
TOP_K_RETRIEVAL = 5
VIS_BETA        = 0.20
ENTITY_LAMBDA   = 0.15
DEPTH_CAP       = 2

# ─── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.resolve()
BASE_DIR    = PROJECT_DIR / 'outputs'
RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = BASE_DIR / 'plots'
DATA_RAW    = PROJECT_DIR / 'data' / 'raw'
DATA_PROC   = PROJECT_DIR / 'data' / 'processed'
EXP_LOGS    = PROJECT_DIR / 'experiments' / 'logs'
EXP_RESULTS = PROJECT_DIR / 'experiments' / 'results'
NOTEBOOKS   = PROJECT_DIR / 'notebooks'

# Create directories
for d in [RESULTS_DIR, FIGURES_DIR, DATA_RAW, DATA_PROC, EXP_LOGS, EXP_RESULTS, NOTEBOOKS]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Manas Routing Keywords ─────────────────────────────────────────────────────
VISUAL_KW  = {'color','colour','shape','texture','wearing','holding',
               'background','foreground','depicted','shown'}
TEXT_KW    = {'read','written','text','word','letter','sign','number','digit',
               'label','caption','says','printed','spell'}
MATH_KW    = {'calculate','compute','ratio','average','how many','count',
               'total','sum','difference','percent','fraction','multiply'}
COMPARE_KW = {'older','younger','higher','lower','taller','shorter',
               'earlier','later','bigger','smaller','more','less','better'}
VERIFY_KW  = {'support','refute','verify','fact','true','false','correct',
               'incorrect','claim','statement'}
MCHOICE_KW = {'which of','which option','select the','choose the','a)','b)','c)','d)'}
# FIX G: removed forced scienceqa→mchoice override; let manas route naturally
DATASET_OVERRIDES = {}

# ─── HuggingFace Token ──────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get('HF_TOKEN', 'hf_REPLACE_WITH_YOUR_TOKEN')
os.environ['HF_TOKEN'] = HF_TOKEN
