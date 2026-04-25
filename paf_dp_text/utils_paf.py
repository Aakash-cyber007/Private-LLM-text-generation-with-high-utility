"""
utils_paf.py
============
Self-contained utilities for the PaF-DP-Text repo.
Covers: model registry, prompts, privacy accounting, clipping,
        embedding, dataset helpers, logging, and pickle I/O.
"""

import sys, math, torch, random, pickle, logging
import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────
MODELS = {
    'tinyllama1B': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'llama3.2-1B': 'meta-llama/Llama-3.2-1B-Instruct',
}

# ─────────────────────────────────────────────────────────────
# YELP LABELS  (10 categories × 5 stars = 50 classes)
# ─────────────────────────────────────────────────────────────
YELP_CATS = [
    'Business Category: Arts & Entertainment',
    'Business Category: Bars',
    'Business Category: Beauty & Spas',
    'Business Category: Event Planning & Services',
    'Business Category: Grocery',
    'Business Category: Health & Medical',
    'Business Category: Home & Garden',
    'Business Category: Hotels & Travel',
    'Business Category: Restaurants',
    'Business Category: Shopping',
]
YELP_STARS = [
    'Review Stars: 1.0', 'Review Stars: 2.0', 'Review Stars: 3.0',
    'Review Stars: 4.0', 'Review Stars: 5.0',
]

# ─────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────
_SYSTEM = "You are a helpful assistant."

_PRV_YELP = (
    'Here is a Yelp review with Business Category: {cat} and {stars}.\n'
    'Text: "{text}"\n'
    'Please write one similar review of a *fictional* business '
    'in the same category with the same star rating. '
    'Poor reviews have lower scores; excellent reviews have higher scores.'
)
_PUB_YELP = (
    'Please write a fake Yelp customer review of a fictional {cat} business '
    'with {stars}. Poor reviews have lower scores; excellent reviews have higher scores.'
)


def _apply_template(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def get_prv_prompt(text, tokenizer, cat='', stars='', **_):
    user = _PRV_YELP.format(cat=cat, stars=stars, text=text)
    return _apply_template(tokenizer, _SYSTEM, user)


def get_pub_prompt(tokenizer, cat='', stars='', **_):
    user = _PUB_YELP.format(cat=cat, stars=stars)
    return _apply_template(tokenizer, _SYSTEM, user)


# ─────────────────────────────────────────────────────────────
# BASIC SETUP
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_device(gpu: int = 0):
    if torch.cuda.is_available() and gpu >= 0:
        dev = f'cuda:{gpu}'
    else:
        dev = 'cpu'
    return torch.device(dev)


def setup_logging(filename: str, resume: bool = False):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fmt_c = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    fmt_f = logging.Formatter(
        "%(asctime)s %(name)-20s %(levelname)-8s %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_c)

    fh = logging.FileHandler(filename, mode='a' if resume else 'w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_f)

    # avoid duplicate handlers on re-import
    if not root.handlers:
        root.addHandler(ch)
        root.addHandler(fh)


def pickle_dump(var, path):
    with open(path, 'wb') as f:
        pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# TEXT UTILS
# ─────────────────────────────────────────────────────────────
def preprocess(s: str) -> str:
    return ' '.join(str(s).split())


def postprocess(s: str) -> str:
    return ' '.join(str(s).split())


# ─────────────────────────────────────────────────────────────
# DATASET BATCHING
# ─────────────────────────────────────────────────────────────
def batchify(lst, s: int, n: int):
    """Yield n consecutive non-overlapping slices of size s."""
    lst = list(lst)
    assert len(lst) >= n * s, (
        f"List has {len(lst)} items but need {n*s} for {n} batches of {s}")
    for i in range(n):
        yield lst[i * s: (i + 1) * s]


# ─────────────────────────────────────────────────────────────
# CLIPPING
# ─────────────────────────────────────────────────────────────
def clip_logit(logit: np.ndarray, publogit: np.ndarray, C: float) -> np.ndarray:
    """Difference clipping (DClip) from InvisibleInk."""
    return publogit + np.clip(logit - publogit, -C, C)


def get_topkplus_mask(pub: np.ndarray, k: int, clip: float, batch: int):
    """
    Top-k+ mask (InvisibleInk §3.3).
    Returns (boolean mask over vocab, indices of extension set).
    """
    vocab = len(pub)
    k = min(k, vocab)
    kthresh  = np.partition(pub, -k)[-k]
    k_ext    = kthresh - 2 * clip / batch
    mask     = pub >= k_ext
    ext_idxs = np.where(np.logical_and(pub >= k_ext, pub <= kthresh))[0]
    return mask, ext_idxs


# ─────────────────────────────────────────────────────────────
# PRIVACY ACCOUNTING  (zCDP → (ε,δ)-DP)
# ─────────────────────────────────────────────────────────────
def compute_rho(T: int, C: float, n: int, tau: float) -> float:
    """ρ-zCDP cost for T tokens, clip C, private batch n, temperature τ."""
    return T * 0.5 * (C / (n * tau)) ** 2


def cdp_delta(rho: float, eps: float) -> float:
    if rho == 0:
        return 0.0
    amin, amax = 1.01, (eps + 1) / (2 * rho) + 2
    for _ in range(1000):
        alpha = (amin + amax) / 2
        deriv = (2 * alpha - 1) * rho - eps + math.log1p(-1.0 / alpha)
        if deriv < 0:
            amin = alpha
        else:
            amax = alpha
    delta = math.exp(
        (alpha - 1) * (alpha * rho - eps) + alpha * math.log1p(-1 / alpha)
    ) / (alpha - 1.0)
    return min(delta, 1.0)


def cdp_eps(rho: float, delta: float = 1e-6) -> float:
    if delta >= 1 or rho == 0:
        return 0.0
    emin, emax = 0.0, rho + 2 * math.sqrt(rho * math.log(1 / delta))
    for _ in range(1000):
        eps = (emin + emax) / 2
        if cdp_delta(rho, eps) <= delta:
            emax = eps
        else:
            emin = eps
    return emax


def cdp_rho(eps: float, delta: float = 1e-6) -> float:
    if delta >= 1:
        return 0.0
    rmin, rmax = 0.0, eps + 1
    for _ in range(1000):
        rho = (rmin + rmax) / 2
        if cdp_delta(rho, eps) <= delta:
            rmin = rho
        else:
            rmax = rho
    return rmin


def get_clip(eps: float, T: int, tau: float, batch: int,
             delta: float = 1e-6) -> float:
    """Required clipping norm C to achieve (eps, delta)-DP."""
    rho_tot = cdp_rho(eps, delta)
    rho_tok = rho_tot / T
    if rho_tok <= 0:
        return np.inf
    # sensitivity = 2C / (n * tau)  →  C = n * tau * sqrt(2 * rho_tok / 1)
    n = batch - 1   # private batch size
    return n * tau * math.sqrt(2 * rho_tok)


def get_epsilon(T: int, C: float, batch: int, tau: float,
                delta: float = 1e-6) -> float:
    rho = compute_rho(T, C, batch - 1, tau)
    return cdp_eps(rho, delta)


# ─────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────
def embed_texts(texts, model_name: str = 'all-mpnet-base-v2',
                device=None, batch_size: int = 64) -> np.ndarray:
    device = device or torch.device('cpu')
    model = SentenceTransformer(model_name, device=str(device))
    embeddings = model.encode(texts, batch_size=batch_size,
                              show_progress_bar=True)
    return embeddings
