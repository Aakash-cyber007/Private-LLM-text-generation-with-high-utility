"""
Utility functions for prompting, privacy accounting, logging, I/O, and simple dataset helpers.
Author: Vishnu Vinod
License: GPLv3
"""

from __future__ import annotations

import os
import sys
import copy
import math
import random
from typing import Any, Iterator, Iterable, List, Optional, Sequence, Tuple, Union

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    FOUND_TORCH = True
except (ImportError, ModuleNotFoundError):
    FOUND_TORCH = False
    
try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    FOUND_TRANSFORMERS = True
except (ImportError, ModuleNotFoundError):
    FOUND_TRANSFORMERS = False


############################################################################
# MODEL GENERATION AND PROMPT TEMPLATES
############################################################################

PROMPT_TEMPLATE = [
    {"role": "system", "content": None},
    {"role": "user", "content": None},
]

PRV_PROMPT = """
You are given a **DATASET_DESCRIPTION** and a **PRIVATE_REFERENCE**. 

### TASK:
- Generate new synthetic text sample which could belong to the dataset described in the DATASET_DESCRIPTION.
- Copy the length, style, structure, tone, and vocabulary of the PRIVATE_REFERENCE in synthetic text sample.

### DATASET_DESCRIPTION:
{}

### PRIVATE_REFERENCE:
{}

### RULES:
1. Output only the synthetic text sample, no prefix or suffix annotations or any explanations.
2. Output format should be pure text unless alternate formatting like JSON etc., is specified.
3. Keep the average length of the synthetic text sample similar to that of the PRIVATE_REFERENCE.
4. Maintain coherence, fluency, and relevance to the dataset context.
5. Do NOT include any analysis, explanations or reasoning — only output the final synthetic text sample.

### OUTPUT:
Return only the synthetic text sample as a single block of text.
"""

PUB_PROMPT = """
You are given a **DATASET_DESCRIPTION**.

### TASK:
- Generate new synthetic text sample which could belong to the dataset described in the DATASET_DESCRIPTION.

### DATASET_DESCRIPTION:
{}

### RULES:
1. Output only the synthetic text sample, no prefix or suffix annotations or any explanations.
2. Output format should be pure text unless alternate formatting like JSON etc., is specified.
4. Maintain coherence, fluency, and relevance to the dataset context.
5. Do NOT include any analysis, explanations or reasoning — only output the final synthetic text sample.

### OUTPUT:
Return only the synthetic text sample as a single block of text.
"""


############################################################################
# BASIC SETUP
############################################################################

def setup_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.
    Args:
        seed: non-negative integer seed.
    Returns:
        None
    """
    print(f"Setting seed for reproducibility... seed = {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    
    if not FOUND_TORCH:
        raise ModuleNotFoundError(
            """PyTorch not found. Please install PyTorch. For details, see `https://github.com/cerai-iitm/invisibleink` 
                and `https://pytorch.org/get-started/locally/`.
            """)
    
    # set torch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def setup_device(gpu: Union[int, str] = -1) -> Union[torch.device, str]:
    """Return a torch.device instance.
    Args:
        gpu: GPU index to use. Use -1 for CPU and "auto" for auto device map.
    Returns:
        torch.device instance or "auto" for auto device map.
    """
    if not (isinstance(gpu, int) or gpu == "auto"):
        raise TypeError("gpu must be an integer, -1, or 'auto'.")
    
    if not FOUND_TORCH:
        raise ModuleNotFoundError(
            """PyTorch not found. Please install PyTorch. For details, see `https://github.com/cerai-iitm/invisibleink` 
                and `https://pytorch.org/get-started/locally/`.
            """)

    if gpu == "auto": device = "auto"
    elif gpu < 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
    elif isinstance(gpu, int) and gpu >= 0:
        device_count = torch.cuda.device_count()
        if 0 <= gpu < device_count:
            device = torch.device(f"cuda:{gpu}")
        else: 
            raise RuntimeError(
                f"Requested GPU index {gpu} is invalid. Available GPUs: {list(range(device_count))}"
            )
    else: device = torch.device("cpu")
    
    print('PyTorch Version:', torch.__version__)
    print('Device:', device)
    return device


############################################################################
# DATASET AND MODEL UTILS
############################################################################


def preprocess(s: Any) -> str:
    """Normalize whitespace and coerce to string. None becomes empty string.
    Args:
        s: Input to preprocess. Ideally a string, other data types are coerced to string.
    Returns:
        Preprocessed string.
    """
    if s is None or (pd.isna(s) if not isinstance(s, List) else pd.isna(s).any()): 
        return ""
    else: 
        s = str(s)
        s = s.strip()
    return " ".join(str(s).split())


def _parse_dtype(dtype: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
    """
    Parse dtype string to torch.dtype.
    Args:
        dtype: Data type as string or torch.dtype
    Returns:
        torch.dtype object
    """
    if not FOUND_TORCH:
        raise ModuleNotFoundError(
            """PyTorch not found. Please install PyTorch. For details, see `https://github.com/cerai-iitm/invisibleink` 
                and `https://pytorch.org/get-started/locally/`.
            """)
    
    # default to bfloat16
    if dtype is None: torch_dtype = torch.bfloat16
    if isinstance(dtype, torch.dtype): torch_dtype = dtype
    
    dtype_map = {
        "half": torch.float16,
        "float": torch.float32,
        "double": torch.float64,
        "float16": torch.float16,
        "float32": torch.float32,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if isinstance(dtype, str): 
        dtype_lower = dtype.lower()
        if dtype_lower not in dtype_map:
            raise ValueError(f"Unknown dtype: {dtype}. Supported: {list(dtype_map.keys())}")
        torch_dtype = dtype_map[dtype_lower]
    
    # check for bfloat16 support
    if torch_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("bfloat16 not supported on this device. Falling back to float16.")
        torch_dtype = torch.float16
    return torch_dtype


def load_hf_tokenizer(
    name_or_path: str,
    padding_side: str = "left",
    truncation_side: str = "right",
    auth_token: Optional[str] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    allow_download: bool = True
) -> AutoTokenizer:
    """
    Load a tokenizer (which has a chat template) from HuggingFace.
    Args:
        name_or_path: HuggingFace model identifier or file path
        padding_side: Side for padding ("left" or "right")
        truncation_side: Side for truncation ("left" or "right")
        auth_token: HuggingFace authentication token for private models 
        allow_download: Whether to allow downloading model if not found locally
        dtype: Data type for model ("float32", "float16", "bfloat16", or torch.dtype)
    Returns:
        Loaded huggingface tokenizer
    """
    if not FOUND_TORCH:
        raise ModuleNotFoundError(
            """PyTorch not found. Please install PyTorch. For details, see `https://github.com/cerai-iitm/invisibleink` 
                and `https://pytorch.org/get-started/locally/`.
            """)
    if not FOUND_TRANSFORMERS:
        raise ModuleNotFoundError(
            """Transformers not found. Please install Transformers. For details, see `https://github.com/cerai-iitm/invisibleink` 
                and `https://huggingface.co/transformers/installation.html`.
            """)
    if not isinstance(name_or_path, (str, Path)) or len(name_or_path.strip()) == 0:
        raise ValueError("name_or_path must be a non-empty string or path.")
    
    # Determine if user passed a local directory or HF Hub model name
    is_local = os.path.isdir(name_or_path)
    if not is_local and not allow_download:
        raise FileNotFoundError(
            f"Model '{name_or_path}' not found locally and downloads disabled."
        )
    
    model_name = name_or_path
    print(f"Loading tokenizer for {model_name}")
    torch_dtype = _parse_dtype(dtype)  # validate dtype if provided
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=auth_token,
        dtype=torch_dtype,
        padding_side=padding_side,
        truncation_side=truncation_side,
    )
    
    # Set pad token if not available
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    
    # Ensure tokenizer compatibility
    if not hasattr(tokenizer, "apply_chat_template"):
        print("Tokenizer may not support chat formatting. Proceed with caution.")
    
    print("Tokenizer loaded successfully")
    return tokenizer


def load_hf_model(
    name_or_path: Union[str, Path],
    dtype: Optional[Union[str, torch.dtype]] = None,
    device_map: Optional[Union[str, torch.device]] = None,
    auth_token: Optional[str] = None,
    allow_download: bool = True,
    trust_remote_code: bool = True,
    **kwargs: Any,
) -> AutoModelForCausalLM:
    """
    Load a causal language model from HuggingFace.
    Args:
        name_or_path: HuggingFace model identifier or file path
        dtype: Data type for model ("float32", "float16", "bfloat16", or torch.dtype)
        device_map: Device mapping strategy ("auto" or custom single-device)
        auth_token: HuggingFace authentication token for private models 
        allow_download: Whether to allow downloading model if not found locally
        trust_remote_code: Whether to trust remote code from model hub
        **kwargs: Additional arguments for AutoModelForCausalLM.from_pretrained()
    Returns:
        Loaded model
    """
    if not FOUND_TORCH:
        raise ModuleNotFoundError(
            """PyTorch not found. Please install PyTorch. For details, see `https://github.com/cerai-iitm/invisibleink` 
                and `https://pytorch.org/get-started/locally/`.
            """)
    if not FOUND_TRANSFORMERS:
        raise ModuleNotFoundError(
            """Transformers not found. Please install Transformers. For details, see `https://github.com/cerai-iitm/invisibleink` 
                and `https://huggingface.co/transformers/installation.html`.
            """)
    
    if not isinstance(name_or_path, (str, Path)) or len(name_or_path.strip()) == 0:
        raise ValueError("name_or_path must be a non-empty string or path.")
    
    # Determine if user passed a local directory or HF Hub model name
    is_local = os.path.isdir(name_or_path)
    if not is_local and not allow_download:
        raise FileNotFoundError(
            f"Model '{name_or_path}' not found locally and downloads disabled."
        )
    
    model_name = name_or_path
    print(f"Loading model {model_name}")
    torch_dtype = _parse_dtype(dtype)
    
    # Prepare model kwargs
    model_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": trust_remote_code,
        **kwargs,
    }
    
    # Use specified dtype, device_map, and auth_token to load model
    if auth_token is not None: model_kwargs["token"] = auth_token
    if torch_dtype is not None: model_kwargs["dtype"] = torch_dtype
    if device_map is not None: model_kwargs["device_map"] = device_map
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    
    # Move to device if not using device_map
    if device_map is None:
        device_map = torch.device("cpu")
        model = model.to(device_map)
    
    # Set to eval mode by default
    model.eval()
    print(f"Model loaded successfully on {"all devices" if device_map == "auto" else device_map}")
    print(f"Model dtype: {model.dtype}")
    return model

            
def get_prompt(
    tokenizer, 
    dataset_desc: Optional[str] = None, 
    private_ref: Optional[str] = "",
    system_prompt: Optional[str] = "You are a synthetic text generator. Generate high-quality and coherent text based on the given prompts.",
    pub_prompt: Optional[str] = PUB_PROMPT,
    prv_prompt: Optional[str] = PRV_PROMPT
    ) -> str:
    """
    Create a prompt including private reference text for generation (if provided).
    Args:
        tokenizer: HuggingFace tokenizer supporting apply_chat_template().
        dataset_desc: Text describing dataset properties or style.
        private_ref: Private text used to condition generation (optional).
        system_prompt: System prompt for the language model; instructs it to act as a synthetic text generator, by default.
        prv_prompt: User prompt for the language model to prompt with private reference text; must contain 2 format fields for the dataset description and private reference
        pub_prompt: User prompt for the language model to prompt without private reference text; must contain 1 format field for the dataset description
    Returns:
        A formatted prompt string ready for model.generate().
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must not be None.")
    if not hasattr(tokenizer, "apply_chat_template"):
        raise AttributeError(
            "Tokenizer does not support apply_chat_template(). "
            "Use a chat-capable tokenizer (LLAMA, Mistral instruct, etc.)"
        )  
    if dataset_desc is None: raise ValueError("dataset_desc cannot be NoneType. Please input a short description of your dataset.")
    elif isinstance(dataset_desc, (int, float, str)): dataset_desc = str(dataset_desc)
    else: raise ValueError("dataset_desc should be string convertible.")

    if private_ref is None: private_ref = ""
    elif isinstance(private_ref, (int, float, str)): private_ref = str(private_ref)
    else: raise ValueError("private_ref should be string convertible.")
    
    if isinstance(system_prompt, (int, float, str)): system_prompt = str(system_prompt)
    else: raise ValueError("system_prompt should be string convertible.")
    if isinstance(pub_prompt, (int, float, str)): pub_prompt = str(pub_prompt)
    else: raise ValueError("pub_prompt should be string convertible.")
    if isinstance(prv_prompt, (int, float, str)): prv_prompt = str(prv_prompt)
    else: raise ValueError("prv_prompt should be string convertible.")

    if private_ref == "": user_prompt = pub_prompt
    else: user_prompt = prv_prompt

    template = copy.deepcopy(PROMPT_TEMPLATE)  
    template[0]['content'] = system_prompt
    template[1]['content'] = user_prompt.format(dataset_desc, private_ref)
    prompt_txt = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
    return prompt_txt


def batchify(lst: Sequence[Any], s: int, n: int) -> Iterator[List[Any]]:
    """Generate n batches of size s from lst.
    Args:
        lst: input sequence.
        s: batch size (positive int).
        n: number of batches to produce.
    Returns:
        Slices of the list as standard Python lists.
    """
    if not isinstance(s, int) or s <= 0:
        raise ValueError("s must be a positive integer")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    
    length = len(lst)
    if length < n * s:
        raise ValueError(f"List too small for creating {n} batches of size {s} (len={length})")
    for i in range(0, n):
        yield list(lst[i * s : (i + 1) * s])


def combined_mean_std(
    means: Union[Iterable[float], np.ndarray],
    stds: Optional[Union[Iterable[float], np.ndarray]] = None,
    lens: Optional[Union[Iterable[int], Iterable[float], np.ndarray]] = None,
) -> Tuple[float, float]:
    """
    Compute a combined mean and standard deviation given per-group statistics.
    Args:
        means: Iterable of group means.
        stds: Iterable of group standard deviations. If None, assumes zero variance.
        lens: Iterable of sample sizes for each group. If None, assumes equal sizes.
    Returns:
        (mean, std): The aggregated mean and standard deviation across all groups.
    """

    # Convert to numpy arrays
    means = np.asarray(means, dtype=float)
    if means.size == 0: raise ValueError("means cannot be empty.")

    # Handle stds and lens
    if stds is None: stds = np.zeros_like(means, dtype=float)
    else: stds = np.asarray(stds, dtype=float)
    if lens is None: lens = np.ones_like(means, dtype=float)
    else: lens = np.asarray(lens, dtype=float)

    # Check that shapes are compatible and lens do not contain negative values
    if not (len(means) == len(stds) == len(lens)):
        raise ValueError("`means`, `stds`, and `lens` must be the same length.")
    if np.any(lens < 0):
        raise ValueError("`lens` must contain non-negative values.")

    # calculate mean and variance
    tot_len = np.sum(lens)
    mean = float(np.sum(means * lens) / tot_len)
    variance = np.sum(lens * (stds**2 + (means - mean) ** 2)) / tot_len
    std = float(np.sqrt(max(variance, 0.0)))
    return mean, std


############################################################################
# INVISIBLEINK UTILS
############################################################################

def difference_clip(logit: Union[np.ndarray, torch.Tensor], 
        publogit: Union[np.ndarray, torch.Tensor], 
        clip_norm: float
    ) -> np.ndarray:
    """Upper bound the L-infinity norm of the difference between private and public logits by clip_norm.
    Works with numpy arrays or torch tensors. Returns numpy array.

    Args:
        logit: np.ndarray or torch.Tensor
        publogit: np.ndarray or torch.Tensor
        clip_norm: non-negative float.
    Returns:
        clipped numpy array.
    """
    # check for valid clip_norm
    if not (isinstance(clip_norm, (float, int)) and clip_norm >= 0):
        raise ValueError("clip_norm must be a non-negative number")
    
    # If torch tensors, convert to cpu numpy safely
    try:
        if isinstance(logit, torch.Tensor):
            logit_np = logit.detach().cpu().numpy()
        else:
            logit_np = np.asarray(logit)
        if isinstance(publogit, torch.Tensor):
            pub_np = publogit.detach().cpu().numpy()
        else:
            pub_np = np.asarray(publogit)
    except Exception:
        raise TypeError("logit and publogit must be array-like or torch.Tensor")

    # implicit shape check via numpy broadcasting
    clipped = pub_np + np.clip(logit_np - pub_np, -clip_norm, clip_norm)
    return clipped


def get_topk(pub_logits: Union[np.ndarray, Sequence[float]], k: int, clip: float, batch: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a mask for truncation prior to top-k+ sampling.
    Args:
        pub: 1-D array-like of public logits.
        k: top-k truncation parameter.
        clip: clipping norm used in difference clipping.
        batch: positive int (same as batch size used by clipping scheme).

    Returns:
        mask_pub: boolean mask as numpy with same shape as pub.
        idxs: numpy array of indices that lie in the extended band.
    """
    pub_arr = np.asarray(pub_logits)
    if pub_arr.ndim != 1:
        raise ValueError("pub must be a 1-D array-like.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive int.")
    if not isinstance(batch, int) or batch <= 0:
        raise ValueError("batch must be positive int")
    if not isinstance(clip, (int, float)):
        raise TypeError("clip must be numeric")
    if k > pub_arr.size:
        raise ValueError("k cannot be larger than number of elements in pub.")

    kthresh = np.partition(pub_arr, -k)[-k]
    k_ext = kthresh - 2.0 * float(clip) / float(batch)

    mask_pub = pub_arr >= k_ext
    idxs = np.where(np.logical_and(pub_arr >= k_ext, pub_arr <= kthresh))[0]
    return mask_pub, idxs


############################################################################
# PRIVACY UTILS
############################################################################

def cdp_delta(rho: float, eps: float) -> float:
    """Convert (rho)-zCDP to delta for a given eps to get (eps, delta)-DP (ADP) guarantees.
    Uses bounds in: https://arxiv.org/pdf/2004.00010v3.pdf#page=13.
    Args:
        rho: non-negative float, zCDP parameter (privacy budget).
        eps: non-negative float, epsilon (privacy budget) for ADP.
    Returns:
        delta: float in [0, 1], delta (failure probability) for ADP.
    """
    for name, val in (("rho", rho), ("eps", eps)):
        if not (isinstance(val, (int, float)) and val >= 0):
            raise TypeError(f"{name} must be a non-negative number.")

    rho = float(rho)
    eps = float(eps)

    # if privacy is 0 (completely private) then delta is 0
    if rho == 0: return 0.0

    amin, amax = 1.0001, max(2.0, (eps + 1.0) / (2.0 * rho) + 2.0)
    alpha = None
    for _ in range(1000):
        alpha = 0.5 * (amin + amax)
        log1p_term = math.log1p(-1.0 / alpha)
        derivative = (2.0 * alpha - 1.0) * rho - eps + log1p_term
        if derivative < 0: amin = alpha
        else: amax = alpha

    # handle exponentiation overflow and fit in range [0,1]
    exponent = (alpha - 1.0) * (alpha * rho - eps) + alpha * math.log1p(-1.0 / alpha)
    try:
        delta = math.exp(exponent) / (alpha - 1.0)
    except OverflowError:
        delta = 0.0
    return min(max(delta, 0.0), 1.0)


def cdp_eps(rho: float, delta: float = 1e-6) -> float:
    """Compute smallest eps such that rho-zCDP implies (eps, delta)-DP (ADP) for a given delta.
    Args:
        rho: non-negative float, zCDP parameter (privacy budget).
        delta: float in (0, 1), delta (failure probability) for ADP.
    Returns:
        eps: non-negative float, smallest epsilon (privacy budget) for ADP.
    """
    if not (isinstance(rho, (int, float)) and rho >= 0):
        raise ValueError("rho must be non-negative.")
    if not (isinstance(delta, (float, int)) and 0.0 < float(delta) <= 1.0):
        raise ValueError("delta must be in (0,1).")

    # if completely private then epsilon is 0
    if rho == 0: return 0.0

    # binary search
    epsmin, epsmax = 0.0, rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
    for _ in range(1000):
        eps = 0.5 * (epsmin + epsmax)
        if cdp_delta(rho, eps) <= delta: epsmax = eps
        else: epsmin = eps
    return float(epsmax)


def cdp_rho(eps: float, delta: float = 1e-6) -> float:
    """Compute smallest rho such that rho-zCDP implies (eps,delta)-DP (ADP).
    Args:
        eps: non-negative float, epsilon (privacy budget) for ADP.
        delta: float in (0, 1), delta (failure probability) for ADP.
    Returns:
        rho: non-negative float, smallest zCDP parameter (privacy budget).
    """
    if not (isinstance(eps, (int, float)) and eps >= 0):
        raise ValueError("eps must be non-negative.")
    if not (isinstance(delta, (float, int)) and 0.0 < float(delta) <= 1.0):
        raise ValueError("delta must be in (0,1).")

    # if completely private then rho is 0
    if eps == 0.0: return 0.0

    # binary search
    rhomin, rhomax = 0.0, max(1.0, eps + 1.0)
    for _ in range(2000):
        rho = 0.5 * (rhomin + rhomax)
        if cdp_delta(rho, eps) <= delta: rhomin = rho
        else: rhomax = rho
    return float(rhomin)


def compute_rho(num_toks: int, clip_norm: float, batch_size: int, temp: float) -> float:
    """Compute rho (zCDP) per-token multiplied by num_toks.
    Args:
        num_toks: positive int, number of tokens generated.
        clip_norm: positive float, clipping norm.
        batch_size: positive int, number of private references used per generation (B in the paper).
        temp: positive float, sampling temperature used during generation.
    Returns:
        rho_tot: float, total rho (zCDP) guarantee.
    """
    for name, val in (("num_toks", num_toks), ("batch_size", batch_size)):
        if not (isinstance(val, int) and val > 0):
            raise ValueError(f"{name} must be a positive integer.")
    for name, val in (("clip_norm", clip_norm), ("temp", temp)):
        if not (isinstance(val, (float, int)) and val > 0):
            raise ValueError(f"{name} must be a positive number.")

    rho_tok = 0.5 * (float(clip_norm) / (float(batch_size) * float(temp))) ** 2
    rho_tot = float(num_toks) * rho_tok
    return rho_tot


def get_epsilon(num_toks: int, clip_norm: float, batch_size: int, temp: float, delta: float = 1e-6) -> float:
    """Get (eps, delta)-DP epsilon for InvisibleInk-style mechanism.
    Args:
        num_toks: positive int, number of tokens generated.
        clip_norm: positive float, clipping norm for difference clipping.
        batch_size: positive int (>1), number of LLM inferences per generation (B+1 in the paper).
        temp: positive float, sampling temperature used during generation.
        delta: float in (0, 1), delta (failure probability) for ADP.
    Returns:
        eps: float, epsilon (privacy budget) for (eps, delta)-DP (ADP).
    """
    if not (isinstance(batch_size, int) or batch_size > 1):
        raise ValueError("batch_size must be an integer > 1.")
    if not (isinstance(num_toks, int) and num_toks > 0):
        raise ValueError("num_toks must be a positive integer.")
    if not (isinstance(delta, (float, int)) and 0.0 < float(delta) < 1.0):
        raise ValueError("delta must be in (0,1).")
    for name, val in (("clip_norm", clip_norm), ("temp", temp)):
        if not (isinstance(val, (float, int)) and val > 0):
            raise ValueError(f"{name} must be a positive number.")
    
    rho = compute_rho(num_toks, clip_norm, batch_size - 1, temp)
    eps = cdp_eps(rho, delta)
    return eps


def get_clip(epsilon: float, num_toks: int, temp: float, batch_size: int, delta: float = 1e-6) -> float:
    """Compute required clipping norm for target eps (inverse of get_epsilon).
    Args:
        eps: non-negative float, target epsilon (privacy budget) for (eps, delta)-DP (ADP).
        num_toks: positive int, number of tokens generated.
        temp: positive float, sampling temperature used during generation.
        batch_size: positive int (>1), number of LLM inferences per generation (B+1 in the paper).
        delta: float in (0, 1), delta (failure probability) for ADP.
    Returns:
        clip: float, required clipping norm for difference clipping.
    """
    if not (isinstance(batch_size, int) or batch_size > 1):
        raise ValueError("batch_size must be an integer > 1.")
    if not (isinstance(num_toks, int) and num_toks > 0):
        raise ValueError("num_toks must be a positive integer.")
    if not (isinstance(delta, (float, int)) and 0.0 < float(delta) < 1.0):
        raise ValueError("delta must be in (0,1).")
    for name, val in (("epsilon", epsilon), ("temp", temp)):
        if not (isinstance(val, (float, int)) and val >= 0):
            raise ValueError(f"{name} must be a non-negative number.")
        
    rho_tot = cdp_rho(epsilon, delta)
    rho_tok = rho_tot / float(num_toks)
    clip = float(temp) * float(batch_size - 1) * math.sqrt(max(0.0, 2.0 * rho_tok))
    return clip