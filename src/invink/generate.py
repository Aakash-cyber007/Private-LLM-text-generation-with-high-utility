"""
Generation Function for InvisibleInk
Author: Vishnu Vinod
License: GPLv3

MODIFICATION: Token selection step replaced with RNM-Exponential noise
(Report-Noisy-Max with Exponential noise), which is equivalent to the
Permute-and-Flip mechanism (McKenna & Sheldon, NeurIPS 2020; Ding et al. 2021).

Noise calibration derivation:
  InvisibleInk Theorem 2 (Vinod et al., NeurIPS 2025):
    rho_tok = Delta^2 / (2 * tau^2)   [per-token zCDP, Delta = clip_norm/B]

  Softmax (Exponential Mechanism) satisfies this tightly.

  RNM-Exp with scale lambda satisfies pure eps-DP with eps = 2*Delta/lambda
  (McKenna & Sheldon 2020). Converting pure eps-DP -> zCDP (Bun & Steinke 2016):
    rho_rnm <= eps^2 / 2 = 2*Delta^2 / lambda^2

  Matching rho_rnm = rho_tok:
    2*Delta^2 / lambda^2  =  Delta^2 / (2*tau^2)
    lambda^2 = 4*tau^2
    lambda   = 2 * tau   =   2 * temperature

  lambda is INDEPENDENT of clip_norm and batch_size because tau already
  encodes them: tau = clip_norm / (B * sqrt(2*rho_tok)).

Bug fix: previous code had `lam = 2*temp / (batch-1)` which incorrectly
divided by (B-1). Temperature already encodes the 1/B sensitivity factor.
"""

from __future__ import annotations

import os
import math
import random
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union
from types import SimpleNamespace
from collections import abc

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import special as spl

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


if FOUND_TORCH and FOUND_TRANSFORMERS:
    from .utils import PUB_PROMPT, PRV_PROMPT
    from .utils import combined_mean_std
    from .utils import setup_seed, setup_device
    from .utils import load_hf_model, load_hf_tokenizer
    from .utils import batchify, preprocess, get_prompt
    from .utils import get_clip, get_epsilon, get_topk, difference_clip


def generate(
    txt_list_or_path: Optional[Union[str, Iterable[str], Path, pd.DataFrame]] = None,
    model_name_or_path: Optional[Union[str, Path]] = None,
    dataset_desc: Optional[str] = None,
    system_prompt: Optional[str] = "You are a synthetic text generator. Generate high-quality and coherent text based on the given prompts.",
    pub_prompt: Optional[str] = PUB_PROMPT,
    prv_prompt: Optional[str] = PRV_PROMPT,
    epsilon: Optional[float] = 10.0, 
    print_text: Optional[bool] = False,
    column_name: Optional[str] = 'text',
    drop_empty: bool = True,
    batch_size: Optional[int] = 8, 
    num: Optional[Union[int, str]] = "auto",
    max_toks: Optional[Union[int, str]] = "auto",
    per_device_minibatch_size: Optional[Union[int, str]] = "auto",
    delta: Optional[float] = 1e-5, 
    temperature: Optional[float] = 1.0,
    topk: Optional[int] = 100, 
    dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
    device_map: Optional[Union[str, torch.device]] = "auto",
    auth_token: Optional[str] = None,
    allow_download: bool = True,
    trust_remote_code: bool = True,
    padding_side: str = "left",
    truncation_side: str = "right",
    random_seed: int = 42,
) -> Any:
    """
    Generating private synthetic text given a batch of input texts, model name,
    privacy budget, batch size and other generation configuration information.

    Token selection uses RNM-Exponential noise (Report-Noisy-Max with Exponential
    noise), equivalent to the Permute-and-Flip mechanism, replacing the original
    Softmax / Exponential Mechanism. All other arguments are unchanged.

    Args: (same as original InvisibleInk generate())
        txt_list_or_path: List of private texts
        model_name_or_path: HuggingFace model identifier or local path
        dataset_desc: Brief non-sensitive description of the dataset
        system_prompt: System prompt for the LLM
        prv_prompt: User prompt with private reference text
        pub_prompt: User prompt without private reference text
        epsilon: Privacy parameter for (epsilon,delta)-DP
        print_text: Whether to print generated texts
        column_name: Column name in DataFrame/CSV containing texts
        drop_empty: Drop empty rows/strings
        batch_size: Maximum LLM inferences per generated token (= B+1)
        num: Number of synthetic samples; "auto" = len(texts)//(batch_size-1)
        max_toks: Max tokens per sample; "auto" = mean + 2*std of input lengths
        per_device_minibatch_size: GPU batch size; "auto" = batch_size
        delta: Failure probability for (epsilon,delta)-DP
        temperature: Sampling temperature (tau); controls privacy-utility tradeoff
        topk: Top-k+ parameter; -1 for full vocabulary
        dtype: Model precision
        device_map: Device mapping strategy
        auth_token: HuggingFace authentication token
        allow_download: Allow downloading model if not found locally
        trust_remote_code: Trust remote code from model hub
        padding_side: Side for padding
        truncation_side: Side for truncation
        random_seed: Random seed for reproducibility
    Returns:
        SimpleNamespace with .texts, .lens, .epsilon_spent, .topk_avg,
        .topk_std, .expansion_set_counts
    """
    
    if txt_list_or_path is None:
        raise ValueError("No reference texts specified.")
    if epsilon is None:
        raise ValueError("Epsilon (privacy budget) is not specified.")
    
    # ── Load input texts ──────────────────────────────────────────────────────
    print('Loading data....')
    if not isinstance(txt_list_or_path, (str, abc.Iterable, Path, pd.DataFrame)):
        raise ValueError("txt_list_or_path must be a string, path or DataFrame.")
    if isinstance(txt_list_or_path, pd.DataFrame):
        data = txt_list_or_path
    elif isinstance(txt_list_or_path, (str, Path)):
        try:
            data = pd.read_csv(txt_list_or_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
    elif isinstance(txt_list_or_path, abc.Iterable):
        data = pd.DataFrame({'text': list(txt_list_or_path)})
    else:
        raise TypeError("Invalid input type for txt_list_or_path.")

    if data.shape[1] == 1:
        text_series = data.iloc[:, 0]
    elif column_name in data.columns:
        text_series = data[column_name]
    else:
        raise ValueError(f"Column '{column_name}' not found in dataframe.")

    cleaned_data = text_series.map(preprocess).to_list()
    texts = cleaned_data if not drop_empty else [t for t in cleaned_data if t]
    print('Data loaded successfully!')
    print('----------------\n')

    # ── Model name validation ─────────────────────────────────────────────────
    if model_name_or_path is None:
        model_name_or_path = ""
    if not isinstance(model_name_or_path, (str, Path)) or \
            len(str(model_name_or_path).strip()) == 0:
        raise ValueError("model_name_or_path must be a non-empty string or path.")
    is_local = os.path.isdir(model_name_or_path)
    if not is_local and not allow_download:
        raise FileNotFoundError(
            f"Model '{model_name_or_path}' not found locally and downloads disabled."
        )
    model_name = model_name_or_path

    # ── Input validation ──────────────────────────────────────────────────────
    if not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer.")
    if not (isinstance(epsilon, (int, float)) and epsilon >= 0):
        raise ValueError("epsilon must be a non-negative number.")
    if not (isinstance(batch_size, int) and batch_size > 0):
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(topk, int):
        raise ValueError("topk must be an integer.")
    for name, val in (("padding_side", padding_side),
                      ("truncation_side", truncation_side)):
        if not isinstance(val, str):
            raise ValueError(f"{name} must be a string.")
    for name, val in (("num", num), ("max_toks", max_toks),
                      ("per_device_minibatch_size", per_device_minibatch_size)):
        if not ((isinstance(val, int) and val > 0) or val == "auto"):
            raise ValueError(f"{name} must be a positive integer or 'auto'.")
    if dataset_desc is None:
        raise ValueError("dataset_desc is required.")

    # ── Device and model setup ────────────────────────────────────────────────
    setup_seed(seed=random_seed)
    device = setup_device(device_map)

    print('Loading model and tokenizer....')
    tokenizer = load_hf_tokenizer(
        name_or_path=model_name,
        padding_side=padding_side,
        truncation_side=truncation_side,
        allow_download=allow_download,
        auth_token=auth_token,
        dtype=dtype,
    )
    model = load_hf_model(
        name_or_path=model_name,
        dtype=dtype,
        device_map=device,
        auth_token=auth_token,
        allow_download=allow_download,
        trust_remote_code=trust_remote_code,
    )
    device = model.device if device == "auto" else device

    # ── Vocabulary size ───────────────────────────────────────────────────────
    if hasattr(model, 'vocab_size'):
        vocab_size = model.vocab_size
    elif hasattr(model.config, 'vocab_size'):
        vocab_size = model.config.vocab_size
    elif hasattr(model.config, 'text_config') and \
            hasattr(model.config.text_config, 'vocab_size'):
        vocab_size = model.config.text_config.vocab_size
    elif hasattr(tokenizer, 'vocab_size'):
        vocab_size = tokenizer.vocab_size
    else:
        dummy = tokenizer("dummy", return_tensors="pt").to(device).input_ids
        out = model.generate(dummy, past_key_values=None, use_cache=True,
                             max_new_tokens=1, pad_token_id=tokenizer.eos_token_id,
                             output_logits=True, return_dict_in_generate=True)
        vocab_size = out.logits[0].cpu().numpy().shape[1]

    print('Model and tokenizer loaded successfully!\n')
    print('----------------\n')

    if topk < 0:
        topk = vocab_size

    # ── Auto-parameter selection ──────────────────────────────────────────────
    if num == "auto":
        num = len(texts) // (batch_size - 1)
        print(f"Auto-calculated num = {num}.")
    if max_toks == "auto":
        token_lengths = [len(tokenizer.encode(t)) for t in texts]
        max_toks = int(np.mean(token_lengths) + 2 * np.std(token_lengths))
        print(f"Auto-calculated max_toks = {max_toks}.")
    if per_device_minibatch_size == "auto" or \
            per_device_minibatch_size > batch_size:
        print(f"Set the minibatch size to be equal to batch_size ({batch_size})")
        per_device_minibatch_size = batch_size
    num_minibatches = batch_size // per_device_minibatch_size

    if num * (batch_size - 1) > len(texts):
        raise ValueError(
            'Not enough private samples! Use smaller batch_size or fewer generations.'
        )

    # ── Privacy accounting ────────────────────────────────────────────────────
    clip_norm = get_clip(
        epsilon=epsilon,
        num_toks=max_toks,
        batch_size=batch_size,
        delta=delta,
        temp=temperature,
    )

    # ── RNM-Exp noise scale ───────────────────────────────────────────────────
    # Derivation (see module docstring):
    #   Softmax EM satisfies rho_tok = Delta^2/(2*tau^2) per token (zCDP).
    #   RNM-Exp with Exp(lambda) satisfies pure eps-DP: eps = 2*Delta/lambda.
    #   Converting pure eps-DP -> zCDP: rho <= eps^2/2 = 2*Delta^2/lambda^2.
    #   Matching: lambda = 2 * temperature.
    #   References: McKenna & Sheldon (NeurIPS 2020); Ding et al. (2021);
    #               Vinod, Pillutla & Thakurta (NeurIPS 2025) Theorem 2.
    rnm_lambda = 2.0 * temperature

    text_batches = list(batchify(lst=texts, s=batch_size - 1, n=num))

    results = {
        'text':     [],
        'len':      [],
        'eps':      [],
        'topk_avg': [],
        'topk_std': [],
        'ext':      [],
    }

    # ── Generation loop ───────────────────────────────────────────────────────
    print('Begin synthetic text generation....')
    if print_text:
        print('----------------\n')

    for i in range(num) if print_text else tqdm(range(num)):
        text_batch = text_batches[i]
        cache = [None] * num_minibatches
        token_seq = torch.tensor([], dtype=int, device=device)
        batch_prompts = []

        for txt in text_batch:
            prompt = get_prompt(
                tokenizer=tokenizer,
                dataset_desc=dataset_desc,
                system_prompt=system_prompt,
                pub_prompt=pub_prompt,
                prv_prompt=prv_prompt,
                private_ref=txt,
            )
            batch_prompts.append(prompt)
        # public prompt appended last
        batch_prompts.append(get_prompt(
            tokenizer=tokenizer,
            dataset_desc=dataset_desc,
            system_prompt=system_prompt,
            pub_prompt=pub_prompt,
            prv_prompt=prv_prompt,
        ))

        encoded = tokenizer(
            batch_prompts, return_tensors='pt',
            padding=True, truncation=True
        ).to(device)
        minibatch_masks  = list(torch.split(encoded.attention_mask,
                                            per_device_minibatch_size))
        minibatch_tokens = list(torch.split(encoded.input_ids,
                                            per_device_minibatch_size))

        counter = 0
        topk_counts, ext_count = [], 0

        for _ in range(max_toks):
            logits = np.zeros((batch_size, vocab_size))

            for j in range(num_minibatches):
                masks   = minibatch_masks[j]
                prompts = minibatch_tokens[j]
                low  = j * per_device_minibatch_size
                high = (j + 1) * per_device_minibatch_size
                token_seq_cast = torch.broadcast_to(
                    token_seq, (prompts.shape[0], token_seq.shape[0])
                )
                mask_append   = torch.cat(
                    (masks, torch.ones_like(token_seq_cast)), 1
                )
                prompt_append = torch.cat((prompts, token_seq_cast), 1)

                output = model.generate(
                    prompt_append,
                    past_key_values=cache[j],
                    use_cache=True,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=mask_append,
                    do_sample=True,
                    temperature=temperature,
                    top_p=1.0,
                    output_logits=True,
                    return_dict_in_generate=True,
                )
                logits[low:high, :] = output.logits[0].cpu().numpy()
                cache[j] = output.past_key_values

            del output
            torch.cuda.empty_cache()

            # DClip + aggregate
            pub_logits, prv_logits = logits[-1], logits[:-1]
            clipped_logits = difference_clip(
                logit=prv_logits,
                publogit=pub_logits,
                clip_norm=clip_norm,
            )
            avg_clip_logits = np.mean(clipped_logits, axis=0)

            # Top-k+ mask
            pub_mask, idxs = get_topk(
                pub_logits=pub_logits,
                k=topk,
                clip=clip_norm,
                batch=batch_size,
            )
            avg_clip_logits = np.where(pub_mask, avg_clip_logits, -np.inf)
            topk_counts.append(np.sum(pub_mask))

            # ── RNM-Exp token selection ───────────────────────────────────
            # Only sample from the top-k+ valid set (pub_mask == True).
            # Add Exp(rnm_lambda) noise to each valid logit and take argmax.
            # This is equivalent to Permute-and-Flip (Ding et al. 2021) and
            # provably dominates Softmax in utility (McKenna & Sheldon 2020).
            valid_idx    = np.where(pub_mask)[0]
            valid_logits = avg_clip_logits[valid_idx]
            exp_noise    = np.random.exponential(
                scale=rnm_lambda, size=valid_logits.shape
            )
            nxt_token = int(valid_idx[np.argmax(valid_logits + exp_noise)])
            # ─────────────────────────────────────────────────────────────

            token_seq = torch.cat(
                (token_seq, torch.tensor([nxt_token], device=device))
            )
            if nxt_token in idxs:
                ext_count += 1
            counter += 1

            # EOS check — nxt_token is a plain Python int after RNM argmax
            eos_id = model.generation_config.eos_token_id
            if isinstance(eos_id, (list, tuple)):
                if nxt_token in eos_id:
                    break
            elif nxt_token == eos_id:
                break

        # store results
        cleaned_text = preprocess(
            tokenizer.decode(token_seq, skip_special_tokens=True)
        )
        results['text'].append(cleaned_text)
        if print_text:
            print(f'Text Number: {i+1}/{num}')
            print(cleaned_text)
            print('----------------\n')

        results['topk_avg'].append(np.mean(topk_counts))
        results['topk_std'].append(np.std(topk_counts))
        results['ext'].append(int(ext_count))
        results['len'].append(int(counter))

        eps_calc = get_epsilon(
            num_toks=counter,
            clip_norm=clip_norm,
            batch_size=batch_size,
            temp=temperature,
            delta=delta,
        )
        results['eps'].append(float(eps_calc))

    print('Generation complete!')
    print('----------------\n')

    output = SimpleNamespace(
        texts=results['text'],
        lens=results['len'],
        epsilon_spent=results['eps'],
        topk_avg=float(
            combined_mean_std(
                results['topk_avg'], results['topk_std'], results['len']
            )[0]
        ),
        topk_std=float(
            combined_mean_std(
                results['topk_avg'], results['topk_std'], results['len']
            )[1]
        ),
        expansion_set_counts=results['ext'],
    )
    return output
