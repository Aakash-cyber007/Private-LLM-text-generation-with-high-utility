"""
Generation Function for InvisibleInk
Author: Vishnu Vinod
License: GPLv3
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
    Generating private synthetic text given a batch of input texts, model name, privacy budget, batch size and other generation configuration information
    Args:
        txt_list_or_path: List of private texts to generate synthetic data, supplied as a list of strings, a pandas dataframe or a ".csv" filepath
        model_name_or_path: HuggingFace model identifier or folder path to downloaded file
        dataset_desc: brief description of the dataset detailing non-privacy sensitive information such as layout, format, broad content etc.
        system_prompt: System prompt for the language model; instructs it to act as a synthetic text generator, by default.
        prv_prompt: User prompt for the language model to prompt with private reference text; must contain 2 format fields for the dataset description and private reference
        pub_prompt: User prompt for the language model to prompt without private reference text; must contain 1 format field for the dataset description
        epsilon: privacy parameter for (epsilon,delta)-DP or Approximate DP
        print_text: bool, whether or not to print generated texts; defaults to False
        column_name: column containing text strings in pandas.DataFrame object or .csv file
            ; defaults to 'text'
        drop_empty: drop empty rows/strings in Iterator, pandas.DataFrame object or .csv file
        batch_size: maximum number of LLM inferences per generated token; defaults to 8
        num: Number of synthetic text samples to be generated; 
            defaults to "auto" which calculates num as (number of input texts) // (batch_size-1)
        max_toks: Maximum number of tokens to be generated per sample; 
            defaults to "auto" calculated as mean + 2 * std of the number of tokens in the input texts
        per_device_minibatch_size: Maximum number of batched prompts input to the LLM at a time; 
            defaults to "auto" (sets the same as batch_size); 
            To resolve CUDA out-of-memory errors, lower the per_device_minibatch_size; 
            48GB of RAM supports a minibatch size of 16 with a 1B parameter model loaded with "torch.bfloat16" precision.
        delta: float in [0, 1], delta (failure probability) for (epsilon,delta)-DP or Approximate DP; 
            defaults to 1e-5
        temperature: float, sampling temperature for the softmax-based probabilistic decoding step; 
            defaults to 1.0
        topk: int, topk parameter for truncated decoding to zero out the probabilities of all except the top "k" most probable tokens; 
            set to -1 for the full vocabulary setting;
            defaults to 100
        device_map: Device mapping strategy; 
            "auto" or custom single-device (GPU ID or -1 for CPU); 
            defaults to "auto"
        auth_token: HuggingFace authentication token for private models
        allow_download: Whether to allow downloading model if not found locally; 
            defaults to True
        trust_remote_code: Whether to trust remote code from model hub; 
            defaults to True
        padding_side: Side for padding ("left" or "right"); 
            defaults to "left"
        truncation_side: Side for truncation ("left" or "right"); 
            defaults to "right"
        dtype: Data type for model ("float32", "float16", "bfloat16", or torch.dtype); 
            defaults to "bfloat16" (if available)
    Returns:
        object containing generated synthetic text, sequence of tokens, generation model details, length of generation, privacy budget used (epsilon for ADP and rho for zCDP), mean and standard deviation in the top-k+ threshold, and number of tokens sampled from the expansion set
    """
    
    if txt_list_or_path is None: raise ValueError("No reference texts specified. Please input reference texts for generation.")
    if epsilon is None: raise ValueError("Epsilon (privacy budget) is not specified. Please specify the total privacy budget for generation.")
    
    
    # check for input text type
    print('Loading data....')
    if not isinstance(txt_list_or_path, (str, abc.Iterable, Path, pd.DataFrame)):
        raise ValueError("txt_list_or_path must be a non-empty string, path or pandas.DataFrame object.")
    if isinstance(txt_list_or_path, pd.DataFrame):
        data = txt_list_or_path
    elif isinstance(txt_list_or_path, (str, Path)):
        try:
            data = pd.read_csv(txt_list_or_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
    elif isinstance(txt_list_or_path, abc.Iterable):
        data = pd.DataFrame({'text' : list(txt_list_or_path)})
    else:
        raise TypeError("Invalid input type. txt_list_or_path should be a list of strings, a pandas.DataFrame object or a path to a .csv file.")
    
    # get pandas.Series object with all texts
    if data.shape[1] == 1:
        text_series = data.iloc[:, 0]
    elif column_name in data.columns:
        text_series = data[column_name]
    else:
        raise ValueError(f"Column {column_name} is not present in the dataframe. Please input correct column!")        
    
    # clean and drop empty strings if needed
    cleaned_data = text_series.map(preprocess).to_list()
    if not drop_empty: texts = cleaned_data
    else: texts = [text for text in cleaned_data if text]
    print('Data loaded successfully!')
    print('----------------\n')
    
    # Use a default model if not given, Determine if user passed a local directory or HF Hub model name
    if model_name_or_path is None:
        model_name_or_path = ""
    if not isinstance(model_name_or_path, (str, Path)) or len(model_name_or_path.strip()) == 0:
        raise ValueError("name_or_path must be a non-empty string or path.")
    is_local = os.path.isdir(model_name_or_path)
    if not is_local and not allow_download:
        raise FileNotFoundError(
            f"Model '{model_name_or_path}' not found locally and downloads disabled."
        )
    model_name = model_name_or_path
    
    # check inputs
    if not isinstance(random_seed, int):
        raise ValueError("random_seed should be an integer.")
    if not (isinstance(epsilon, (int, float)) and epsilon >=0):
        raise ValueError("epsilon must be a non-negative number")
    if not (isinstance(batch_size, int) and batch_size > 0):
        raise ValueError("batch_size must be a positive integer.")
    if not (isinstance(topk, int)):
        raise ValueError("top-k parameter must be an integer.")
    for name, val in (("padding_side", padding_side), ("truncation_side", truncation_side)):
        if not (isinstance(val, str)):
            raise ValueError(f"{name} must be a string.")
    for name, val in (("num", num), ("max_toks", max_toks), ("per_device_minibatch_size", per_device_minibatch_size)):
        if not ((isinstance(val, int) and val > 0) or val=="auto"):
            raise ValueError(f"Invalid Input. {name} must be a positive integer or auto-selected.")
    if dataset_desc is None:
        raise ValueError('Dataset description is empty. Please input a brief non privacy-sensitive description of the reference dataset.')
    
    # set random seed
    setup_seed(seed = random_seed)
    device = setup_device(device_map)
    
    # load model and tokenizer, set device
    print('Loading model and tokenizer....')
    tokenizer = load_hf_tokenizer(
        name_or_path = model_name,
        padding_side = padding_side, 
        truncation_side = truncation_side, 
        allow_download=allow_download,
        auth_token = auth_token, 
        dtype = dtype, 
    )
    model = load_hf_model(
        name_or_path = model_name,
        dtype = dtype,
        device_map = device,
        auth_token = auth_token,
        allow_download = allow_download,
        trust_remote_code = trust_remote_code
    )
    device = model.device if device=="auto" else device
    
    # get vocabulary size (logit length) - not all models have a vocab_size attribute
    if hasattr(model, 'vocab_size'): 
        vocab_size = model.vocab_size
    elif hasattr(model.config, 'vocab_size'):
        vocab_size = model.config.vocab_size
    elif hasattr(model.config, 'text_config'):
        if hasattr(model.config.text_config, 'vocab_size'):
            vocab_size = model.config.text_config.vocab_size
        else: raise RuntimeError('The given model text_config does not have a defined vocabulary size')
    elif hasattr(tokenizer, 'vocab_size'):
        vocab_size = tokenizer.vocab_size
    else:
        dummy_input = tokenizer("dummy input", return_tensors="pt").to(device).input_ids
        dummy_output = model.generate(dummy_input, past_key_values = None, use_cache=True, max_new_tokens = 1, 
                                pad_token_id = tokenizer.eos_token_id, output_logits=True,
                                return_dict_in_generate=True)
        vocab_size = dummy_output.logits[0].cpu().numpy().shape[1]

    # vocab_size = model.config.vocab_size
    # vocab_size = tokenizer.vocab_size
    print('Model and tokenizer loaded successfully!\n')
    print('----------------\n')
    
    # set full vocabulary setting if top-k < 0
    if topk < 0: topk = vocab_size
    
    # automatic parameter selection
    if num == "auto":
        num = len(texts) // (batch_size - 1)
        print(f"Auto-calculate the number of synthetic text sequences to be generated, num = {num}.")
    if max_toks == "auto":
        token_lengths = []
        for txt in texts:
            token_len = len(tokenizer.encode(txt))
            token_lengths.append(token_len)
        # set max_toks to be two standard deviations above the mean token length
        max_toks = int((np.mean(token_lengths) + 2 * np.std(token_lengths)) // 1)
        print(f"Auto-calculate the max. tokens in each generated sequence, max_toks = {max_toks}.")
    if per_device_minibatch_size == "auto" or per_device_minibatch_size > batch_size:
        print(f"Set the minibatch size to be equal to batch_size ({batch_size})")
        per_device_minibatch_size = batch_size
    num_minibatches = batch_size // per_device_minibatch_size
    
    # check if there are enough private samples
    if num * (batch_size - 1) > len(texts):
        raise ValueError('Not enough private samples! Use smaller batch sizes or generate fewer synthetic samples.')

    # Calculate clipping threshold
    clip_norm = get_clip(
        epsilon = epsilon,
        num_toks = max_toks,
        batch_size = batch_size,
        delta = delta,
        temp = temperature,
    )
    
    # batchify texts
    text_batches = list(batchify(lst = texts, s = batch_size - 1, n = num))
    
    results = {
        'text': [],
        'len': [],
        'eps': [],
        'topk_avg': [],
        'topk_std': [],
        'ext': [],
    }
    
    # iterate over generations
    print('Begin synthetic text generation....')
    if print_text: print('----------------\n')
    for i in range(num) if print_text else tqdm(range(num)):
        text_batch = text_batches[i]
        cache = [None] * num_minibatches
        token_seq = torch.tensor([], dtype=int, device=device)
        batch_prompts = []
        
        # batch consists of [....B private prompts.... + 1 public prompt]
        for txt in text_batch:
            prompt = get_prompt(
                tokenizer = tokenizer, 
                dataset_desc = dataset_desc,  
                system_prompt = system_prompt, 
                pub_prompt = pub_prompt,
                prv_prompt = prv_prompt,
                private_ref = txt,
            )
            batch_prompts.append(prompt)
        #add public prompt at the end
        prompt = get_prompt(
            tokenizer = tokenizer, 
            dataset_desc = dataset_desc,  
            system_prompt = system_prompt, 
            pub_prompt = pub_prompt,
            prv_prompt = prv_prompt,
        )
        batch_prompts.append(prompt)
        
        # get minibatches of encoded prompts
        encoded = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(device)
        minibatch_masks = list(torch.split(encoded.attention_mask, per_device_minibatch_size))
        minibatch_tokens = list(torch.split(encoded.input_ids, per_device_minibatch_size))
        
        # generate token by token
        counter = 0
        topk_counts, ext_count = [], 0
        for _ in range(max_toks):
            logits = np.zeros((batch_size, vocab_size))
            
            # iterate over minibatches
            for j in range(num_minibatches):
                # get minibatch of prompt tokens and append generated token sequence to it
                masks = minibatch_masks[j]
                prompts = minibatch_tokens[j]
                low, high = j * per_device_minibatch_size, (j+1) * per_device_minibatch_size
                token_seq_cast = torch.broadcast_to(token_seq, (prompts.shape[0], token_seq.shape[0]))                  
                mask_append = torch.cat((masks, torch.ones_like(token_seq_cast)), 1)
                prompt_append = torch.cat((prompts, token_seq_cast), 1)
                
                # generate outputs and store the logits and past key value pairs for future use
                output = model.generate(prompt_append, past_key_values = cache[j], use_cache=True,
                                        max_new_tokens = 1, pad_token_id = tokenizer.eos_token_id, 
                                        attention_mask = mask_append, do_sample = True, 
                                        temperature = temperature, top_p=1.0, output_logits=True, 
                                        return_dict_in_generate=True)
                
                # save only logits and KV cache
                logits[low:high, :] = output.logits[0].cpu().numpy()
                cache[j] = output.past_key_values
            
            # clear cache
            del output
            torch.cuda.empty_cache()

            # get pub/prv logits clip using DClip and average the clipped logits
            pub_logits, prv_logits = logits[-1], logits[:-1]
            clipped_logits = difference_clip(
                logit = prv_logits,
                publogit = pub_logits,
                clip_norm = clip_norm
            )
            avg_clip_logits = np.mean(clipped_logits, axis=0)

            # get next token
            pub_mask, idxs = get_topk(
                pub_logits = pub_logits,
                k = topk,
                clip = clip_norm,
                batch = batch_size
            )
            avg_clip_logits = np.where(pub_mask, avg_clip_logits, -np.inf)
            topk_counts.append(np.sum((pub_mask)))
            
            # get next token sampled
            # --- Permute-and-Flip (PF) Mechanism Replacement ---
            q = torch.tensor(avg_clip_logits / temperature, device=device)
            q_star = torch.max(q)
            p = torch.exp(q - q_star)
            
            # 3. 'R' defines the random permutation order (smallest R is checked first)
            # 4. 'U' represents the independent coin flip for each token
            R = torch.rand(vocab_size, device=device)
            U = torch.rand(vocab_size, device=device)
            valid = U <= p
            
            # 6. Pick the valid token that appears first in the permutation (minimum R)
            nxt_token = torch.argmin(torch.where(valid, R, torch.inf)).item()
            token_seq = torch.cat((token_seq, torch.tensor([nxt_token], device=device)))
            if nxt_token in idxs: ext_count += 1
            counter += 1
            
            # break loop if EOS is encountered
            if nxt_token == model.generation_config.eos_token_id:
                break
        
        # store results in a dictionary
        cleaned_text = preprocess(tokenizer.decode(token_seq, skip_special_tokens=True))
        results['text'].append(cleaned_text)
        if print_text:
            print(f'Text Number: {i+1}/{num}')
            print(cleaned_text)
            print('----------------\n')
        
        results['topk_avg'].append(np.mean(topk_counts))
        results['topk_std'].append(np.std(topk_counts))
        results['ext'].append(int(ext_count))
        results['len'].append(int(counter))
        
        # calculate the data-depenedent privacy guarantees for the generated sequence
        eps_calc = get_epsilon(
            num_toks = counter,
            clip_norm = clip_norm,
            batch_size = batch_size,
            temp = temperature,
            delta = delta
        )
        results['eps'].append(float(eps_calc))
    print('Generation complete!')
    print('----------------\n')
    
    # get outputs    
    output = SimpleNamespace(
        texts = results['text'],
        lens = results['len'],
        epsilon_spent = results['eps'],
        topk_avg = float(combined_mean_std(results['topk_avg'], results['topk_std'], results['len'])[0]),
        topk_std = float(combined_mean_std(results['topk_avg'], results['topk_std'], results['len'])[1]),
        expansion_set_counts = results['ext']
    )
    return output
