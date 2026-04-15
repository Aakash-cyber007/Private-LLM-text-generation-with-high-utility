"""File containing utility functions for evaluation."""
import csv
import torch
import mauve
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.nn import CrossEntropyLoss
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

from sentence_transformers import SentenceTransformer
    
# relevant NER categories
NER_CATS = [
    'BIOLOGICAL_STRUCTURE', 'SIGN_SYMPTOM', 'SEVERITY', 'THERAPEUTIC_PROCEDURE', 'DISTANCE', 'DOSAGE',
    'MEDICATION', 'DISEASE_DISORDER', 'LAB_VALUE', 'DIAGNOSTIC_PROCEDURE', 'DETAILED_DESCRIPTION', 'ADMINISTRATION', 'COREFERENCE'
]

# read last lines from csv file
def read_csv_last(path, num=1000):
    with open(path, 'r') as file:
        total = sum(1 for row in csv.reader(file))
    skip_rows = range(1, total - num)
    last_rows = pd.read_csv(path, skiprows=skip_rows)
    return last_rows

# compute combined mean and std of a list of means and stds
def combined_mean_std(means, stds=None, lens=None):
    means = np.asarray(means, dtype=float)
    
    # zero std deviation if not given
    if stds is None: stds = np.zeros_like(means, dtype=float)
    else: stds = np.asarray(stds, dtype=float)
    
    # equal size populations if not given
    if lens is None: lens = np.ones_like(means, dtype=float)
    else: lens = np.asarray(lens, dtype=float)
    
    # compute population mean, variance and std
    tot_len = np.sum(lens)
    mean = np.sum(means * lens)/tot_len
    variance = np.sum(lens * (stds**2 + (means-mean)**2))/tot_len
    std = np.sqrt(max(variance, 0.0))
    return mean, std


############################################################################
# COMPUTE EVAL METRICS
############################################################################

# compute MAUVE scores
def compute_mauve(real_embed, fake_embed, num = 1000):
    real = real_embed[-1*num:]
    fake = fake_embed[:num]
    score = mauve.compute_mauve(p_features=real, q_features=fake, num_buckets = num//20, mauve_scaling_factor = 0.9)
    return score.mauve

# compute MedNER counts
def compute_ner(fake_txts, modelname = "Clinical-AI-Apollo/Medical-NER", device = torch.device('cpu')):
    pipe = pipeline("token-classification", model=modelname, aggregation_strategy='simple', device=device)
    fake_ner = pipe(fake_txts)
    ners = []
    for x in fake_ner:
        count = 0
        for i in x:
            count = count + 1 if i['entity_group'] in NER_CATS else count
        ners.append(count)
    return ners

# compute text perplexity from a text list
def compute_ppl(data, model_id='gpt2', batch_size = 4, add_start_token = True, device=torch.device('cpu'), max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.to(device)
    
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        assert (len(existing_special_tokens) > 0), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        assert (tokenizer.bos_token is not None), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else: max_tokenized_len = max_length

    if add_start_token and max_length: max_tokenized_len = max_length - 1
    else:max_tokenized_len = max_length

    encodings = tokenizer(data, add_special_tokens=False, padding=True, truncation=True, max_length=max_tokenized_len, return_tensors="pt", return_attention_mask=True).to(device)
    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")
    for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat([torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1)
        labels = encoded_batch
        with torch.no_grad(): out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        slogits = out_logits[..., :-1, :].contiguous()
        slabels = labels[..., 1:].contiguous()
        smask = attn_mask[..., 1:].contiguous()
        perplexity_batch = torch.exp((loss_fct(slogits.transpose(1, 2), slabels)*smask).sum(1)/smask.sum(1))
        ppls += perplexity_batch.tolist()
        
    ppls = np.array(ppls)
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}