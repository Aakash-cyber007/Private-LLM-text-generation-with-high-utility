"""File containing utility functions for prompting, privacy accounting, logging, etc."""

import sys
import math
import torch
import random
import pickle
import logging
import numpy as np
from sentence_transformers import SentenceTransformer


############################################################################
# MODEL GENERATION AND PROMPT TEMPLATES
############################################################################

MODELS = {
    'tinyllama1B': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'llama3.2-1B':'meta-llama/Llama-3.2-1B-Instruct',
    'llama3-8B':'meta-llama/Meta-Llama-3-8B-Instruct',
}

PROMPT_TEMPLATE = [
    {"role": "system", "content": "You are a chatbot"},
    {"role": "user", "content": None},
]

PROMPTS = {
    'yelp': {
        'prv': 'Here is a text with Business Category : {} and Review Stars : {} out of 5.0. \n Text : "{}" \n Please output one more similar review of a fictional place with same score and of the same category of business. Poor reviews have lower scores and good reviews have higher scores.',
        'pub': 'Please give me a fake Yelp customer review of a fictional business of {} with a {} out of 5. Poor reviews have lower scores and good reviews have higher scores.',
    },
    'mimic': {
        'prv': 'Here is the text of the discharge summary of a patient discharged from a hospital {}{}\n Text : "{}" \n Please give me another text of a fake patient discharge summary for discharge from a hospital. Include typical sections like admitting diagnosis, major procedures (if any), discharge medications (using fictional drug names and dosages), and general follow-up instructions. Do not include any names, dates, or specific medical record numbers. The output text must begin with the exact words "Discharge Instructions:".',
        'pub': 'Please give me text of a fake patient discharge summary for discharge from a hospital {}{}. I only need fictional and representative examples for a non-medical purpose. Include typical sections like admitting diagnosis, major procedures (if any), discharge medications (using fictional drug names and dosages), and general follow-up instructions. Do not include any names, dates, or specific medical record numbers. The output text must begin with the exact words "Discharge Instructions:".',
    },
    'tab': {
        'prv': "Here is the text of a case transcript set before the European Court for Human Rights. {}{}\n Text : '{}' \n Please output a similar transcript of a fictional case under European Court for Human Rights. Begin with the phrase: 'PROCEDURE:\n\nThe case originated in an application'.",
        'pub': "Please output a transcript of a fictional case under European Court for Human Rights. {}{}Begin with the phrase: 'PROCEDURE:\n\nThe case originated in an application'.",
    },
}

# YELP dataset - 50x Categorical Labels - 5x Score labels, 10x Business Category labels
yelp_labels1 = ['Business Category: Arts & Entertainment', 'Business Category: Bars', 'Business Category: Beauty & Spas', 
           'Business Category: Event Planning & Services', 'Business Category: Grocery', 'Business Category: Health & Medical',
           'Business Category: Home & Garden', 'Business Category: Hotels & Travel',
           'Business Category: Restaurants', 'Business Category: Shopping']
yelp_labels2 = ['Review Stars: 1.0', 'Review Stars: 2.0', 'Review Stars: 3.0', 'Review Stars: 4.0', 'Review Stars: 5.0']


############################################################################
# BASIC SETUP
############################################################################

logger = logging.getLogger(__name__)

# setup random seeds
def setup_seed(seed):
    logger.info('Setting seed for reproducibility...')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# setup device
def setup_device(gpu = -1):
    dev = 'cuda:'+str(gpu) if (torch.cuda.is_available() and gpu>=0) else 'cpu'
    device = torch.device(dev)
    logger.info('PyTorch Version:', torch.__version__)
    logger.info('Device:', dev)
    return device
    
# setup logging to file and console
def setup_logging(filename = None, resume = False):
    root_logger = logging.getLogger()
    console = logging.StreamHandler(sys.stdout)
    file = logging.FileHandler(filename = filename, mode = 'a' if resume else 'w')

    root_logger.setLevel(logging.DEBUG)
    console.setLevel(logging.INFO)
    file.setLevel(logging.DEBUG)

    chformatter = logging.Formatter("%(asctime)s ==> %(message)s", "%m/%d/%Y %I:%M:%S %p")
    fhformatter = logging.Formatter("%(asctime)s : %(name)-12s %(levelname)-8s ==> %(message)s", "%m/%d/%Y %I:%M:%S %p")
    console.setFormatter(chformatter)
    file.setFormatter(fhformatter)

    root_logger.addHandler(console)
    root_logger.addHandler(file)
    
# pickle utils - save variable to filepath
def pickle_dump(var, path):
    with open(path, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)

# pickle utils - load variable from filepath
def pickle_load(path):
    with open(path, 'rb') as handle: 
        var = pickle.load(handle)
    return var

############################################################################
# DATASET AND GENERATION UTILS
############################################################################

# strip whitespaces
def preprocess(s):
    return ' '.join(str(s).split())

# strip whitespaces
def postprocess(s):
    return ' '.join(str(s).split())

# check if string is numeric
def isnum(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

# sample categorically from dataset (with equal number of samples per category)
def get_samples(dataset, batch = 127, category = -1, num_cats = 4, samples_per_cat = 30_000):
    low, high = 0, num_cats * samples_per_cat
    if category>=0:
        low = samples_per_cat * category
        high = low + samples_per_cat
    indices = np.random.randint(low, high, batch)
    return dataset[indices]

# create private prompt with private information in context
def get_prv_prompt(input, tokenizer, category='', score='', model='tinyllama1B', dataset='mimic'):
    prompt = PROMPTS[dataset]['prv']
    template = PROMPT_TEMPLATE.copy()
    template[1]['content'] = prompt.format(category, score, input)
    prompt_txt = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
    return prompt_txt

# create public prompt with no private information in context
def get_pub_prompt(tokenizer, category='', score='', model='tinyllama1B', dataset='mimic'):
    prompt = PROMPTS[dataset]['pub']
    template = PROMPT_TEMPLATE.copy()
    template[1]['content'] = prompt.format(category, score, input)
    prompt_txt = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
    return prompt_txt

# generator to create batches of lists from a single list
def batchify(lst, s, n):
    assert len(lst) >= n*s, "List too small for creating {} batches of size {}!".format(n, s)
    for i in range(0, n):
        yield lst[i*s : (i + 1)*s]

# Difference Clipping function
def clip_logit(logit, publogit, C):
    clipped = publogit + np.clip(logit - publogit, -C, C)
    return clipped

# Logit clipping function in Amin et al.(2024)
def clip_amin(logit, C):
    shift = logit + (C-np.max(logit, axis=1))[:, np.newaxis]
    return np.clip(shift, -1*C, C) 

# Top-K+ Sampling
def get_topk(pub, k, clip, batch):
    kthresh = np.partition(pub, -k)[-k]
    k_ext = kthresh - 2*clip/batch
    
    idxs = np.where(np.logical_and(pub>=k_ext, pub<=kthresh))[0]
    mask_pub = (pub>=k_ext)
    return mask_pub, idxs

############################################################################
# PRIVACY UTILS
############################################################################

# compute a rho-zCDP guarantee for InvisibleInk and Amin et al.(2024)
def compute_rho(T, C, s, tau, sig = 0):
    rho_prv = 0.5*(C/(s*tau))**2
    rho_pub = 2/(s*sig)**2 if sig>0  else 0
    rho_tot = T*(rho_prv + rho_pub)
    return rho_tot

# converting rho-zCDP to (eps,delta)-DP
# https://arxiv.org/pdf/2004.00010v3.pdf#page=13
def cdp_delta(rho, eps):
    assert (rho>=0 and eps>=0)
    if rho==0: return 0 
    
    amin, amax = 1.01, (eps+1)/(2*rho)+2
    for i in range(1000): #should be enough iterations
        alpha=(amin+amax)/2
        derivative = (2*alpha-1)*rho-eps+math.log1p(-1.0/alpha)
        if derivative<0: amin=alpha
        else: amax=alpha
    #now calculate delta
    delta = math.exp((alpha-1)*(alpha*rho-eps)+alpha*math.log1p(-1/alpha)) / (alpha-1.0)
    return min(delta,1.0) #delta<=1 always

# compute smallest eps such that rho-CDP implies (eps,delta)-DP
def cdp_eps(rho, delta = 1e-6):
    assert (rho>=0 and delta>0)
    if delta>=1 or rho==0: return 0.0 #if delta>=1 or rho=0 then anything goes
    
    # binary search to compute epsmax we use the standard bound
    epsmin, epsmax = 0.0, rho+2*math.sqrt(rho*math.log(1/delta)) 
    for i in range(1000):
        eps=(epsmin+epsmax)/2
        if cdp_delta(rho,eps)<=delta: epsmax=eps
        else: epsmin=eps
    return epsmax


# compute smallest rho such that rho-CDP implies (eps,delta)-DP
def cdp_rho(eps, delta=1e-6):
    assert (eps>=0 and delta>0)
    if delta>=1: return 0.0
    
    # binary search
    rhomin, rhomax = 0.0, eps+1 #maintain cdp_delta(rhomax,eps)>delta
    for i in range(1000):
        rho=(rhomin+rhomax)/2
        if cdp_delta(rho,eps)<=delta: rhomin=rho
        else: rhomax=rho
    return rhomin

# get (eps,delta)-DP guarantee from RDP guarantee
def rdp_eps(rdp, alpha, target_delta=1e-6):
    if target_delta**2 + math.expm1(-1*rdp) > 0:
        epsilon = 0
    elif alpha > 1.01:
        epsilon = rdp + math.log1p(-1/alpha) - math.log(target_delta*alpha)/(alpha-1)
    else:
        epsilon = np.inf
    return epsilon

# compute (eps,delta)-DP guarantee for InvisibleInk and Amin et al.(2024)
def get_epsilon(T, C, batch, tau, sig=0, delta=1e-6):
    rho = compute_rho(T, C, batch-1, tau, sig)
    eps = cdp_eps(rho, delta)
    return eps

# compute required clipping norm for InvisibleInk and Amin et al.(2024)
def get_clip(eps, T, tau, batch, sig=0, delta=1e-6):
    rho_tot = cdp_rho(eps, delta)
    rho_tok = rho_tot/T
    rho_pub = 2/((batch-1)*sig)**2 if sig>0  else 0
    if rho_pub >= rho_tok: 
        clip = np.inf
    else:
        rho_prv = rho_tok - rho_pub
        clip = tau*(batch-1)*np.sqrt(2*rho_prv)
    return clip

############################################################################
# ADAPMIXED UTILS to help implement Flemings et al (2024b)
############################################################################

# calculating mixing probabilities
def prob_mix(pub, prv, lamb = 1e-4):
    prv_mean = prv if len(prv.shape)==1 else np.mean(prv, axis=0)
    prv_mean = prv_mean/np.sum(prv_mean)
    pmix = prv_mean*lamb + pub*(1-lamb)
    return pmix/np.sum(pmix)
    
# noisy screening
def noisy_screen(pub, prv, topk=60, sigma=0.01):
    ind = np.argpartition(pub, -1*topk)[-1*topk:]
    pubk, prvk = pub[ind], prv[ind]
    noisek = np.random.normal(0, sigma, pubk.shape)
    prvk = prvk + noisek

    pubs = pubk/np.sum(pubk)
    prvs = prvk/np.sum(prvk)
    return pubs, prvs
    
# compute Renyi Divergence
def get_div(p, q, alpha=15, tol=1e-12):
    if np.isinf(alpha): return np.log(np.max(p / q))
    else:
        ratio = p**alpha * q**(1-alpha)
        sum_term = max(np.sum(ratio), tol)
        return 1.0 / (alpha - 1.0) * np.log(sum_term)

# compute symmetric Renyi Divergence
def renyi_div(p, q, alpha=15):
    p, q = p/np.sum(p), q/np.sum(q)
    return max(get_div(p, q, alpha), get_div(q, p, alpha))


############################################################################
# EMBEDDING UTILS
############################################################################

# calculate sentence-transformer embeddings
def embed_txts(txts, modelname = 'all-mpnet-base-v2', device = torch.device('cpu')):
    logger.info('Generating sentence-transformer embeddings ....')
    model = SentenceTransformer(modelname, device=device)
    embed = model.encode(txts, batch_size=256)
    logger.info('Embeddings Generated!')
    return embed
