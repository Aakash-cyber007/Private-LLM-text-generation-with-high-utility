"""
generate_paf.py
===============
Differentially private text generation using the
**Permute-and-Flip (PaF)** mechanism (McKenna & Sheldon, 2020).

Theory
------
At each autoregressive step we have a score vector  s ∈ ℝ^|V|  built by
difference-clipping private logits toward the public logit baseline:

    s[t] = (1/n) Σ_i DClip(ℓ_i[t] − ℓ_pub[t], C) + ℓ_pub[t]

The **exponential mechanism** would sample token t* ∝ exp(s[t] / Δ),
where Δ = 2C/n is the ℓ∞ sensitivity.

**Permute-and-Flip** is an equivalent, but *utility-superior* sampler:
    1.  Draw a uniform random permutation π of {1,…,|V|}.
    2.  Walk tokens in order of π.
    3.  For each candidate token t in that order, flip a biased coin:
            accept t  with probability  p_t / p_max
        where p_t = exp(s[t] / Δ) and p_max = max_t p_t.
    4.  Return the first accepted token.

The efficient implementation below avoids an explicit loop:
    - Scale scores by 1/Δ and subtract the max for numerical stability.
    - Compute "relative weights" w_t = exp(scaled_score_t).
    - Generate uniform arrival times u_t ~ Uniform[0, 1] per token.
    - The PaF winner is  argmin_t (u_t / w_t)  over the allowed set.
      (This is the Gumbel-max / reservoir-sampling equivalence.)

This is provably identical in distribution to PaF and is O(|V|) per step.

Privacy
-------
Identical to InvisibleInk:  the same DClip → same sensitivity Δ = 2C/n
→ same zCDP parameter ρ = T·(C/(n·τ))²/2 → same (ε,δ)-DP guarantee.
"""

import os, sys, logging, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

import utils_paf as U

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# PaF token selection
# ─────────────────────────────────────────────────────────────

def paf_select(avg_clip_logits: np.ndarray,
               pub_mask: np.ndarray,
               clip: float,
               n_private: int,
               temp: float) -> int:
    """
    Efficient Permute-and-Flip via the Gumbel-max trick.

    Parameters
    ----------
    avg_clip_logits : shape (vocab,) — averaged difference-clipped logits
    pub_mask        : shape (vocab,) — boolean, True = token is in top-k+ set
    clip            : clipping norm C
    n_private       : number of private texts in batch  (= batch - 1)
    temp            : sampling temperature τ

    Returns
    -------
    int  — selected token index
    """
    sensitivity = 2.0 * clip / (n_private * temp)   # Δ = 2C / (n·τ)

    # Scale scores; apply top-k+ mask (disallowed → -∞)
    scores = np.where(pub_mask, avg_clip_logits / temp, -np.inf)

    # Subtract max for numerical stability (doesn't change argmin below)
    max_s = np.max(scores[pub_mask]) if pub_mask.any() else 0.0
    scores = scores - max_s

    # Relative weights  w_t = exp(s_t / sensitivity)
    # (sensitivity cancels with the temp scaling above — kept explicit for clarity)
    weights = np.exp(scores / sensitivity)          # shape (vocab,)
    weights = np.where(pub_mask, weights, 0.0)      # zero out masked tokens

    # Uniform arrival times u_t ~ U[0,1]; PaF winner = argmin u_t / w_t
    # Equivalently: argmax  (log w_t - log u_t)  = argmax  (s_t/Δ + Gumbel)
    u = np.random.uniform(0.0, 1.0, len(weights))
    # guard against w=0 at masked positions
    with np.errstate(divide='ignore', invalid='ignore'):
        key = np.where(pub_mask, np.log(weights) - np.log(u), -np.inf)

    return int(np.argmax(key))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DP text generation — Permute-and-Flip')

    # I/O
    parser.add_argument('--model',       default='tinyllama1B')
    parser.add_argument('--folder',      default='./results/')
    parser.add_argument('--embed_model', default='sentence-transformers/all-mpnet-base-v2')

    # privacy & generation
    parser.add_argument('--eps',       default=10,   type=float)
    parser.add_argument('--delta',     default=1e-6, type=float)
    parser.add_argument('--batch',     default=8,    type=int,
                        help='Total batch size (private + 1 public prompt)')
    parser.add_argument('--minibatch', default=4,    type=int,
                        help='GPU minibatch size; reduce if OOM')
    parser.add_argument('--num',       default=50,   type=int,
                        help='Number of synthetic texts to generate')
    parser.add_argument('--tokens',    default=200,  type=int,
                        help='Max tokens per generation')
    parser.add_argument('--temp',      default=1.2,  type=float)
    parser.add_argument('--topk',      default=50,   type=int,
                        help='k for Top-k+ mask; -1 = full vocabulary')

    # util
    parser.add_argument('--seed', default=42,  type=int)
    parser.add_argument('--gpu',  default=0,   type=int)
    parser.add_argument('--write_every', default=10, type=int)
    args = parser.parse_args()

    # ── 0. HF auth ──────────────────────────────────────────
    token = os.environ.get('HF_TOKEN', '')
    if not token:
        print('\n' + '='*60)
        print('  HF_TOKEN not set.')
        print('  Run:  export HF_TOKEN="hf_..."  then retry.')
        print('='*60 + '\n')
        sys.exit(1)
    login(token)

    # ── 1. Privacy accounting ────────────────────────────────
    clip = U.get_clip(args.eps, args.tokens, args.temp, args.batch, args.delta)
    if not np.isfinite(clip):
        raise ValueError(
            'Clipping norm is ∞ — increase --eps or --batch.')
    effective_eps = U.get_epsilon(
        args.tokens, clip, args.batch, args.temp, args.delta)

    # ── 2. Output folder ─────────────────────────────────────
    tag = (f'{args.model}-eps{args.eps}-b{args.batch}'
           f'-T{args.tokens}-k{args.topk}-t{args.temp}')
    outfolder = os.path.join(args.folder, 'paf', tag)
    os.makedirs(outfolder, exist_ok=True)

    # ── 3. Setup ─────────────────────────────────────────────
    U.setup_seed(args.seed)
    device = U.setup_device(args.gpu)
    U.setup_logging(os.path.join(outfolder, f'run_{args.seed}.log'))
    logger = logging.getLogger(__name__)

    logger.info(f'Args: {args}')
    logger.info(f'Clipping norm C = {clip:.4f}')
    logger.info(f'Effective ε = {effective_eps:.4f}  (target {args.eps})')
    logger.info(f'Device: {device}')

    # ── 4. Load model ────────────────────────────────────────
    model_id = U.MODELS[args.model]
    logger.info(f'Loading {args.model} …')
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, padding_side='left', truncation_side='right')
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation='eager')
    tokenizer.pad_token = tokenizer.eos_token
    vocab = model.config.vocab_size
    logger.info(f'Model ready. Vocab size = {vocab:,}')

    topk = vocab if args.topk < 0 else args.topk
    write_every = min(args.write_every, args.num)

    # ── 5. Load Yelp dataset ─────────────────────────────────
    data_root = './data/yelp'
    txts, labels = [], []
    for i, cat in enumerate(U.YELP_CATS):
        for j, stars in enumerate(U.YELP_STARS):
            csv_path = os.path.join(data_root, f'train_{i}_{j}.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f'{csv_path} not found — run setup.py first.')
            n_per_cat = max(
                1, (args.num * (args.batch - 1))
                // (len(U.YELP_CATS) * len(U.YELP_STARS)))
            n_batches = max(1, n_per_cat // (args.batch - 1))
            df = pd.read_csv(csv_path, nrows=n_per_cat * 2)
            df['text'] = df['text'].fillna('').apply(U.preprocess)
            txts  += list(U.batchify(df['text'].tolist(),
                                     s=args.batch - 1, n=n_batches))
            labels += [(i, j)] * n_batches

    total_gens = min(args.num, len(txts))
    logger.info(f'Dataset ready — using {total_gens} batches.')

    # minibatch sanity
    mb = min(args.minibatch, args.batch)
    while args.batch % mb != 0:
        mb -= 1
    n_mb = args.batch // mb

    # ── 6. Generation loop ───────────────────────────────────
    results = dict(
        text=[], token_seq=[], length=[],
        eps_actual=[], topk_avg=[], topk_std=[], ext_frac=[])

    write_header = True
    logger.info('Starting PaF generation …')

    for gi in tqdm(range(total_gens)):
        txt_batch = txts[gi]
        cat_i, star_j = labels[gi]

        # build prompts
        prv_prompts = [
            U.get_prv_prompt(t, tokenizer,
                             cat=U.YELP_CATS[cat_i],
                             stars=U.YELP_STARS[star_j])
            for t in txt_batch]
        pub_prompt = U.get_pub_prompt(tokenizer,
                                      cat=U.YELP_CATS[cat_i],
                                      stars=U.YELP_STARS[star_j])
        all_prompts = prv_prompts + [pub_prompt]   # pub is LAST

        encoded = tokenizer(
            all_prompts, return_tensors='pt',
            padding=True, truncation=True).to(device)
        mb_masks  = list(torch.split(encoded.attention_mask, mb))
        mb_tokens = list(torch.split(encoded.input_ids, mb))

        token_seq = torch.tensor([], dtype=torch.long, device=device)
        cache     = [None] * n_mb
        topk_cnts, ext_cnt, step = [], 0, 0

        for _ in range(args.tokens):
            logits = np.zeros((args.batch, vocab), dtype=np.float32)

            for j in range(n_mb):
                lo, hi = j * mb, (j + 1) * mb
                pmts = mb_tokens[j]
                msks = mb_masks[j]
                tsc  = torch.broadcast_to(
                    token_seq, (pmts.shape[0], token_seq.shape[0]))
                out = model.generate(
                    torch.cat([pmts, tsc], 1),
                    past_key_values=cache[j], use_cache=True,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.cat(
                        [msks, torch.ones_like(tsc)], 1),
                    do_sample=True, temperature=args.temp,
                    top_k=vocab, top_p=1.0,
                    output_logits=True, return_dict_in_generate=True)
                logits[lo:hi] = out.logits[0].cpu().float().numpy()
                cache[j] = out.past_key_values

            del out
            torch.cuda.empty_cache()

            pub_logits = logits[-1]           # last prompt = public
            prv_logits = logits[:-1]          # all others = private

            # difference clipping
            clipped = U.clip_logit(prv_logits, pub_logits, clip)
            avg_clipped = clipped.mean(axis=0)

            # top-k+ mask
            mask, ext_idxs = U.get_topkplus_mask(
                pub_logits, topk, clip, args.batch - 1)
            topk_cnts.append(int(mask.sum()))

            # ── PaF token selection ──────────────────────────
            nxt = paf_select(avg_clipped, mask, clip,
                             args.batch - 1, args.temp)

            token_seq = torch.cat(
                [token_seq, torch.tensor([nxt], device=device)])
            if nxt in ext_idxs:
                ext_cnt += 1
            step += 1

            eos = model.generation_config.eos_token_id
            if isinstance(eos, list):
                if nxt in eos:
                    break
            elif nxt == eos:
                break

        gen_text = U.postprocess(
            tokenizer.decode(token_seq, skip_special_tokens=True))
        rho = U.compute_rho(step, clip, args.batch - 1, args.temp)
        eps_actual = U.cdp_eps(rho, args.delta)

        results['text'].append(gen_text)
        results['token_seq'].append(token_seq.cpu().numpy())
        results['length'].append(step)
        results['eps_actual'].append(eps_actual)
        results['topk_avg'].append(float(np.mean(topk_cnts)))
        results['topk_std'].append(float(np.std(topk_cnts)))
        results['ext_frac'].append(ext_cnt / max(step, 1))

        if (gi + 1) % write_every == 0:
            lo = gi - write_every + 1
            chunk = {k: results[k][-write_every:] for k in results
                     if k != 'token_seq'}
            chunk['token_seq'] = results['token_seq'][-write_every:]
            pd.DataFrame(chunk, index=range(lo, gi + 1)).to_csv(
                os.path.join(outfolder, f'data_{args.seed}.csv'),
                header=write_header, mode='a')
            write_header = False

    # ── 7. Embeddings ────────────────────────────────────────
    logger.info('Computing embeddings …')
    all_texts = list(
        pd.read_csv(os.path.join(outfolder, f'data_{args.seed}.csv'))
        .fillna('')['text'])
    embed = U.embed_texts(all_texts, args.embed_model, device)
    results['embed'] = embed
    U.pickle_dump(results,
                  os.path.join(outfolder, f'results_{args.seed}.pkl'))

    logger.info(f'Done!  Generated {len(all_texts)} texts.')
    logger.info(f'Mean length  : {np.mean(results["length"]):.1f} tokens')
    logger.info(f'Max ε actual : {max(results["eps_actual"]):.4f}')
    logger.info(f'Ext-set frac : {np.mean(results["ext_frac"]):.4f}')
