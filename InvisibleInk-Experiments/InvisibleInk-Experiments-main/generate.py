'''Code to generate synthetic data using InvisibleInk (ours)'''

## IMPORT LIBRARIES
import os
import logging
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import special as spl
from scipy.optimize import brentq

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

import utils
import eval_utils
MODELS = utils.MODELS

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Reproduce results')
    parser.add_argument('--sess', default='InvisibleInk', type=str, help='session name')
    parser.add_argument('--method', default='invink', type=str, help='method name')
    
    # filepath and model arguments
    parser.add_argument('--model',  default='tinyllama1B', type=str, help='LLM name')
    parser.add_argument('--dataset', default='yelp', type=str, help='dataset name')
    parser.add_argument('--folder',  default='./results/', type=str, help='expt results folder')
    parser.add_argument('--embed_model', default='sentence-transformers/all-mpnet-base-v2', type=str, help='embedding model')
    
    # generation arguments (common)
    parser.add_argument('--num', default=50, type=int, help='number of synthetic generations')
    parser.add_argument('--eps', default=10, type=float, help='privacy budget (epsilon)')
    parser.add_argument('--delta', default=1e-6, type=float, help='delta (failure probability)')
    parser.add_argument('--tokens', default=500, type=int, help='total number of tokens')
    parser.add_argument('--batch', default=8, type=int, help='batch size for generation')
    parser.add_argument('--minibatch', default=16, type=int, help='minibatch size')
    
    # arguments for invink
    parser.add_argument('--temp', default=1.1, type=float, help='sampling temperature')
    parser.add_argument('--topk', default=-1, type=int, help='k for top-k+ sampling; -1 for full vocabulary')
    
    # util arguments
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--gpu', default=0, type=int, help='allocate GPU, -1 for CPU execution')
    parser.add_argument('--write_every', default=10, type=int, help='number of iterations')
    args = parser.parse_args()
    
    # HF login
    login(os.environ["HF_TOKEN"])
    
    # calculate clipping threshold (note: batch size in the code includes public prompt)
    clip = utils.get_clip(args.eps, args.tokens, args.temp, args.batch)
    if clip == np.inf: raise ValueError('Clipping norm must be finite! Recommended: Increase epsilon (privacy budget) or batch size.')
    
    # get folder and expt names [results/dataset/method]
    expt_name = args.model + '-eps_{}-batch_{}-tokens_{}-topk_{}-clip_{:.3f}-temp_{:.1f}'
    expt_folder = os.path.join(os.path.join(args.folder, args.dataset), args.method)
    expt_name = expt_name.format(args.eps, args.batch, args.tokens, args.topk, clip, args.temp)
    outfolder = os.path.join(expt_folder, expt_name)
    if not os.path.exists(outfolder): os.makedirs(outfolder)

    # set device, random seed and logging
    utils.setup_seed(args.seed)
    device = utils.setup_device(args.gpu)
    utils.setup_logging(os.path.join(outfolder,'datagen_{}.log'.format(args.seed)))
    logger = logging.getLogger(__name__)
    
    # log stuff
    logger.info(args)
    logger.info('Seed: {}'.format(args.seed))
    logger.info('Device: {}'.format(device))
    logger.info('PID: {}'.format(os.getpid()))
    logger.info('Clip: {}\n'.format(clip))

    # download tokenizer and model after login to HF
    logger.info('Downloading LLM (model + tokenizer): {} ....'.format(args.model))
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, padding_side='left', truncation_side='right')
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map=device, attn_implementation="eager")
    tokenizer.pad_token = tokenizer.eos_token
    logger.info('Model Ready!\n')
    vocab = model.vocab_size
    
    # set topk parameter for full vocabulary setting
    if args.topk < 0: args.topk = vocab
    
    # adjust csv writer frequency
    if args.write_every > args.num: args.write_every = args.num

    # load dataset and batchify
    logger.info('Get dataset: {} ....'.format(args.dataset))
    datapath = os.path.join('./data', args.dataset)
    if args.dataset=='mimic': real_txt_path = os.path.join(datapath, 'discharge_instr.csv')
    elif args.dataset=='yelp': real_txt_path = os.path.join(datapath, 'train.csv')
    elif args.dataset=='tab': real_txt_path = os.path.join(datapath, 'tab_procedure.csv')
    else: raise ValueError('Invalid dataset name!')
    if args.dataset=='yelp':
        # load yelp labels
        yelp_labels1 = utils.yelp_labels1
        yelp_labels2 = utils.yelp_labels2
        txts, labels = [], []
        
        for i, l1 in enumerate(yelp_labels1):
            for j, l2 in enumerate(yelp_labels2):
                if not os.path.exists(os.path.join(datapath, 'train_{}_{}.csv'.format(i, j))):
                    raise ValueError('Dataset not found! Please run dataset.py to create the dataset')
                
                # number of samples should be divisible by 50
                num_per_cat = (args.num * (args.batch-1)) // (len(yelp_labels1)*len(yelp_labels2))
                num_batches = num_per_cat // (args.batch-1)
                
                df = pd.read_csv(os.path.join(datapath, 'train_{}_{}.csv'.format(i, j)), nrows = num_per_cat)
                txts_cat = list(utils.batchify(df['text'], s = args.batch-1, n = num_batches))
                labels_cat = [(i, j) for _ in range(num_batches)]
                
                txts += txts_cat
                labels += labels_cat
    else:
        if os.path.exists(real_txt_path): 
            df = pd.read_csv(real_txt_path, nrows = args.batch*args.num)
        else: raise ValueError('Enter a valid path to a dataset')
        txts = list(utils.batchify(df['text'], s = args.batch-1, n = args.num))
    logger.info('Dataset Ready!\n')
    
    # setup minibatching for execution
    if args.minibatch > args.batch: args.minibatch = args.batch
    minibatches = args.batch // args.minibatch
    assert (args.batch%args.minibatch == 0), "Minibatch size {} must perfectly divide batch size {}!".format(args.minibatch, args.batch)
    
    # store results in this variable
    results = {
        'token_seq': [],
        'text': [],
        'len': [],
        'embed': {},
        'eps': [],
        'topk_avg': [],
        'topk_std': [],
        'ext': [],
        'delta': args.delta,
    }

    # iterate over different generations
    write_header = True
    logger.info('Beginning Generation ....')
    for i in tqdm(range(args.num)):
        txt_batch = txts[i]                                                     # get batch of texts
        cache = [None]*(minibatches)                                               # past key_values
        token_seq = torch.tensor([], dtype=int, device=device)                  # selected token sequence
        
        # add category information for conditional generation with Yelp
        if args.dataset == 'yelp':
            cat, score = labels[i]
            batch_prompts = [utils.get_prv_prompt(txt, tokenizer, yelp_labels1[cat], yelp_labels2[score], model=args.model, dataset=args.dataset) for txt in txt_batch]    # get prompts from texts
            batch_prompts.append(utils.get_pub_prompt(tokenizer, yelp_labels1[cat], yelp_labels2[score], model=args.model, dataset=args.dataset))                          # add public prompt
            
        # carry out unconditional generation for other datasets
        else:  
            batch_prompts = [utils.get_prv_prompt(txt, tokenizer, model=args.model, dataset=args.dataset) for txt in txt_batch]    # get prompts from texts
            batch_prompts.append(utils.get_pub_prompt(tokenizer, model=args.model, dataset=args.dataset))                          # add public prompt
            
        # get minibatches of encoded prompts
        encoded = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(device)           # encode all prompts
        minibatch_masks = list(torch.split(encoded.attention_mask, args.minibatch))
        minibatch_tokens = list(torch.split(encoded.input_ids, args.minibatch))                                     # get minibatches of prompts
        
        # generate token by token
        counter = 0
        topk_counts, ext_count = [], 0
        for _ in range(args.tokens):
            logits = np.zeros((args.batch, vocab))
            
            # iterate over minibatches
            for j in range(minibatches):
                # get minibatch of prompt tokens and append generated token sequence to it
                masks = minibatch_masks[j]
                prompts = minibatch_tokens[j]
                low, high = j*args.minibatch, (j+1)*args.minibatch
                token_seq_cast = torch.broadcast_to(token_seq, (prompts.shape[0], token_seq.shape[0]))                  
                mask_append = torch.cat((masks, torch.ones_like(token_seq_cast)), 1)
                prompt_append = torch.cat((prompts, token_seq_cast), 1)
                
                # generate outputs and store the logits and past key value pairs for future use
                output = model.generate(prompt_append, past_key_values = cache[j], use_cache=True,
                                        max_new_tokens = 1, pad_token_id = tokenizer.eos_token_id, attention_mask = mask_append,
                                        do_sample=True, temperature=args.temp, top_k=vocab, top_p=1.0,
                                        output_logits=True, return_dict_in_generate=True)
                
                # save only logits and KV cache
                logits[low:high, :] = output.logits[0].cpu().numpy()
                cache[j] = output.past_key_values
            
            # clear cache
            del output
            torch.cuda.empty_cache()

            # get pub/prv logits clip using DClip and average the clipped logits
            pub_logits, prv_logits = logits[-1], logits[:-1]
            clipped_logits = utils.clip_logit(prv_logits, pub_logits, clip)
            avg_clip_logits = np.mean(clipped_logits, axis=0)

            # get next token
            pub_mask, idxs = utils.get_topk(pub_logits, args.topk, clip, args.batch-1)
            avg_clip_logits = np.where(pub_mask, avg_clip_logits, -np.inf)
            topk_counts.append(np.sum((pub_mask)))
            
            # get next token sampled
            probs = spl.softmax(avg_clip_logits/args.temp)
            nxt_token = np.random.choice(np.arange(vocab), p = probs)
            token_seq = torch.cat((token_seq, torch.tensor([nxt_token], device=device)))
            if nxt_token in idxs: ext_count += 1
            counter += 1
            
            # break loop if EOS is encountered
            if (nxt_token == model.generation_config.eos_token_id).any():
                break
        
        # store results in a dictionary
        results['text'].append(utils.postprocess(tokenizer.decode(token_seq, skip_special_tokens=True)))
        results['token_seq'].append(token_seq.cpu().numpy())
        results['topk_avg'].append(np.mean(topk_counts))
        results['topk_std'].append(np.std(topk_counts))
        results['ext'].append(np.mean(ext_count))
        results['len'].append(counter)
        
        # calculate the data-depenedent privacy guarantees for the generated sequence
        rho_calc = utils.compute_rho(counter, clip, args.batch-1, args.temp)
        eps_calc = utils.cdp_eps(rho_calc, args.delta)
        results['eps'].append(eps_calc)
        
        # save results to a csv file every few iterations
        if (i+1)%args.write_every == 0:
            res_dict = {}
            indices = list(range(i-args.write_every+1, i+1))
            res_dict['text'] = results['text'][-1*args.write_every:]
            res_dict['token_seq'] = results['token_seq'][-1*args.write_every:]
            res_dict['topk_avg'] = results['topk_avg'][-1*args.write_every:]
            res_dict['topk_std'] = results['topk_std'][-1*args.write_every:]
            res_dict['ext'] = results['ext'][-1*args.write_every:]
            res_dict['len'] = results['len'][-1*args.write_every:]
            res_dict['eps'] = results['eps'][-1*args.write_every:]
            writer_df = pd.DataFrame(res_dict, index=indices)
            writer_df.to_csv(os.path.join(outfolder,'data_{}.csv'.format(args.seed)), header = write_header, mode = 'a')
            write_header = False
    
    # calculate and store embeddings
    logger.info('Generation Complete!')
    reload_txts = list(pd.read_csv(os.path.join(outfolder,'data_{}.csv'.format(args.seed))).fillna(' ')['text'])
    results['embed'][args.embed_model] = utils.embed_txts(reload_txts, modelname = args.embed_model, device = device)
    utils.pickle_dump(results, os.path.join(outfolder, 'results_{}.pickle'.format(args.seed)))
    
    # begin evaluation
    logger.info('\n')
    logger.info('Beginning Evaluation ....')
    
    # reload the dataset and real embeddings
    real_txt = list(eval_utils.read_csv_last(real_txt_path, args.num)['text'])
    if os.path.exists(os.path.join(datapath, 'embed_{}.pickle'.format(args.dataset))):
        real_embed = utils.pickle_load(os.path.join(datapath, 'embed_{}.pickle'.format(args.dataset)))
    else: real_embed = utils.embed_txts(real_txt, modelname = args.embed_model, device = device)
    
    # get synthetic texts and embeddings
    syn_txt = reload_txts
    syn_embed = results['embed'][args.embed_model]
    
    # calculate mauve scores
    mauve_score = eval_utils.compute_mauve(real_embed, syn_embed, num = args.num)
    logging.info('MAUVE Score: {:.4f}\n'.format(mauve_score))
    
    # calculate medNER counts
    ner_counts = eval_utils.compute_ner(syn_txt, device = device)
    logging.info('MedNER Counts - Mean: {:.4f}, Std: {:.4f}\n'.format(np.mean(ner_counts), np.std(ner_counts)))
    
    # calculate PPL scores
    real_ppl = eval_utils.compute_ppl(real_txt, model_id='gpt2', batch_size = 8, device=device, max_length=512)
    syn_ppl = eval_utils.compute_ppl(syn_txt, model_id='gpt2', batch_size = 8, device=device, max_length=512)
    diff_ppl = np.abs(syn_ppl['mean_perplexity'] - real_ppl['mean_perplexity'])
    logging.info('DeltaPPL (GenPPL - RealPPL): {:.4f}\n'.format(diff_ppl))
    
    # additional important statistics
    logging.info('Generation Length - Mean: {:.4f}, Std: {:.4f}'.format(np.mean(results['len']), np.std(results['len'])))
    logging.info('Maximum data-dependent epsilon: {:.4f}\n'.format(np.max(results['eps'])))
    
    # top-k+ sampling statistics for Invisible Ink
    topk_mean, topk_std = eval_utils.combined_mean_std(results['topk_avg'], results['topk_std'], results['len'])
    logging.info('Top-k Counts - Mean: {:.4f}, Std: {:.4f}'.format(topk_mean, topk_std))
    logging.info('Tokens sampled from expansion set - Mean: {:.4f} Std: {:.4f}\n'.format(np.mean(results['ext']), np.std(results['ext'])))
    
    # end
    logger.info('Evaluation Complete!')