import os
import json
import pickle
import logging
import pandas as pd
from tqdm import tqdm
import argparse

import utils
MODELS = utils.MODELS

logger = logging.getLogger(__name__)

# load yelp labels
yelp_labels1 = utils.yelp_labels1
yelp_labels2 = utils.yelp_labels2

# Prepare MIMIC-IV discharge instructions dataset
def prepare_dataset_mimic_instr(folder = './data/mimic', save=True):
    logger.info('Reading MIMIC-IV discharge notes ....')
    df = pd.read_csv(os.path.join(folder, 'discharge.csv'))
    prompt_txts = []
    
    logger.info('Searching for discharge summaries ....')
    for x in tqdm(df['text']):
        pos1 = x.find('Discharge Instructions')
        pos2 = x.find('\nFollow')
        if pos1>=0 and pos2>=0 and pos2>pos1: 
            prompt_txts.append(utils.preprocess(x[pos1:pos2]))
            
    if save: 
        logger.info('Saving MIMIC-IV discharge summaries ....')
        df_instr = pd.DataFrame({'text':prompt_txts})
        df_instr.to_csv(os.path.join(folder, 'discharge_instr.csv'))
    
    logger.info('Dataset ready! Total Entries: {}'.format(len(prompt_txts)))
    return prompt_txts

# prepare the Yelp reviews dataset
def prepare_dataset_yelp(folder = './data/yelp', save=True):
    logger.info('Reading yelp reviews ....')
    df = pd.read_csv(os.path.join(folder, 'train.csv'))
    prompt_txts = []
    
    logger.info('Sorting yelp reviews by category and score....')
    grouped = df.groupby(df.label1)
    for i, l1 in enumerate(yelp_labels1):
        df1 = grouped.get_group(l1)
        grouped2 = df1.groupby(df1.label2)

        for j, l2 in enumerate(yelp_labels2):
            df2 = grouped2.get_group(l2)
            df2 = df2.reset_index(drop=True)
            
            txts = []
            for x in tqdm(df2['text']):
                txts.append(utils.preprocess(x))
            
            df2['text'] = txts
            prompt_txts += list(df2['text'])
            df2.to_csv(os.path.join(folder, 'train_{}_{}.csv'.format(i, j)), index=False)
            
    if save: 
        logger.info('Saving yelp reviews ....')
        df_yelp = pd.DataFrame({'text':prompt_txts})
        df_yelp.to_csv(os.path.join(folder, 'yelp_reviews.csv'))
    
    logger.info('Dataset ready! Total Entries: {}'.format(len(prompt_txts)))
    return prompt_txts

# prepare the TAB-ECHR dataset
def prepare_dataset_tab(folder = './data/tab', save=True):
    logger.info('Reading TAB dataset ....')
    with open(os.path.join(folder, 'echr_train.json'), 'r') as f:
        data = json.load(f)
    
    prompt_txts = []
    logger.info('Tokenize and truncate strings to max_len tokens ....')
    for i in tqdm(range(len(data))):
        txt = data[i]['text']
        pos1 = txt.find('PROCEDURE')
        pos2 = txt.find('THE FACTS')
        txt2 = txt[pos1:pos2]
        if len(txt2) < 10: continue
        prompt_txts.append(utils.preprocess(txt2))
    
    if save: 
        logger.info('Saving TAB casefiles ....')
        df_yelp = pd.DataFrame({'text': prompt_txts})
        df_yelp.to_csv(os.path.join(folder, 'tab_procedure.csv'))
    
    logger.info('Dataset ready! Total Entries: {}'.format(len(prompt_txts)))
    return prompt_txts  

if __name__ == "__main__":
    ## ARGUMENTS
    parser = argparse.ArgumentParser(description='Dataset Creation')
    parser.add_argument('--sess', default='Dataset Creation', type=str, help='session name')
    parser.add_argument('--dataset', default='mimic', type=str, help='dataset name')
    parser.add_argument('--folder',  default='./data/mimic', type=str, help='dataset folder')
    parser.add_argument('--embed', action='store_true', help='whether to compute embeddings')
    parser.add_argument('--embed_model',  default='sentence-transformers/all-mpnet-base-v2', type=str, help='embedding model')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--gpu', default=0, type=int, help='allocate GPU, -1 for CPU execution')
    args = parser.parse_args()
    
    # setup random seed
    utils.setup_seed(args.seed)
    device = utils.setup_device(args.gpu)
    utils.setup_logging(os.path.join(args.folder,'dataprep_{}.log'.format(args.dataset)))
    logger = logging.getLogger(__name__)
    
    # prepare datasets
    if args.dataset == 'mimic': txts = prepare_dataset_mimic_instr(args.folder)
    elif args.dataset == 'yelp': txts = prepare_dataset_yelp(args.folder)
    elif args.dataset == 'tab': txts = prepare_dataset_tab(args.folder)
    else: raise ValueError("Invalid Dataset: Valid options are ['mimic', 'yelp', 'tab']")
    
    if args.embed:
        embed = utils.embed_txts(txts, modelname = args.embed_model, device = device)
        with open(os.path.join(args.folder,'embed_{}.pickle'.format(args.dataset)), 'wb') as handle:
            pickle.dump(embed, handle, protocol=pickle.HIGHEST_PROTOCOL)
    