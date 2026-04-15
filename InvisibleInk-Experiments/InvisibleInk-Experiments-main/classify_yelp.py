import os
import glob
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, top_k_accuracy_score

import utils

# Yelp Dataset class
class YelpDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
# get list of evaluation folders for a specific model
def get_eval_flist(folder, model):
    flist = [f for f in os.listdir(folder) if (os.path.isdir(os.path.join(folder, f)) and (model in f))]
    flist.sort()
    return flist

# get all csv filenames in a folder
def get_csv_files(folder_path):
    return [os.path.basename(f) for f in glob.glob(os.path.join(folder_path, "*.csv"))]

# get hyperparameters from folder name
def get_hparams(f, hp_list):
    ls = f.split('-')
    hparams = {}
    for l in ls:
        pairs = l.split('_')
        if len(pairs) == 2 and pairs[0] in hp_list:
            hparams[pairs[0]] = float(pairs[1]) if utils.isnum(pairs[1]) else pairs[1]
    return hparams

# create mapping from category-star combinations to class indices
def create_label_mapping():
    # Extract categories from yelp_labels1 (remove 'Business Category: ' prefix)
    categories = [label.replace('Business Category: ', '') for label in utils.yelp_labels1]
    # Extract stars from yelp_labels2 (remove 'Review Stars: ' prefix)
    stars = [label.replace('Review Stars: ', '') for label in utils.yelp_labels2]
    
    label_to_idx = {}
    idx_to_label = {}
    idx = 0
    
    for category in categories:
        for star in stars:
            combined_label = f"{category}_{star}"
            label_to_idx[combined_label] = idx
            idx_to_label[idx] = combined_label
            idx += 1
    
    return label_to_idx, idx_to_label

# Load YELP dataset from CSV files and create 50-class labels
def load_yelp_data(train_path=None, test_path=None, label_to_idx=None, idx_to_label=None):
    train_texts, train_labels = None, None
    if train_path and os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        train_texts = train_df['text'].apply(utils.preprocess).tolist()
        if len(train_texts)%50!=0:
            raise ValueError(f"Expected 50N training samples, got {len(train_texts)}")
        skipping = len(train_texts)//50
        train_labels = [i//skipping for i in range(len(train_texts))]
    
    # Create combined labels
    def create_combined_label(row):
        category = row['label1'].replace('Business Category: ', '')
        star = str(float(row['label2'].replace('Review Stars: ', '')))  # Ensure star is in format like "1.0"
        combined = f"{category}_{star}"
        return label_to_idx.get(combined, -1)  # Return -1 for unknown combinations
    
    test_texts, test_labels = None, None
    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        test_df['combined_label'] = test_df.apply(create_combined_label, axis=1)
        test_df = test_df[test_df['combined_label'] != -1]
        
        test_texts = test_df['text'].apply(utils.preprocess).tolist()
        test_labels = test_df['combined_label'].tolist()
    
    return train_texts, train_labels, test_texts, test_labels

# train model for one epoch
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, logger):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct_predictions/total_samples:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    
    logger.info(f'Epoch {epoch} - Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy

# Evaluate the model with comprehensive metrics for 50-class classification
def evaluate_model(model, dataloader, device, logger, idx_to_label, split_name="Test"):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Evaluating {split_name}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # Top-k accuracies for multi-class
    top_3_acc = top_k_accuracy_score(all_labels, all_logits, k=3)
    top_5_acc = top_k_accuracy_score(all_labels, all_logits, k=5)
    top_10_acc = top_k_accuracy_score(all_labels, all_logits, k=10)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Log main metrics
    logger.info(f'{split_name} Results:')
    logger.info(f'{split_name} Loss: {avg_loss:.4f}')
    logger.info(f'{split_name} Accuracy: {accuracy:.4f}')
    logger.info(f'{split_name} Top-3 Accuracy: {top_3_acc:.4f}')
    logger.info(f'{split_name} Top-5 Accuracy: {top_5_acc:.4f}')
    logger.info(f'{split_name} Top-10 Accuracy: {top_10_acc:.4f}')
    logger.info(f'{split_name} Weighted Precision: {precision:.4f}')
    logger.info(f'{split_name} Weighted Recall: {recall:.4f}')
    logger.info(f'{split_name} Weighted F1: {f1:.4f}')
    
    # Detailed classification report
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    class_report = classification_report(all_labels, all_predictions, target_names=class_names, zero_division=0)
    logger.info(f'{split_name} Classification Report:\n{class_report}')
    
    # Log per-category and per-star performance
    category_performance = {}
    star_performance = {}
    
    for i, (prec, rec, f1_score, supp) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class, support_per_class)):
        label_name = idx_to_label[i]
        category, star = label_name.split('_')
        
        if category not in category_performance:
            category_performance[category] = {'precision': [], 'recall': [], 'f1': [], 'support': []}
        if star not in star_performance:
            star_performance[star] = {'precision': [], 'recall': [], 'f1': [], 'support': []}
        
        category_performance[category]['precision'].append(prec)
        category_performance[category]['recall'].append(rec)
        category_performance[category]['f1'].append(f1_score)
        category_performance[category]['support'].append(supp)
        
        star_performance[star]['precision'].append(prec)
        star_performance[star]['recall'].append(rec)
        star_performance[star]['f1'].append(f1_score)
        star_performance[star]['support'].append(supp)
    
    # Log category-wise performance
    logger.info(f'{split_name} Category-wise Performance:')
    for category, metrics in category_performance.items():
        avg_prec = np.mean(metrics['precision'])
        avg_rec = np.mean(metrics['recall'])
        avg_f1 = np.mean(metrics['f1'])
        total_supp = np.sum(metrics['support'])
        logger.info(f'  {category}: Precision={avg_prec:.4f}, Recall={avg_rec:.4f}, F1={avg_f1:.4f}, Support={total_supp}')
    
    # Log star-wise performance
    logger.info(f'{split_name} Star-wise Performance:')
    for star, metrics in star_performance.items():
        avg_prec = np.mean(metrics['precision'])
        avg_rec = np.mean(metrics['recall'])
        avg_f1 = np.mean(metrics['f1'])
        total_supp = np.sum(metrics['support'])
        logger.info(f'  {star} stars: Precision={avg_prec:.4f}, Recall={avg_rec:.4f}, F1={avg_f1:.4f}, Support={total_supp}')
    
    # Confusion matrix (only log, don't print due to size)
    cm = confusion_matrix(all_labels, all_predictions)
    logger.info(f'{split_name} Confusion Matrix Shape: {cm.shape}')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'top_3_accuracy': top_3_acc,
        'top_5_accuracy': top_5_acc,
        'top_10_accuracy': top_10_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'logits': all_logits,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'category_performance': category_performance,
        'star_performance': star_performance,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RoBERTa YELP 50-Class Classification Finetuning')
    parser.add_argument('--name', default='roberta_yelp50class', type=str, help='experiment name')
    parser.add_argument('--dataset', default='yelp', type=str, help='dataset name')
    parser.add_argument('--model',  default='tinyllama1B', type=str, help='LLM name')
    parser.add_argument('--method', default='invink', type=str, help='method name')
    parser.add_argument('--folder', default='./results', type=str, help='path to training folder')
    parser.add_argument('--test_path', default='./data/yelp/test.csv', type=str, help='path to test dataset')
    parser.add_argument('--model_name', default='roberta-base', type=str, help='RoBERTa model name')
    
    parser.add_argument('--num_train', default=50, type=int, help='number of training samples')
    parser.add_argument('--max_length', default=512, type=int, help='maximum sequence length')
    parser.add_argument('--batch_size', default=20, type=int, help='batch size')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--num_epochs', default=15, type=int, help='number of training epochs')
    parser.add_argument('--warmup_frac', default=0.1, type=float, help='warmup steps fraction')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--gpu', default=0, type=int, help='GPU device, -1 for CPU')
    
    args = parser.parse_args()
    
    
    # Create output directory
    expt_folder = os.path.join(os.path.join(args.folder, args.dataset), args.method)
    
    # Setup seed and logging
    utils.setup_seed(args.seed)
    device = utils.setup_device(args.gpu)
    utils.setup_logging(os.path.join(expt_folder,'{}_{}.log'.format(args.name, args.model)))
    save_path = os.path.join(expt_folder, '{}_{}.pickle'.format(args.name, args.model))
    logger = logging.getLogger(__name__)
    
    # log basic stuff
    logger.info(args)
    logger.info('Seed: {}'.format(args.seed))
    logger.info('Device: {}'.format(device))
    logger.info('PID: {}'.format(os.getpid()))
    logger.info('Dataset: {}'.format(args.dataset))
    logging.info('Model: {}'.format(args.model))
    logger.info("Starting RoBERTa YELP 50-class classification finetuning")
    
    # Create label mapping
    label_to_idx, idx_to_label = create_label_mapping()
    
    # Load test dataset
    logger.info("Loading YELP Test dataset for 50-class classification...")
    _, _, test_texts, test_labels = load_yelp_data(None, args.test_path, label_to_idx, idx_to_label)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    test_dataloader = None
    if test_texts:
        test_dataset = YelpDataset(test_texts, test_labels, tokenizer, args.max_length)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    logger.info(f"Test samples: {len(test_texts)}")
    logger.info(f"Number of classes: {len(label_to_idx)}")
    
    
    # set list of relevant hyperparameters for each method
    hp_list = None
    if args.method == 'adapmixed': hp_list = ['eps', 'batch']
    elif args.method == 'amin': hp_list = ['eps', 'batch', 'clip', 'temp']
    elif args.method == 'invink': hp_list = ['eps', 'batch', 'topk', 'clip', 'temp']
    else: raise ValueError('Invalid method name!')
    
    
    # get all files in the folder
    eval_results = []
    flist = get_eval_flist(expt_folder, args.model)
    
    # go through list of folders and keep evaluating
    for f in flist:
        
        # print hyperparams and get csv list
        hparam = get_hparams(f, hp_list)
        csvlist = get_csv_files(os.path.join(expt_folder, f))
        logging.info(hparam)
        
        for csvfile in csvlist:
            logging.info('Evaluating synthetic data from file: {}'.format(csvfile))    
            train_path = os.path.join(expt_folder, os.path.join(f, csvfile))
            
            # skip invalid paths
            if os.path.exists(train_path) == False:
                logging.info('Synthetic data not found! Skipping ...')
                continue
            
            train_texts, train_labels, _, _= load_yelp_data(train_path, None, label_to_idx, idx_to_label)
            logger.info(f"Train samples: {len(train_texts)}")
            
            # Initialize tokenizer and model
            logger.info(f"Loading {args.model_name} tokenizer and model...")
            model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=50)
            model.to(device)
            
            # Log model info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
            # Create datasets and dataloaders
            train_dataset = YelpDataset(train_texts, train_labels, tokenizer, args.max_length)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            
            
            # Setup optimizer and scheduler
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            total_steps = len(train_dataloader) * args.num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(args.warmup_frac * total_steps),  # 10% of total steps
                num_training_steps=total_steps
            )
            
            logger.info(f"Total training steps: {total_steps}")
            logger.info(f"Warmup steps: {args.warmup_frac * total_steps}")
            
            # Training loop
            logger.info("Starting training...")
            best_accuracy = 0.0
            best_f1 = 0.0
            train_losses = []
            train_accuracies = []
            
            for epoch in range(1, args.num_epochs + 1):
                train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, scheduler, device, epoch, logger)
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
            
            # Final evaluation
            logger.info("Training completed!")
            
            logger.info("Final evaluation on test set:")
            eval_result = evaluate_model(model, test_dataloader, device, logger, idx_to_label, "Final Test")
            eval_result['hparam'] = hparam
            eval_result['train_log'] = {
                'losses': train_losses,
                'accuracies': train_accuracies
            }
            
            # store results
            eval_results.append(eval_result)
            logging.info('*****************************************\n\n')
    
    # store evaluation data
    logging.info('Storing all results ....')
    utils.pickle_dump(eval_results, save_path)
    logging.info('Done!')