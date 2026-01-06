import pandas as pd
import numpy as np
import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class DLSentimentTrainer:
    def __init__(self, config_path='src/config/dl_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.config['pretrained_model']
        )
        
        self.model = None
        self.training_time = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def load_data(self, train_path, val_path):
        """Load and prepare datasets"""
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        train_dataset = SentimentDataset(
            train_df['text_cleaned'].values,
            train_df['label'].values,
            self.tokenizer,
            self.config['model_params']['max_length']
        )
        
        val_dataset = SentimentDataset(
            val_df['text_cleaned'].values,
            val_df['label'].values,
            self.tokenizer,
            self.config['model_params']['max_length']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def create_model(self):
        """Initialize the model"""
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config['pretrained_model'],
            num_labels=self.config['model_params']['num_labels']
        )
        self.model.to(self.device)
        return self.model
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )
            
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader):
        """Main training loop with early stopping"""
        
        self.create_model()
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        total_steps = len(train_loader) * self.config['training']['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=total_steps
        )
        
        print("\nStarting training...")
        start_time = time.time()
        
        for epoch in range(self.config['training']['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")
            
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            val_loss, val_acc = self.validate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_loss < self.best_val_loss - self.config['early_stopping']['min_delta']:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model(is_best=True)
                print("Saved best model!")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config['early_stopping']['patience']:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break
        
        self.training_time = time.time() - start_time
        print(f"\nTotal training time: {self.training_time:.2f} seconds")
        
        self.load_best_model()
        
    def save_model(self, is_best=False):
        """Save model checkpoint"""
        model_dir = Path(self.config['paths']['model_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            save_path = Path(self.config['paths']['best_model_path'])
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
    
    def load_best_model(self):
        """Load the best saved model"""
        best_path = Path(self.config['paths']['best_model_path'])
        self.model = DistilBertForSequenceClassification.from_pretrained(best_path)
        self.model.to(self.device)
    
    def evaluate(self, test_path):
        """Evaluate on test set"""
        test_df = pd.read_csv(test_path)
        
        test_dataset = SentimentDataset(
            test_df['text_cleaned'].values,
            test_df['label'].values,
            self.tokenizer,
            self.config['model_params']['max_length']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )
        
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        print("\n=== DL Model Evaluation ===")
        print(classification_report(true_labels, predictions,
                                   target_names=['Negative', 'Neutral', 'Positive']))
        
        return {
            'predictions': predictions,
            'true_labels': true_labels,
            'accuracy': accuracy_score(true_labels, predictions)
        }

def train_dl_model():
    """Main training pipeline for DL model"""
    
    trainer = DLSentimentTrainer()
    
    train_loader, val_loader = trainer.load_data(
        'data/processed/train.csv',
        'data/processed/val.csv'
    )
    
    trainer.train(train_loader, val_loader)
    
    results = trainer.evaluate('data/processed/test.csv')
    
    return trainer, results

if __name__ == '__main__':
    trainer, results = train_dl_model()