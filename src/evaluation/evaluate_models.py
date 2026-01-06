import pandas as pd
import numpy as np
import json
import time
import pickle
import torch
from pathlib import Path
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self):
        self.ml_model = None
        self.ml_vectorizer = None
        self.dl_model = None
        self.dl_tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_ml_model(self):
        """Load trained ML model"""
        with open('saved_models/ml/tfidf_vectorizer.pkl', 'rb') as f:
            self.ml_vectorizer = pickle.load(f)
        
        with open('saved_models/ml/logistic_model.pkl', 'rb') as f:
            self.ml_model = pickle.load(f)
    
    def load_dl_model(self):
        """Load trained DL model"""
        model_path = 'saved_models/dl/best_model'
        self.dl_model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.dl_tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.dl_model.to(self.device)
        self.dl_model.eval()
    
    def predict_ml(self, texts):
        """Predict using ML model"""
        X = self.ml_vectorizer.transform(texts)
        predictions = self.ml_model.predict(X)
        return predictions
    
    def predict_dl(self, texts):
        """Predict using DL model"""
        predictions = []
        
        for text in texts:
            encoding = self.dl_tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.dl_model(input_ids=input_ids, attention_mask=attention_mask)
                pred = torch.argmax(outputs.logits, dim=1)
                predictions.append(pred.item())
        
        return np.array(predictions)
    
    def measure_inference_speed(self, texts, n_runs=100):
        """Measure average inference time"""
        
        sample_texts = texts[:n_runs] if len(texts) > n_runs else texts
        
        start = time.time()
        _ = self.predict_ml(sample_texts)
        ml_time = (time.time() - start) / len(sample_texts) * 1000
        
        start = time.time()
        _ = self.predict_dl(sample_texts)
        dl_time = (time.time() - start) / len(sample_texts) * 1000
        
        return ml_time, dl_time
    
    def get_model_size(self):
        """Get model file sizes"""
        ml_size = (
            Path('saved_models/ml/logistic_model.pkl').stat().st_size +
            Path('saved_models/ml/tfidf_vectorizer.pkl').stat().st_size
        ) / (1024 * 1024)
        
        dl_size = sum(
            f.stat().st_size for f in Path('saved_models/dl/best_model').rglob('*') 
            if f.is_file()
        ) / (1024 * 1024)
        
        return ml_size, dl_size
    
    def evaluate_both_models(self, test_df):
        """Comprehensive evaluation of both models"""
        
        self.load_ml_model()
        self.load_dl_model()
        
        texts = test_df['text_cleaned'].values
        true_labels = test_df['label'].values
        
        print("Predicting with ML model...")
        ml_predictions = self.predict_ml(texts)
        
        print("Predicting with DL model...")
        dl_predictions = self.predict_dl(texts)
        
        ml_metrics = self._compute_metrics(true_labels, ml_predictions, "ML Model")
        dl_metrics = self._compute_metrics(true_labels, dl_predictions, "DL Model")
        
        print("\nMeasuring inference speed...")
        ml_speed, dl_speed = self.measure_inference_speed(texts)
        
        ml_size, dl_size = self.get_model_size()
        
        comparison = {
            'ml_model': {
                **ml_metrics,
                'inference_time_ms': ml_speed,
                'model_size_mb': ml_size
            },
            'dl_model': {
                **dl_metrics,
                'inference_time_ms': dl_speed,
                'model_size_mb': dl_size
            }
        }
        
        with open('data/processed/model_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison, ml_predictions, dl_predictions
    
    def _compute_metrics(self, y_true, y_pred, model_name):
        """Compute detailed metrics"""
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1, 2]
        )
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_per_class': {
                'negative': float(precision[0]),
                'neutral': float(precision[1]),
                'positive': float(precision[2])
            },
            'recall_per_class': {
                'negative': float(recall[0]),
                'neutral': float(recall[1]),
                'positive': float(recall[2])
            },
            'f1_per_class': {
                'negative': float(f1[0]),
                'neutral': float(f1[1]),
                'positive': float(f1[2])
            },
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        print(f"\n=== {model_name} Metrics ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Scores: Neg={f1[0]:.3f}, Neu={f1[1]:.3f}, Pos={f1[2]:.3f}")
        
        return metrics
    
    def plot_confusion_matrices(self, ml_cm, dl_cm):
        """Plot confusion matrices side by side"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        labels = ['Negative', 'Neutral', 'Positive']
        
        sns.heatmap(ml_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title('ML Model Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        sns.heatmap(dl_cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=labels, yticklabels=labels, ax=axes[1])
        axes[1].set_title('DL Model Confusion Matrix')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('data/processed/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

def evaluate_models():
    """Main evaluation pipeline"""
    
    evaluator = ModelEvaluator()
    
    test_df = pd.read_csv('data/processed/test.csv')
    
    comparison, ml_preds, dl_preds = evaluator.evaluate_both_models(test_df)
    
    ml_cm = np.array(comparison['ml_model']['confusion_matrix'])
    dl_cm = np.array(comparison['dl_model']['confusion_matrix'])
    evaluator.plot_confusion_matrices(ml_cm, dl_cm)
    
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"\nML Model:")
    print(f"  Accuracy: {comparison['ml_model']['accuracy']:.4f}")
    print(f"  Speed: {comparison['ml_model']['inference_time_ms']:.2f} ms/sample")
    print(f"  Size: {comparison['ml_model']['model_size_mb']:.2f} MB")
    
    print(f"\nDL Model:")
    print(f"  Accuracy: {comparison['dl_model']['accuracy']:.4f}")
    print(f"  Speed: {comparison['dl_model']['inference_time_ms']:.2f} ms/sample")
    print(f"  Size: {comparison['dl_model']['model_size_mb']:.2f} MB")
    
    return comparison

if __name__ == '__main__':
    comparison = evaluate_models()