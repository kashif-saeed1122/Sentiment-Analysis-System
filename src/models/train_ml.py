import pandas as pd
import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class MLSentimentTrainer:
    def __init__(self, config_path='src/config/ml_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.vectorizer = None
        self.model = None
        self.training_time = 0
        
    def load_data(self, train_path, val_path):
        """Load training and validation data"""
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        X_train = train_df['text_cleaned'].values
        y_train = train_df['label'].values
        X_val = val_df['text_cleaned'].values
        y_val = val_df['label'].values
        
        return X_train, y_train, X_val, y_val
    
    def create_vectorizer(self):
        """Create TF-IDF vectorizer with config parameters"""
        vec_config = self.config['vectorizer']
        
        self.vectorizer = TfidfVectorizer(
            max_features=vec_config['max_features'],
            ngram_range=tuple(vec_config['ngram_range']),
            min_df=vec_config['min_df'],
            max_df=vec_config['max_df'],
            sublinear_tf=True
        )
        
        return self.vectorizer
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train ML model with hyperparameter tuning"""
        
        print("Vectorizing text data...")
        self.create_vectorizer()
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_val_vec = self.vectorizer.transform(X_val)
        
        print(f"Feature matrix shape: {X_train_vec.shape}")
        
        print("\nStarting hyperparameter tuning...")
        clf_config = self.config['classifier']
        grid_config = self.config['grid_search']
        
        base_model = LogisticRegression(random_state=42)
        
        grid_search = GridSearchCV(
            base_model,
            param_grid=clf_config,
            cv=grid_config['cv'],
            scoring=grid_config['scoring'],
            n_jobs=grid_config['n_jobs'],
            verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train_vec, y_train)
        self.training_time = time.time() - start_time
        
        self.model = grid_search.best_estimator_
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Training time: {self.training_time:.2f} seconds")
        
        val_pred = self.model.predict(X_val_vec)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"Validation accuracy: {val_acc:.4f}")
        
        return self.model
    
    def save_model(self):
        """Save trained model and vectorizer"""
        model_dir = Path(self.config['paths']['model_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config['paths']['vectorizer_path'], 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(self.config['paths']['model_path'], 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"\nModel saved to: {self.config['paths']['model_dir']}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vec)
        
        print("\n=== ML Model Evaluation ===")
        print(classification_report(y_test, y_pred, 
                                target_names=['Negative', 'Neutral', 'Positive']))
        
        return {
            'predictions': y_pred,
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

def train_ml_model():
    """Main training pipeline for ML model"""
    
    trainer = MLSentimentTrainer()
    
    X_train, y_train, X_val, y_val = trainer.load_data(
        'data/processed/train.csv',
        'data/processed/val.csv'
    )
    
    trainer.train(X_train, y_train, X_val, y_val)
    
    trainer.save_model()
    
    test_df = pd.read_csv('data/processed/test.csv')
    results = trainer.evaluate(test_df['text_cleaned'].values, 
                              test_df['label'].values)
    
    return trainer, results

if __name__ == '__main__':
    trainer, results = train_ml_model()