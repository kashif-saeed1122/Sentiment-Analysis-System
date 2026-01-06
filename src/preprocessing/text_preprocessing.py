import re
import pandas as pd
import numpy as np
from pathlib import Path
import emoji
import json

class TextPreprocessor:
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.special_chars = re.compile(r'[^a-zA-Z0-9\s.,!?]')
        
    def remove_urls(self, text):
        return self.url_pattern.sub('', text)
    
    def remove_mentions(self, text):
        return self.mention_pattern.sub('', text)
    
    def handle_hashtags(self, text):
        return self.hashtag_pattern.sub(lambda m: m.group(0)[1:], text)
    
    def convert_emojis(self, text):
        return emoji.demojize(text, delimiters=(" ", " "))
    
    def remove_extra_spaces(self, text):
        return ' '.join(text.split())
    
    def clean_special_chars(self, text):
        text = self.special_chars.sub(' ', text)
        return text
    
    def basic_clean(self, text):
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        has_caps = text.isupper()
        
        text = self.convert_emojis(text)
        
        url_count = len(self.url_pattern.findall(text))
        text = self.remove_urls(text)
        if url_count > 0:
            text += " haslink"
        
        text = self.handle_hashtags(text)
        
        text = self.remove_mentions(text)
        
        text = self.handle_negations(text)
        
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\']', ' ', text)
        
        if '!!!' in text or '???' in text:
            text += " emphasis"
        
        if re.search(r'(.)\1{2,}', text):
            text += " repeated"
        
        text = self.remove_extra_spaces(text)
        
        return text.strip()
    
    def handle_negations(self, text):
        """Handle negations by marking negated words"""
        negations = ["not", "no", "never", "n't", "cannot", "nowhere", "nothing"]
        words = text.split()
        
        for i, word in enumerate(words):
            if word in negations and i + 1 < len(words):
                # Mark next word as negated
                words[i + 1] = "NOT_" + words[i + 1]
        
        return " ".join(words)
    
    def process_dataset(self, df):
        """Clean and preprocess the entire dataset"""
        
        print("Starting text preprocessing...")
        
        initial_count = len(df)
        df = df.copy()
        
        df['text_cleaned'] = df['text'].apply(self.basic_clean)
        
        df = df[df['text_cleaned'].str.len() > 10]
        
        df = df.drop_duplicates(subset=['text_cleaned'])
        
        print(f"Removed {initial_count - len(df)} samples during cleaning")
        print(f"Final dataset size: {len(df)}")
        
        return df
    
    def split_dataset(self, df, train_ratio=0.7, val_ratio=0.15):
        """Split dataset into train, validation, and test sets"""
        
        np.random.seed(42)
        df = df.sample(frac=1).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]
        
        print(f"\nDataset split:")
        print(f"Train: {len(train_df)} samples")
        print(f"Validation: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df, val_df, test_df, output_dir='data/processed'):
        """Save processed datasets"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(output_path / 'train.csv', index=False)
        val_df.to_csv(output_path / 'val.csv', index=False)
        test_df.to_csv(output_path / 'test.csv', index=False)
        
        stats = {
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'train_class_dist': train_df['label'].value_counts().to_dict(),
            'avg_length_cleaned': train_df['text_cleaned'].str.len().mean()
        }
        
        with open(output_path / 'preprocessing_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nProcessed data saved to: {output_path}")

def preprocess_pipeline(raw_data_path='data/raw/raw_sentiment_data.csv'):
    """Main preprocessing pipeline"""
    
    print("Loading raw data...")
    df = pd.read_csv(raw_data_path)
    
    preprocessor = TextPreprocessor()
    
    df_cleaned = preprocessor.process_dataset(df)
    
    train_df, val_df, test_df = preprocessor.split_dataset(df_cleaned)
    
    preprocessor.save_processed_data(train_df, val_df, test_df)
    
    return train_df, val_df, test_df

if __name__ == '__main__':
    preprocess_pipeline()