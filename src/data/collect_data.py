import pandas as pd
import numpy as np
from datasets import load_dataset
import json
import os
from pathlib import Path

class DatasetCollector:
    def __init__(self, output_dir='data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_twitter_sentiment(self, sample_size=500):

        print("Loading Twitter sentiment dataset...")
        dataset = load_dataset("sentiment140", split="train", trust_remote_code=True)
        
        df = pd.DataFrame(dataset)
        df = df.rename(columns={'sentiment': 'label', 'text': 'text'})
        df['label'] = df['label'].map({0: 0, 4: 2})
        
        # Filter out very short tweets
        df = df[df['text'].str.len() > 20]
        
        sampled = self._stratified_sample(df, sample_size)
        sampled['source'] = 'twitter'
        
        print(f"Collected {len(sampled)} Twitter samples")
        return sampled
    
    def collect_imdb_reviews(self, sample_size=500):
        """
        Collect IMDB movie reviews for sentiment analysis.
        Mix of positive/negative reviews from IMDB dataset.
        Collecting exactly 500 samples.
        """
        print("Loading IMDB reviews dataset...")
        dataset = load_dataset("imdb", split="train")
        
        df = pd.DataFrame(dataset)
        df = df.rename(columns={'label': 'label', 'text': 'text'})
        
        # Filter out very short reviews
        df = df[df['text'].str.len() > 30]
        
        sampled = self._stratified_sample(df, sample_size)
        sampled['source'] = 'imdb'
        
        print(f"Collected {len(sampled)} IMDB samples")
        return sampled
    
    def collect_neutral_samples(self, sample_size=200):
        """
        Collect actual neutral sentiment samples.
        Using SST (Stanford Sentiment Treebank) which has neutral labels.
        """
        print("Loading neutral sentiment samples...")
        try:
            # Try to load SST dataset which has neutral samples
            from datasets import load_dataset
            dataset = load_dataset("sst", split="train")
            df = pd.DataFrame(dataset)
            
            # SST has labels 0-4, where 2 is neutral
            neutral_df = df[df['label'] == 2].copy()
            neutral_df['label'] = 1  # Our neutral label
            neutral_df = neutral_df.rename(columns={'sentence': 'text'})
            neutral_df['source'] = 'sst_neutral'
            
            if len(neutral_df) >= sample_size:
                sampled = neutral_df.sample(n=sample_size, random_state=42)
            else:
                sampled = neutral_df
            
            print(f"Collected {len(sampled)} actual neutral samples")
            return sampled
        except:
            print("SST dataset not available, using synthetic neutral samples")
            return self._create_synthetic_neutral(sample_size)
    
    def _create_synthetic_neutral(self, sample_size):
        """Fallback: create synthetic neutral samples"""
        # This is the old method as backup
        neutral_texts = [
            "The product works as described.",
            "It's an average item.",
            "Nothing special but acceptable.",
            "Does what it's supposed to do.",
            "Standard quality product.",
            "It's okay for the price.",
            "Average experience overall.",
            "Meets basic expectations.",
            "Not bad, not great either.",
            "Typical product in this category."
        ] * (sample_size // 10 + 1)
        
        neutral_df = pd.DataFrame({
            'text': neutral_texts[:sample_size],
            'label': [1] * sample_size,
            'source': ['synthetic'] * sample_size
        })
        return neutral_df
    
    def add_neutral_samples(self, df, neutral_ratio=0.20):
        """
        Add neutral samples using better method.
        INCREASED neutral ratio from 15% to 20% for better balance.
        """
        n_neutral = int(len(df) * neutral_ratio)
        return self.collect_neutral_samples(n_neutral)
    
    def _stratified_sample(self, df, sample_size):
        """Stratified sampling to maintain class balance"""
        samples_per_class = sample_size // df['label'].nunique()
        
        sampled_dfs = []
        for label in df['label'].unique():
            class_df = df[df['label'] == label]
            if len(class_df) > samples_per_class:
                sampled_dfs.append(class_df.sample(n=samples_per_class, random_state=42))
            else:
                sampled_dfs.append(class_df)
        
        return pd.concat(sampled_dfs, ignore_index=True)
    
    def combine_and_save(self):
        """Main method to collect and combine datasets"""
        
        # EXACTLY 500 samples from each source
        twitter_df = self.collect_twitter_sentiment(sample_size=500)
        imdb_df = self.collect_imdb_reviews(sample_size=500)
        
        combined_df = pd.concat([twitter_df, imdb_df], ignore_index=True)
        
        # Use better neutral collection (20% instead of 15%)
        neutral_df = self.add_neutral_samples(combined_df, neutral_ratio=0.20)
        final_df = pd.concat([combined_df, neutral_df], ignore_index=True)
        
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        output_path = self.output_dir / 'raw_sentiment_data.csv'
        final_df.to_csv(output_path, index=False)
        
        print(f"\n=== Dataset Collection Summary ===")
        print(f"Total samples: {len(final_df)}")
        print(f"\nClass distribution:")
        print(final_df['label'].value_counts().sort_index())
        print(f"\nSource distribution:")
        print(final_df['source'].value_counts())
        
        stats = {
            'total_samples': len(final_df),
            'class_distribution': final_df['label'].value_counts().to_dict(),
            'source_distribution': final_df['source'].value_counts().to_dict(),
            'avg_text_length': final_df['text'].str.len().mean(),
            'datasets_used': ['sentiment140 (Twitter) - 500 samples', 'IMDB Reviews - 500 samples', 'SST (Neutral)']
        }
        
        with open(self.output_dir / 'collection_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nData saved to: {output_path}")
        return final_df

if __name__ == '__main__':
    collector = DatasetCollector()
    df = collector.combine_and_save()