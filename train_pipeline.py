#!/usr/bin/env python3
"""
Main training pipeline for sentiment analysis system
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.collect_data import DatasetCollector
from src.preprocessing.text_preprocessing import preprocess_pipeline
from src.models.train_ml import train_ml_model
from src.models.train_dl import train_dl_model
from src.evaluation.evaluate_models import evaluate_models
from src.utils.logger import setup_logger

def main():
    logger = setup_logger('main_pipeline', 'training_pipeline.log')
    
    logger.info("="*60)
    logger.info("Starting Sentiment Analysis Training Pipeline")
    logger.info("="*60)
    
    try:
        logger.info("\n[1/5] Collecting datasets...")
        collector = DatasetCollector()
        raw_df = collector.combine_and_save()
        logger.info(f"✓ Collected {len(raw_df)} samples")
        
        logger.info("\n[2/5] Preprocessing data...")
        train_df, val_df, test_df = preprocess_pipeline()
        logger.info(f"✓ Created splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        logger.info("\n[3/5] Training ML model...")
        ml_trainer, ml_results = train_ml_model()
        logger.info(f"✓ ML model trained with {ml_results['accuracy']:.4f} accuracy")
        
        logger.info("\n[4/5] Training DL model...")
        dl_trainer, dl_results = train_dl_model()
        logger.info(f"✓ DL model trained with {dl_results['accuracy']:.4f} accuracy")
        
        logger.info("\n[5/5] Evaluating and comparing models...")
        comparison = evaluate_models()
        logger.info("✓ Evaluation complete")
        
        logger.info("\n" + "="*60)
        logger.info("Pipeline completed successfully!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Review notebooks/model_comparison.ipynb for detailed analysis")
        logger.info("2. Start API: cd src/api && python main.py")
        logger.info("3. Test API: curl -X POST http://localhost:8000/predict-ml -H 'Content-Type: application/json' -d '{\"text\":\"This is great!\"}'")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()