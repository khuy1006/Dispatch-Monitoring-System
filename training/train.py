#!/usr/bin/env python3
"""
Main training script for YOLO classification model.

This script handles the complete pipeline:
1. Data splitting into train/val/test sets
2. Data augmentation for training set
3. Model training with YOLO

Usage:
    python train.py [--config config.yaml] [--no-split] [--no-augment] [--no-train]
"""

import sys
import logging
import argparse
from pathlib import Path
import yaml

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils import DataSplitter, DataAugmenter, YOLOTrainer


def setup_logging(config: dict) -> None:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO').upper())
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Save logs to file if specified
    if log_config.get('save_logs', False):
        log_file = Path(log_config.get('log_file', 'training.log'))
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"✅ Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        logging.error(f"❌ Failed to load config from {config_path}: {e}")
        sys.exit(1)


def validate_config(config: dict) -> bool:
    """Validate configuration parameters."""
    required_sections = ['data', 'split', 'training']
    
    for section in required_sections:
        if section not in config:
            logging.error(f"Missing required config section: {section}")
            return False
    
    # Validate data paths
    source_dir = Path(config['data']['source_dir'])
    if not source_dir.exists():
        logging.error(f"Source directory does not exist: {source_dir}")
        return False
    
    # Validate split ratios
    split_config = config['split']
    total_ratio = split_config['train_ratio'] + split_config['val_ratio'] + split_config['test_ratio']
    if abs(total_ratio - 1.0) > 1e-6:
        logging.error(f"Split ratios must sum to 1.0, got {total_ratio}")
        return False
    
    return True


def split_data(config: dict, force: bool = False) -> bool:
    """Split dataset into train/val/test sets."""
    data_config = config['data']
    split_config = config['split']
    
    output_dir = Path(data_config['output_dir'])
    
    # Check if split already exists
    if output_dir.exists() and not force:
        logging.info(f"Split data already exists at {output_dir}. Skipping split.")
        return True
    
    # Initialize splitter
    splitter = DataSplitter(
        source_dir=data_config['source_dir'],
        dest_dir=str(output_dir),
        classes=data_config['classes']
    )
    
    # Perform split
    return splitter.split_dataset(
        train_ratio=split_config['train_ratio'],
        val_ratio=split_config['val_ratio'],
        test_ratio=split_config['test_ratio']
    )


def augment_data(config: dict) -> bool:
    """Apply data augmentation to training set."""
    aug_config = config.get('augmentation', {})
    
    if not aug_config.get('enabled', True):
        logging.info("Data augmentation disabled in config. Skipping.")
        return True
    
    data_config = config['data']
    train_dir = Path(data_config['output_dir']) / 'train'
    
    # Initialize augmenter
    augmenter = DataAugmenter(
        train_dir=str(train_dir),
        classes=data_config['classes']
    )
    
    # Apply augmentation
    return augmenter.augment_dataset(aug_config['types'])


def train_model(config: dict) -> bool:
    """Train YOLO classification model."""
    training_config = config['training']
    data_config = config['data']
    
    # Initialize trainer
    trainer = YOLOTrainer(model_name=training_config['model_name'])
    
    # Train model
    results = trainer.train(
        data_path=data_config['output_dir'],
        epochs=training_config['epochs'],
        patience=training_config['patience'],
        device=training_config['device'],
        batch_size=training_config['batch_size']
    )
    
    if results is None:
        return False
    
    # Save model info
    model_info = trainer.get_model_info()
    logging.info(f"Model info: {model_info}")
    
    return True


def main():
    """Main function to orchestrate the training pipeline."""
    parser = argparse.ArgumentParser(description='YOLO Classification Training Pipeline')
    parser.add_argument('--config', '-c', default='config.yaml', 
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--no-split', action='store_true',
                       help='Skip data splitting step')
    parser.add_argument('--no-augment', action='store_true', 
                       help='Skip data augmentation step')
    parser.add_argument('--no-train', action='store_true',
                       help='Skip model training step')
    parser.add_argument('--force-split', action='store_true',
                       help='Force data splitting even if output exists')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    
    # Validate configuration
    if not validate_config(config):
        logging.error("❌ Configuration validation failed")
        sys.exit(1)
    
    logging.info("🚀 Starting YOLO Classification Training Pipeline")
    logging.info("=" * 60)
    
    # Step 1: Data Splitting
    if not args.no_split:
        logging.info("📂 Step 1: Data Splitting")
        if not split_data(config, force=args.force_split):
            logging.error("❌ Data splitting failed")
            sys.exit(1)
        logging.info("✅ Data splitting completed\n")
    else:
        logging.info("📂 Step 1: Data Splitting (SKIPPED)\n")
    
    # Step 2: Data Augmentation  
    if not args.no_augment:
        logging.info("🔄 Step 2: Data Augmentation")
        if not augment_data(config):
            logging.error("❌ Data augmentation failed")
            sys.exit(1)
        logging.info("✅ Data augmentation completed\n")
    else:
        logging.info("🔄 Step 2: Data Augmentation (SKIPPED)\n")
    
    # Step 3: Model Training
    if not args.no_train:
        logging.info("🤖 Step 3: Model Training")
        if not train_model(config):
            logging.error("❌ Model training failed")
            sys.exit(1)
        logging.info("✅ Model training completed\n")
    else:
        logging.info("🤖 Step 3: Model Training (SKIPPED)\n")
    
    logging.info("=" * 60)
    logging.info("🎉 Training pipeline completed successfully!")


if __name__ == "__main__":
    main() 