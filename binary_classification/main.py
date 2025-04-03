import os
import time
import json
import logging
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Import from your modules
# In main.py, update the imports
from preprocessing import load_fakenewsnet_csv, enhanced_clean_text, extract_domain, extract_features, format_news_for_llm
from prompts import improved_zero_shot_template, improved_few_shot_template
from classification import test_approaches
from evaluation import calculate_metrics
from visualizations import visualize_approach_results, plot_class_distribution, plot_confusion_matrix

# Add this line after imports
DEFAULT_DATA_PATH = os.path.join('data', 'raw')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('fake_news_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



def main(dataset_path=DEFAULT_DATA_PATH):
    # Your code here
    """Main execution function for fake news detection with improved prompts"""
    try:
        # Start timing
        start_time = datetime.now()
        logger.info(f"Starting fake news detection analysis with improved prompts at {start_time}")
        print(f"Starting fake news detection analysis with improved prompts at {start_time}")
        
        # Load dataset
        dataset_path = "/Users/macintosh/Desktop/MSc BA/Semester 2/LLM/Fake News/data/fakenewsnet"  # Update with your dataset path
        
        # Validate dataset path
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path does not exist: {dataset_path}")
            print(f"Dataset path does not exist: {dataset_path}")
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Load and preprocess data
        logger.info("Loading dataset...")
        print("Loading dataset...")
        df = load_fakenewsnet_csv(dataset_path)
        
        if len(df) == 0:
            logger.error("No data could be loaded from the dataset")
            print("No data could be loaded from the dataset")
            return
        
        print(f"Total dataset size: {len(df)} samples")
        print(f"Class distribution: {df['label'].value_counts().to_dict()}")
        
        # Preprocess data with error handling
        logger.info("Preprocessing data...")
        print("Preprocessing data...")
        df['cleaned_title'] = df['title'].apply(lambda x: enhanced_clean_text(str(x)) if pd.notna(x) else '')
        df['domain'] = df['news_url'].apply(extract_domain)
        
        # Extract features
        logger.info("Extracting features...")
        print("Extracting features...")
        feature_dfs = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            features = extract_features(row)
            features_df = pd.DataFrame([features], index=[idx])
            feature_dfs.append(features_df)
        
        features_df = pd.concat(feature_dfs)
        df = pd.concat([df, features_df], axis=1)
        
        # Create balanced dataset with 1000 samples per class
        logger.info("Creating balanced dataset with 1000 samples per class...")
        print("Creating balanced dataset with 1000 samples per class...")
        samples_per_class = 1000
        
        # Check if there are enough samples
        min_class_samples = df['label'].value_counts().min()
        if min_class_samples < samples_per_class:
            print(f"Warning: Requested {samples_per_class} samples per class, but the minority class only has {min_class_samples} samples.")
            print(f"Using {min_class_samples} samples per class instead.")
            samples_per_class = min_class_samples
        
        fake_sample = df[df['label'] == 'fake'].sample(n=samples_per_class, random_state=42)
        real_sample = df[df['label'] == 'real'].sample(n=samples_per_class, random_state=42)
        balanced_df = pd.concat([fake_sample, real_sample]).reset_index(drop=True)
        
        print(f"Balanced dataset size: {len(balanced_df)} samples")
        print(f"Balanced class distribution: {balanced_df['label'].value_counts().to_dict()}")
        
        # Split into train/test sets with 80-20 ratio
        logger.info("Splitting balanced dataset into 80% train and 20% test...")
        print("Splitting balanced dataset into 80% train and 20% test...")
        train_df, test_df = train_test_split(balanced_df, test_size=0.20, random_state=42, stratify=balanced_df['label'])
        
        print(f"Training set: {len(train_df)} samples ({len(train_df[train_df['label'] == 'fake'])} fake, {len(train_df[train_df['label'] == 'real'])} real)")
        print(f"Test set: {len(test_df)} samples ({len(test_df[test_df['label'] == 'fake'])} fake, {len(test_df[test_df['label'] == 'real'])} real)")
        
        # Save the templates for reference
        logger.info("Saving improved prompt templates...")
        print("Saving improved prompt templates...")
        with open('results/improved_prompt_templates.txt', 'w') as f:
            f.write("IMPROVED ZERO-SHOT TEMPLATE:\n\n")
            f.write(improved_zero_shot_template.replace("{news_data}", "[news_data]"))
            f.write("\n\n\n")
            f.write("IMPROVED FEW-SHOT TEMPLATE:\n\n")
            f.write(improved_few_shot_template.replace("{news_data}", "[news_data]"))
        
        # Test improved approaches
        logger.info("Testing improved approaches...")
        print("Testing improved approaches...")
        
        approaches_to_test = {
            "models": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            "prompt_types": ["zero_shot", "few_shot"]
        }
        
        approach_results = test_approaches(
            test_df,
            models=approaches_to_test["models"],
            prompt_types=approaches_to_test["prompt_types"]
        )
        
        # Visualize and analyze results
        logger.info("Generating visualizations and metrics...")
        print("Generating visualizations and metrics...")
        metrics_data = visualize_approach_results(approach_results)
        
        # Log completion time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() / 60
        logger.info(f"Fake news detection analysis with improved prompts completed successfully in {execution_time:.2f} minutes")
        print(f"Fake news detection analysis with improved prompts completed successfully in {execution_time:.2f} minutes")
        
        # Generate final report
        logger.info("Generating final report...")
        print("Generating final report...")
        with open('results/improved_prompts_final_report.txt', 'w') as f:
            f.write(f"Fake News Detection Analysis Report with Improved Prompts\n")
            f.write(f"Generated on: {end_time}\n")
            f.write(f"Execution time: {execution_time:.2f} minutes\n\n")
            
            f.write("Dataset Information:\n")
            f.write(f"  - Total samples: {len(df)}\n")
            f.write(f"  - Balanced samples: {len(balanced_df)} (1000 per class)\n")
            f.write(f"  - Training samples: {len(train_df)} (80%)\n")
            f.write(f"  - Test samples: {len(test_df)} (20%)\n\n")
            
            f.write("Performance Metrics:\n")
            metrics_table = pd.DataFrame(columns=[
                'Approach', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 
                'Tokens per Article', 'Cost per Classification ($)', 'Processing time (s)'
            ])
            
            for i, (approach, metrics) in enumerate(metrics_data.items()):
                metrics_table.loc[i] = [
                    approach,
                    round(metrics.get('accuracy', 0), 2),
                    round(metrics.get('precision', 0), 2),
                    round(metrics.get('recall', 0), 2),
                    round(metrics.get('f1', 0), 2),
                    round(metrics.get('avg_tokens', 0), 2),
                    round(metrics.get('avg_cost', 0), 6),
                    round(metrics.get('avg_processing_time', 0), 1)
                ]
            
            f.write(metrics_table.to_string(index=False))
        
        logger.info("Analysis complete. Results saved to 'results/' directory.")
        print("Analysis complete. Results saved to 'results/' directory.")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        print(f"Critical error in main execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(traceback.format_exc())
        
def ensure_directory_structure():
    """Ensure all necessary directories exist"""
    os.makedirs('data/raw', exist_ok=True)        
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fake News Detection with LLMs')
    parser.add_argument('--dataset_path', default=DEFAULT_DATA_PATH,
                      help='Path to the FakeNewsNet dataset (default: data/raw)')
    args = parser.parse_args()
    
    # Add this function call before main
    ensure_directory_structure()
    main(args.dataset_path)