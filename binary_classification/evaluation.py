import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Create results directory
os.makedirs('results', exist_ok=True)

def calculate_metrics(results_df):
    """Calculate classification and efficiency metrics"""
    # Filter out rows with missing predictions
    valid_df = results_df.dropna(subset=['predicted_label'])
    
    if len(valid_df) == 0 or 'actual_label' not in valid_df.columns:
        return {}
    
    # Calculate classification metrics
    y_true = valid_df['actual_label']
    y_pred = valid_df['predicted_label']
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label='fake', zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label='fake', zero_division=0),
        'f1': f1_score(y_true, y_pred, pos_label='fake', zero_division=0),
        'samples_evaluated': len(valid_df),
    }
    
    # Add token and time metrics
    metrics['avg_tokens'] = valid_df['tokens_used'].mean() if 'tokens_used' in valid_df.columns else 0
    metrics['total_tokens'] = valid_df['tokens_used'].sum() if 'tokens_used' in valid_df.columns else 0
    metrics['avg_processing_time'] = valid_df['processing_time'].mean() if 'processing_time' in valid_df.columns else 0
    
    # Calculate cost metrics
    cost_per_token = {
        'gpt-4o': 0.000003,  # $0.003 per 1K tokens
        'gpt-4-turbo': 0.00001,  # $0.01 per 1K tokens
        'gpt-3.5-turbo': 0.0000002  # $0.0002 per 1K tokens
    }
    
    if 'model' in valid_df.columns and 'tokens_used' in valid_df.columns:
        valid_df['cost'] = valid_df.apply(
            lambda row: row['tokens_used'] * cost_per_token.get(row['model'], 0.000003),
            axis=1
        )
        metrics['avg_cost'] = valid_df['cost'].mean()
        metrics['total_cost'] = valid_df['cost'].sum()
    
    return metrics