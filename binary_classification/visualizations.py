import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Create figures directory
os.makedirs('figures', exist_ok=True)

def visualize_approach_results(approach_results):
    """Visualize the comparison of different approaches"""
    plt.figure(figsize=(15, 10))
    
    # Collect metrics
    metrics_data = {approach: metrics['metrics'] 
                    for approach, metrics in approach_results.items() 
                    if 'metrics' in metrics}
    
    # Create a DataFrame for easier plotting
    df_metrics = pd.DataFrame({
        'approach': [],
        'metric': [],
        'value': []
    })
    
    for approach, metrics in metrics_data.items():
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            df_metrics = pd.concat([df_metrics, pd.DataFrame({
                'approach': [approach],
                'metric': [metric],
                'value': [metrics.get(metric, 0)]
            })], ignore_index=True)
    
    # Plot classification metrics
    plt.subplot(2, 2, 1)
    sns.barplot(x='approach', y='value', hue='metric', data=df_metrics)
    plt.title('Classification Performance')
    plt.xlabel('Approach')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric')
    plt.tight_layout()
    
    # Plot token usage
    plt.subplot(2, 2, 2)
    token_data = [metrics.get('avg_tokens', 0) for metrics in metrics_data.values()]
    plt.bar(metrics_data.keys(), token_data)
    plt.title('Average Tokens per Article')
    plt.xlabel('Approach')
    plt.ylabel('Tokens')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Plot cost
    plt.subplot(2, 2, 3)
    cost_data = [metrics.get('avg_cost', 0) for metrics in metrics_data.values()]
    plt.bar(metrics_data.keys(), cost_data)
    plt.title('Average Cost per Classification ($)')
    plt.xlabel('Approach')
    plt.ylabel('Cost ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Plot processing time
    plt.subplot(2, 2, 4)
    time_data = [metrics.get('avg_processing_time', 0) for metrics in metrics_data.values()]
    plt.bar(metrics_data.keys(), time_data)
    plt.title('Average Processing Time per Article (s)')
    plt.xlabel('Approach')
    plt.ylabel('Time (s)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('figures/approach_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics_data

def plot_class_distribution(df, column='label', title='Class Distribution'):
    """Plot the distribution of classes in the dataset"""
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.savefig('figures/class_distribution.png', dpi=300)
    plt.close()
    
def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Plot confusion matrix for classification results"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('figures/confusion_matrix.png', dpi=300)
    plt.close()