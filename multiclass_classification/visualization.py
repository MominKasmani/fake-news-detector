import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_confusion_matrix(true_labels, predicted_labels, model_name, prompt_type):
    """Create and save an improved confusion matrix visualization"""
    # Create confusion matrix
    confusion_mat = pd.crosstab(
        true_labels, 
        predicted_labels,
        rownames=["Actual"], 
        colnames=["Predicted"], 
        normalize="index"
    )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_mat, annot=True, fmt=".2f", cmap="Blues", 
                cbar_kws={'label': 'Proportion'})
    
    # Improve label readability
    plt.title(f"Confusion Matrix - {model_name} with {prompt_type} prompt", fontsize=16)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Rotate x-axis labels for better readability and adjust layout
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Save the figure with higher resolution
    plt.savefig(f"figures/confusion_matrix_{model_name}_{prompt_type}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison(metrics_df):
    """Create enhanced visual comparison of model performance"""
    # Use a more readable style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')  # Fallback for different seaborn versions
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), constrained_layout=True)
    
    # Color palette for consistency
    palette = sns.color_palette("viridis", len(metrics_df['Prompt Type'].unique()))
    
    # 1. Accuracy comparison
    sns.barplot(ax=axes[0], x="Model", y="Accuracy", hue="Prompt Type", 
                data=metrics_df, palette=palette)
    axes[0].set_title("Accuracy by Model and Prompt Type", fontsize=16)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy", fontsize=14)
    axes[0].set_xlabel("Model", fontsize=14)
    axes[0].tick_params(labelsize=12)
    axes[0].legend(title="Prompt Type", fontsize=12, title_fontsize=12)
    
    # 2. Token usage comparison
    token_data = pd.melt(metrics_df, 
                       id_vars=["Model", "Prompt Type"],
                       value_vars=["Avg Input Tokens", "Avg Output Tokens", "Avg Total Tokens"],
                       var_name="Token Type", value_name="Tokens")
    
    sns.barplot(ax=axes[1], x="Model", y="Tokens", hue="Token Type", 
                data=token_data, palette="Blues")
    axes[1].set_title("Token Usage by Model and Prompt Type", fontsize=16)
    axes[1].set_ylabel("Average Token Count", fontsize=14)
    axes[1].set_xlabel("Model", fontsize=14)
    axes[1].tick_params(labelsize=12)
    axes[1].legend(title="Token Type", fontsize=12, title_fontsize=12)
    
    # 3. Cost vs. Accuracy
    sns.scatterplot(ax=axes[2], x="Cost per Example ($)", y="Accuracy", 
                  hue="Model", style="Prompt Type", s=200, 
                  data=metrics_df, palette="viridis")
    
    # Add labels to points
    for idx, row in metrics_df.iterrows():
        axes[2].text(row['Cost per Example ($)'] + 0.00001, 
                 row['Accuracy'] + 0.005,
                 f"{row['Model']}-{row['Prompt Type']}", 
                 fontsize=8)
    
    axes[2].set_title("Cost vs. Accuracy Tradeoff", fontsize=16)
    axes[2].set_xlabel("Cost per Example ($)", fontsize=14)
    axes[2].set_ylabel("Accuracy", fontsize=14)
    axes[2].tick_params(labelsize=12)
    axes[2].legend(title="Model", fontsize=12, title_fontsize=12)
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Save figure
    plt.savefig("figures/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_cross_dataset_comparison(cross_metrics, model_name, prompt_type):
    """Create enhanced cross-dataset comparison visualization"""
    # Create a figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    
    # Custom color palette
    palette = ["#3498db", "#e74c3c"]  # Blue and red
    
    # Accuracy plot
    sns.barplot(ax=axes[0], x="Dataset", y="Accuracy", data=cross_metrics, palette=palette)
    axes[0].set_title(f"Accuracy Across Datasets\n({model_name}, {prompt_type})", fontsize=14)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].tick_params(labelsize=12)
    
    # Add value labels on bars
    for i, p in enumerate(axes[0].patches):
        axes[0].annotate(f'{p.get_height():.3f}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'bottom', fontsize=12)
    
    # F1 plot
    sns.barplot(ax=axes[1], x="Dataset", y="F1 Score", data=cross_metrics, palette=palette)
    axes[1].set_title(f"F1 Score Across Datasets\n({model_name}, {prompt_type})", fontsize=14)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("F1 Score", fontsize=12)
    axes[1].tick_params(labelsize=12)
    
    # Add value labels on bars
    for i, p in enumerate(axes[1].patches):
        axes[1].annotate(f'{p.get_height():.3f}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'bottom', fontsize=12)
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Save figure
    plt.savefig(f"figures/cross_dataset_{model_name}_{prompt_type}.png", dpi=300, bbox_inches='tight')
    plt.close()