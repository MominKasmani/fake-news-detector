import os
import json
import time
import dotenv

def create_directories():
    """Create necessary directories for the project"""
    dirs = ["results", "figures", "cache", "data"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        
    print("Created necessary directories")

def save_cache(data, filename):
    """Save data to cache file"""
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    filepath = os.path.join(cache_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    print(f"Saved cache to {filepath}")

def load_cache(filename):
    """Load data from cache file if it exists"""
    filepath = os.path.join("cache", filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"Loaded cache from {filepath}")
        return data
    
    return None

def get_api_key(cmd_line_key=None):
    """Get OpenAI API key from various sources in priority order"""
    # 1. Command-line argument
    if cmd_line_key:
        return cmd_line_key
    
    # 2. Environment variable
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    
    # 3. Configuration file
    try:
        dotenv.load_dotenv()
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            return env_key
    except:
        pass
    
    # 4. Request from user
    print("OpenAI API key not found. Please enter your API key:")
    return input("> ").strip()

def evaluate_cross_dataset_performance(liar_test, fakenewsnet_test, model_name, prompt_type, sample_size=50, client=None):
    """Evaluate how a model trained on one dataset performs on another dataset"""
    from classification import evaluate_model
    from preprocessing import generate_few_shot_examples
    from visualization import plot_cross_dataset_comparison
    from evaluation import calculate_metrics
    import pandas as pd
    
    print(f"\n===== Cross-Dataset Performance: {model_name} with {prompt_type} prompt =====")
    
    # Sample data for efficiency
    if sample_size:
        liar_sample = liar_test.sample(min(sample_size, len(liar_test)), random_state=42)
        fakenewsnet_sample = fakenewsnet_test.sample(min(sample_size, len(fakenewsnet_test)), random_state=42)
    else:
        liar_sample = liar_test
        fakenewsnet_sample = fakenewsnet_test
    
    # Generate few-shot examples from LIAR dataset
    few_shot_examples = generate_few_shot_examples(liar_test)
    
    # Evaluate on LIAR (binary classification for comparison)
    # First convert LIAR multi-class to binary
    liar_sample['binary_label'] = liar_sample['label'].apply(
        lambda x: 'real' if x in ['true', 'mostly-true'] else 'fake'
    )
    
    # Evaluate on LIAR
    liar_results, liar_total_time, liar_avg_time = evaluate_model(
        liar_sample, 
        model_name, 
        prompt_type,
        few_shot_examples=few_shot_examples, 
        sample_size=None,  # Already sampled
        client=client
    )
    
    # Calculate LIAR metrics
    liar_metrics = calculate_metrics(
        liar_results, model_name, prompt_type, liar_total_time, liar_avg_time
    )
    
    # Evaluate on FakeNewsNet using LIAR examples
    # We need to adapt the row format for compatibility
    if fakenewsnet_sample is not None and not fakenewsnet_sample.empty:
        # Add missing columns expected by the LIAR prompt
        for col in ['speaker', 'speaker_job_title', 'context', 'barely_true_counts', 
                    'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']:
            if col not in fakenewsnet_sample.columns:
                fakenewsnet_sample[col] = ''
        
        fakenewsnet_results, fnnet_total_time, fnnet_avg_time = evaluate_model(
            fakenewsnet_sample, 
            model_name, 
            prompt_type,
            few_shot_examples=few_shot_examples, 
            sample_size=None,  # Already sampled
            client=client
        )
        
        # Calculate FakeNewsNet metrics
        fakenewsnet_metrics = calculate_metrics(
            fakenewsnet_results, model_name, prompt_type, fnnet_total_time, fnnet_avg_time
        )
    else:
        fakenewsnet_metrics = None
    
    # Compare performance
    if liar_metrics and fakenewsnet_metrics:
        cross_metrics = pd.DataFrame([
            {
                "Dataset": "LIAR",
                "Accuracy": liar_metrics["accuracy"],
                "F1 Score": liar_metrics["f1_score"],
                "Avg Tokens": liar_metrics["avg_total_tokens"],
                "Cost per Example": liar_metrics["cost_per_example"]
            },
            {
                "Dataset": "FakeNewsNet",
                "Accuracy": fakenewsnet_metrics["accuracy"],
                "F1 Score": fakenewsnet_metrics["f1_score"],
                "Avg Tokens": fakenewsnet_metrics["avg_total_tokens"],
                "Cost per Example": fakenewsnet_metrics["cost_per_example"]
            }
        ])
        
        print("\nCross-Dataset Performance Comparison:")
        print(cross_metrics.to_string(index=False, float_format="%.4f"))
        
        # Create enhanced visualization for cross-dataset comparison
        plot_cross_dataset_comparison(cross_metrics, model_name, prompt_type)
        
        # Save cross-dataset metrics
        os.makedirs('results', exist_ok=True)
        cross_metrics.to_csv(f"results/cross_dataset_{model_name}_{prompt_type}.csv", index=False)
        
        return cross_metrics
    else:
        print("Insufficient metrics for cross-dataset comparison")
        return None