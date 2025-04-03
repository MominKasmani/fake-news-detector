import os
import time
import logging
import argparse
import openai
from preprocessing import load_and_preprocess_data, generate_few_shot_examples, load_fakenewsnet_dataset
from classification import evaluate_model
from evaluation import calculate_metrics, compare_models_and_prompts
from visualization import plot_confusion_matrix, plot_model_comparison
from utils import create_directories, evaluate_cross_dataset_performance, get_api_key

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('liar_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define dataset paths relative to the script
DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(__file__), "data")

def main(dataset_path=DEFAULT_DATASET_PATH, openai_client=None, sample_size=50, models=None, prompt_types=None):
    """Main function to run LIAR dataset evaluation"""
    try:
        start_time = time.time()
        
        # Create necessary directories
        create_directories()
        
        # Load and preprocess data
        train_data, test_data = load_and_preprocess_data(
            os.path.join(dataset_path, "train.tsv"),
            os.path.join(dataset_path, "test.tsv")
        )
        
        # Generate few-shot examples
        few_shot_examples = generate_few_shot_examples(train_data)
        
        # Define configurations to test if not provided
        if not models:
            models = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"]
        
        if not prompt_types:
            prompt_types = ["zero-shot","few-shot"]
        
        configurations = [(model, prompt_type) for model in models for prompt_type in prompt_types]
        
        # Run evaluations
        all_results = []
        all_metrics = []
        
        for model_name, prompt_type in configurations:
            # Evaluate model
            results_df, total_time, avg_time = evaluate_model(
                test_data, 
                model_name, 
                prompt_type,
                few_shot_examples=few_shot_examples, 
                sample_size=sample_size,
                client=openai_client
            )
            
            # Calculate metrics
            metrics = calculate_metrics(
                results_df, model_name, prompt_type, total_time, avg_time
            )
            
            # Create confusion matrix visualization
            valid_results = results_df[results_df["predicted_label"] != "error"]
            if len(valid_results) > 0:
                plot_confusion_matrix(
                    valid_results["true_label"], 
                    valid_results["predicted_label"], 
                    model_name, 
                    prompt_type
                )
            
            all_results.append(results_df)
            if metrics:
                all_metrics.append(metrics)
        
        # Compare model performance
        metrics_df, combined_results = compare_models_and_prompts(all_results, all_metrics)
        
        # Create model comparison visualization
        if metrics_df is not None:
            plot_model_comparison(metrics_df)
        
        # Try cross-dataset evaluation if FakeNewsNet is available
        try:
            fakenewsnet_train, fakenewsnet_test = load_fakenewsnet_dataset()
            if fakenewsnet_test is not None:
                print("\nRunning cross-dataset evaluation...")
                cross_metrics = evaluate_cross_dataset_performance(
                    test_data, fakenewsnet_test, models[0], prompt_types[0], 
                    sample_size=min(30, sample_size),
                    client=openai_client
                )
        except Exception as e:
            logger.warning(f"Skipping cross-dataset evaluation: {str(e)}")
        
        # Calculate execution time
        end_time = time.time()
        execution_time = (end_time - start_time) / 60  # in minutes
        
        logger.info(f"Evaluation complete in {execution_time:.2f} minutes. "
                    f"Results saved to CSV files and visualizations saved as PNG files.")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LIAR Dataset Fake News Detection')
    parser.add_argument('--dataset_path', default=DEFAULT_DATASET_PATH,
                        help='Path to the LIAR dataset directory')
    parser.add_argument('--api_key', default=None,
                        help='OpenAI API key (if not provided via environment variable or .env file)')
    parser.add_argument('--sample_size', type=int, default=50,
                        help='Number of examples to test (default: 50)')
    parser.add_argument('--models', nargs='+', default=['gpt-4o'],
                        help='List of models to evaluate (default: gpt-4o)')
    parser.add_argument('--prompt_types', nargs='+', default=['few-shot'],
                        help='List of prompt types to evaluate (default: few-shot)')
    
    args = parser.parse_args()
    
    # Get API key and initialize OpenAI client
    api_key = get_api_key(args.api_key)
    openai_client = openai.OpenAI(api_key=api_key)
    
    main(args.dataset_path, openai_client, args.sample_size, args.models, args.prompt_types)