import os
import time
import openai
from tqdm.notebook import tqdm
import pandas as pd
import logging
logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
openai.api_key = "sk-project-4o-key"  

from prompts import (
    improved_zero_shot_template,
    improved_few_shot_template
)
from preprocessing import (
    format_news_for_llm,
    load_fakenewsnet_csv,
    extract_features
)
from evaluation import calculate_metrics
from visualizations import (
    visualize_approach_results,
    plot_class_distribution,
    plot_confusion_matrix
)




def classify_with_openai(news_data, prompt_type="zero_shot", model="gpt-4o"):
    """Classify news using OpenAI API with different prompt strategies"""
    try:
        # Select appropriate prompt template
        if prompt_type == "zero_shot":
            prompt = improved_zero_shot_template.format(news_data=news_data)
        elif prompt_type == "few_shot":
            prompt = improved_few_shot_template.format(news_data=news_data)
        else:
            prompt = improved_zero_shot_template.format(news_data=news_data)
        
        start_time = time.time()
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a sophisticated fact-checking AI. Respond ONLY with 'REAL' or 'FAKE' without any additional explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=10,  # Limit to ensure only classification
            top_p=0.9
        )
        end_time = time.time()
        
        # Extract prediction
        prediction_text = response.choices[0].message.content.strip().upper()
        
        # Simplified prediction parsing
        if prediction_text == "REAL":
            prediction = "real"
        elif prediction_text == "FAKE":
            prediction = "fake"
        else:
            logger.warning(f"Unexpected response: {prediction_text}")
            prediction = None
        
        metadata = {
            'tokens_used': response.usage.total_tokens,
            'processing_time': end_time - start_time,
            'model': model,
            'prompt_type': prompt_type
        }
        
        return prediction, metadata
    
    except Exception as e:
        logger.error(f"Error classifying news: {e}")
        return None, {'error': str(e)}


def test_approaches(test_subset, models=None, prompt_types=None):
    """Test approaches with different models and prompt types"""
    if models is None:
        models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    
    if prompt_types is None:
        prompt_types = ["zero_shot", "few_shot"]
    
    approaches = {}
    
    for model in models:
        for prompt_type in prompt_types:
            approach_name = f"{model}_{prompt_type}"
            
            approaches[approach_name] = lambda row, m=model, pt=prompt_type: classify_with_openai(
                format_news_for_llm(row), 
                prompt_type=pt, 
                model=m
            )
    
    # Evaluate each approach
    approach_results = {}
    
    for name, approach_func in approaches.items():
        logger.info(f"\nTesting approach: {name}")
        results = []
        
        for idx, row in tqdm(test_subset.iterrows(), total=len(test_subset), desc=f"Evaluating {name}"):
            try:
                prediction, metadata = approach_func(row)
                
                if prediction:
                    result = {
                        'id': idx,
                        'title': row['title'],
                        'actual_label': row.get('label', 'unknown'),
                        'predicted_label': prediction,
                        **metadata
                    }
                    results.append(result)
            except Exception as e:
                logger.error(f"Error with {name} on index {idx}: {e}")
        
        if results:
            results_df = pd.DataFrame(results)
            metrics = calculate_metrics(results_df)
            
            approach_results[name] = {
                'results_df': results_df,
                'metrics': metrics
            }
            
            logger.info(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
            logger.info(f"F1 Score: {metrics.get('f1', 0):.4f}")
            logger.info(f"Avg Tokens: {metrics.get('avg_tokens', 0):.0f}")
            logger.info(f"Avg Cost: ${metrics.get('avg_cost', 0):.6f}")
            logger.info(f"Avg Time: {metrics.get('avg_processing_time', 0):.1f}s")
            
            # Save individual approach results
            results_df.to_csv(f"results/{name}_results.csv", index=False)
    
    return approach_results
