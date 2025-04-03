import time
import pandas as pd
from tqdm import tqdm
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)

def extract_label(prediction):
    """Extract label from model prediction with improved consistency"""
    prediction = prediction.lower().strip()
    
    # Define valid labels
    valid_labels = ["true", "mostly-true", "half-true", "false", "barely-true", "pants-on-fire"]
    
    # Direct match
    for label in valid_labels:
        if prediction == label:
            return label
    
    # Partial match (in case the model outputs additional text)
    for label in valid_labels:
        if label in prediction:
            return label
    
    # Default to most common class if no match found
    return "half-true"  # Or use your most common class

def evaluate_model(test_data, model_name, prompt_type, few_shot_examples=None, sample_size=None, client=None):
    """Evaluate model performance with a specific prompt type"""
    from prompts import generate_zero_shot_prompt, generate_few_shot_prompt
    import os
    
    print(f"\nEvaluating {model_name} with {prompt_type} prompt...")
    
    # Sample data if needed
    if sample_size and sample_size < len(test_data):
        eval_data = test_data.sample(sample_size, random_state=42)
    else:
        eval_data = test_data.copy()
    
    results = []
    
    # Record start time for performance measurement
    start_time = time.time()
    
    # Process each example
    for _, row in tqdm(eval_data.iterrows(), total=len(eval_data), desc=f"{model_name}_{prompt_type}"):
        # Select prompt based on type
        if prompt_type == "zero-shot":
            prompt = generate_zero_shot_prompt(row)
        elif prompt_type == "few-shot":
            prompt = generate_few_shot_prompt(row, few_shot_examples)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        # Call API with retry logic
        max_retries = 3
        retry_count = 0
        success = False
        
        while not success and retry_count < max_retries:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a Fake News detection AI."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,  # Increased to capture reasoning
                    temperature=0
                )
                
                prediction_text = response.choices[0].message.content.strip()
                
                # Extract label using the improved function
                prediction = extract_label(prediction_text)
                
                # Track token usage
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                success = True
                
            except Exception as e:
                retry_count += 1
                print(f"Error (attempt {retry_count}/{max_retries}): {str(e)}")
                if retry_count < max_retries:
                    time.sleep(2)  # Wait before retrying
                else:
                    prediction = "error"
                    input_tokens = 0
                    output_tokens = 0
                    total_tokens = 0
        
        # Store result
        result = {
            "true_label": row["label"],
            "predicted_label": prediction,
            "statement": row["statement"],
            "speaker": row["speaker"],
            "model": model_name,
            "prompt_type": prompt_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        }
        results.append(result)
    
    # Calculate total processing time
    total_time = time.time() - start_time
    avg_time_per_example = total_time / len(eval_data)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create directories for results
    os.makedirs('results', exist_ok=True)
    
    # Save results to file
    results_df.to_csv(f"results/{model_name}_{prompt_type}_predictions.csv", index=False)
    
    return results_df, total_time, avg_time_per_example