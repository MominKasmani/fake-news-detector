import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os
import time

def calculate_metrics(results_df, model_name, prompt_type, total_time, avg_time_per_example):
    """Calculate evaluation metrics from results"""
    # Filter out errors
    valid_results = results_df[results_df["predicted_label"] != "error"]
    
    # Calculate metrics
    if len(valid_results) > 0:
        accuracy = accuracy_score(valid_results["true_label"], valid_results["predicted_label"])
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_results["true_label"], 
            valid_results["predicted_label"],
            average="weighted",
            zero_division=0
        )
        
        # Calculate class-specific metrics
        class_report = classification_report(
            valid_results["true_label"],
            valid_results["predicted_label"],
            output_dict=True
        )
        
        # Calculate token metrics
        avg_input_tokens = valid_results["input_tokens"].mean()
        avg_output_tokens = valid_results["output_tokens"].mean()
        avg_total_tokens = valid_results["total_tokens"].mean()
        
        # Print metrics
        print(f"\n===== Results for {model_name} with {prompt_type} prompt =====")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Processing time: {total_time:.2f} seconds")
        print(f"Average time per example: {avg_time_per_example:.4f} seconds")
        print(f"Throughput: {60/avg_time_per_example:.2f} examples per minute")
        
        # Print token metrics
        print(f"\n===== Token Usage Metrics =====")
        print(f"Average input tokens per article: {avg_input_tokens:.1f}")
        print(f"Average output tokens per article: {avg_output_tokens:.1f}")
        print(f"Average total tokens per article: {avg_total_tokens:.1f}")
        
        # Calculate cost (approximate)
        if model_name == "gpt-3.5-turbo":
            input_cost_per_token = 0.0015 / 1000  # $0.0015 per 1K tokens
            output_cost_per_token = 0.002 / 1000  # $0.002 per 1K tokens
        elif model_name == "gpt-4o":
            input_cost_per_token = 0.01 / 1000  # $0.01 per 1K tokens
            output_cost_per_token = 0.03 / 1000  # $0.03 per 1K tokens
        elif model_name == "gpt-4-turbo":
            input_cost_per_token = 0.01 / 1000  # $0.01 per 1K tokens
            output_cost_per_token = 0.03 / 1000  # $0.03 per 1K tokens
        else:
            input_cost_per_token = 0.01 / 1000  # Default
            output_cost_per_token = 0.03 / 1000  # Default
        
        # Calculate costs based on actual token usage
        cost_per_example = (avg_input_tokens * input_cost_per_token) + (avg_output_tokens * output_cost_per_token)
        total_cost = cost_per_example * len(valid_results)
        
        print(f"\n===== Cost Metrics =====")
        print(f"Estimated cost per example: ${cost_per_example:.6f}")
        print(f"Estimated total cost: ${total_cost:.4f}")
        
        # Create metrics dictionary
        metrics = {
            "model": model_name,
            "prompt_type": prompt_type,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "processing_time": total_time,
            "avg_time_per_example": avg_time_per_example,
            "throughput": 60/avg_time_per_example,
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "avg_total_tokens": avg_total_tokens,
            "cost_per_example": cost_per_example,
            "total_cost": total_cost,
            "sample_size": len(results_df),
            "valid_examples": len(valid_results),
            "class_report": class_report
        }
        
        return metrics
    else:
        print(f"No valid results for {model_name} with {prompt_type} prompt")
        return None

def compare_models_and_prompts(all_results, all_metrics):
    """Compare multiple model configurations"""
    if all_metrics:
        metrics_df = pd.DataFrame([
            {
                "Model": m["model"],
                "Prompt Type": m["prompt_type"],
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1 Score": m["f1_score"],
                "Avg Input Tokens": m["avg_input_tokens"],
                "Avg Output Tokens": m["avg_output_tokens"],
                "Avg Total Tokens": m["avg_total_tokens"],
                "Avg Time (s)": m["avg_time_per_example"],
                "Throughput (examples/min)": m["throughput"],
                "Cost per Example ($)": m["cost_per_example"],
                "Total Cost ($)": m["total_cost"]
            }
            for m in all_metrics
        ])
        
        print("\n===== Model Comparison Summary =====")
        print(metrics_df.to_string(index=False, float_format="%.4f"))
        
        # Create directories for results
        os.makedirs('results', exist_ok=True)
        
        # Save metrics to CSV
        metrics_df.to_csv("results/model_comparison_metrics.csv", index=False)
        
        # Combine all results
        combined_results = pd.concat(all_results)
        combined_results.to_excel("results/all_predictions.xlsx", index=False)
        
        return metrics_df, combined_results
    else:
        print("No valid metrics to compare")
        return None, pd.concat(all_results) if all_results else None