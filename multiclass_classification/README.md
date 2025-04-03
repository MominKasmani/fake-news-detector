# LIAR Dataset Fake News Detection with LLMs

This project implements a fake news detection system using Large Language Models (LLMs) and prompt engineering, optimized for both accuracy and efficiency on the LIAR dataset.

## Overview

The system uses OpenAI's GPT models with carefully engineered prompts to classify political statements into truthfulness categories. Three prompt strategies are implemented:
- Zero-shot classification
- Few-shot classification with examples
- Chain-of-thought reasoning

## Features

- Comprehensive text preprocessing with NLTK
- Multiple prompt engineering strategies
- Evaluation across multiple OpenAI models
- Detailed performance, efficiency, and cost metrics
- Visualizations for model comparison
- Cross-dataset performance analysis (with FakeNewsNet)

## Dataset

This repository includes the LIAR dataset, which contains 12,800+ human-labeled short statements from PolitiFact.com with truthfulness ratings from "pants-on-fire" (completely false) to "true". The dataset was introduced in the paper "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection" by Wang et al.


### Model and Prompt Selection

You can specify which models and prompt types to use
On the command prompt run:
python main.py --models gpt-3.5-turbo gpt-4-turbo gpt-4o --prompt_types zero-shot few-shot cot