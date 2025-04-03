#### This project implements a comprehensive fake news detection system using Large Language Models (LLMs) and prompt engineering, optimized for both accuracy and efficiency. It includes implementations for two popular datasets: FakeNewsNet and LIAR.

## Overview

The system uses OpenAI's GPT models with carefully engineered prompts to classify news articles and political statements as real or fake. Three prompt strategies are implemented:
- Zero-shot classification
- Few-shot classification with examples
- Chain-of-thought reasoning

## Features

- Comprehensive text preprocessing with NLTK
- Multiple prompt engineering strategies
- Evaluation across multiple OpenAI models
- Detailed performance, efficiency, and cost metrics
- Visualizations for model comparison
- Cross-dataset performance analysis

## Datasets

This repository includes two popular fake news datasets:

### FakeNewsNet Dataset
A comprehensive fake news detection dataset containing news content with social context from different platforms, focusing on:
- **GossipCop**: Celebrity news and entertainment articles
- **PolitiFact**: Primarily political news from a reputable fact-checking website

### LIAR Dataset
A collection of 12,800+ human-labeled short statements from PolitiFact.com with truthfulness ratings from "pants-on-fire" (completely false) to "true". Introduced in the paper "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection" by Wang et al.

## Setup and Installation

1. Clone this repository:
git clone https://github.com/MominKasmani/fake-news-detection.git
cd fake-news-detection
Copy
2. Install required packages:
pip install -r requirements.txt
Copy
3. Both datasets are already included in the repository.

## API Key Setup

You can provide your OpenAI API key in any of the following ways:

1. **Command-line argument**:
python main.py --api_key your_api_key_here
Copy
2. **Environment variable** (recommended for security):
export OPENAI_API_KEY="your_api_key_here"
python main.py
Copy
3. **Configuration file**:
Create a `.env` file in the repository root with:
OPENAI_API_KEY=your_api_key_here
Copy
4. **Direct input**:
If no API key is provided through the above methods, the program will use author's temporary API key.

**Security Note**: For security reasons, never commit your API key to the repository. The `.env` file is included in `.gitignore` to prevent accidental commits.

## Usage

### Running FakeNewsNet Dataset Analysis
python fakenewsnet/main.py
Copy
### Running LIAR Dataset Analysis
python liar/main.py
Copy
### Model and Prompt Selection

You can specify which models and prompt types to use:
python liar/main.py --models gpt-3.5-turbo gpt-4o --prompt_types zero-shot few-shot cot
Copy
Available options:
- **Models**: `gpt-3.5-turbo`, `gpt-4-turbo`, `gpt-4o` (availability depends on your OpenAI account access)
- **Prompt types**: `zero-shot`, `few-shot`,


### 📁 Project Structure
```
fake-news-detector/
├── binary_classification/
│   ├── main.py
│   ├── preprocessing.py
│   ├── prompts.py
│   ├── classification.py
│   ├── evaluation.py
│   └── visualization.py
│
├── multiclass_classification/
│   ├── main.py
│   ├── preprocessing.py
│   ├── prompts.py
│   ├── classification.py
│   ├── evaluation.py
│   └── visualization.py
│
├── data/
│   ├── fakenewsnet/
│   └── liar/
│
├── utils.py
├── results/
├── figures/
└── requirements.txt
```
## Evaluation Metrics

This project evaluates fake news detection on multiple dimensions:
- **Classification Performance**: Accuracy, Precision, Recall, F1-score
- **Efficiency**: Tokens per article, Cost per classification
- **Technical**: Processing time, Cross-dataset performance
