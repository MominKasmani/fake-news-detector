# fake-news-detection

# LLM-Based Fake News Detection

A fake news detection system using OpenAI's GPT models with prompt engineering.

## Repository Structure

- `main.py`: Main execution script
- `preprocessing.py`: Data loading and text processing functions
- `prompts.py`: Templates for zero-shot and few-shot classification
- `classification.py`: OpenAI API interaction for classification
- `evaluation.py`: Results calculation and visualization

## Setup

1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Set your OpenAI API key: `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`

## Usage

1. Download the FakeNewsNet dataset
2. Run: `python main.py`

Results will be saved to the `results/` directory.
