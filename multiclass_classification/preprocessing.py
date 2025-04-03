import nltk
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

def load_and_preprocess_data(train_file, test_file):
    """Load and preprocess LIAR dataset"""
    print("Loading and preprocessing data...")
    
    # Load data
    train = pd.read_csv(train_file, delimiter="\t", encoding="utf-8", on_bad_lines='skip', header=None, 
                     names=[
                         "id", "label", "statement", "subject", "speaker", "speaker_job_title", 
                         "state_info", "party_affiliation", "barely_true_counts", 
                         "false_counts", "half_true_counts", "mostly_true_counts", 
                         "pants_on_fire_counts", "context"
                         ]
                    )

    test = pd.read_csv(test_file, delimiter="\t", encoding="utf-8", on_bad_lines='skip', header=None, 
                     names=[
                         "id", "label", "statement", "subject", "speaker", "speaker_job_title", 
                         "state_info", "party_affiliation", "barely_true_counts", 
                         "false_counts", "half_true_counts", "mostly_true_counts", 
                         "pants_on_fire_counts", "context"
                         ]
                    )
    
    # Initialize NLTK tools
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Text cleaning function
    def clean_text_extended(text):
        if pd.isnull(text):  # Handle NaN values
            return ""
        
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
        text = re.sub(r"[,.-]", " ", text)  # Replace commas, hyphens, and dots with space
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        words = word_tokenize(text.lower())  # Convert to lowercase
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize and remove stopwords
        
        return " ".join(words)

    # Clean all relevant text-based columns
    columns_to_clean = ["statement", "speaker", "speaker_job_title", "context"]
    
    for col in columns_to_clean:
        train[col] = train[col].apply(clean_text_extended)
        test[col] = test[col].apply(clean_text_extended)

    # Convert count-based columns to numeric
    count_columns = ["barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts"]
    train[count_columns] = train[count_columns].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    train.drop(columns=['id'], inplace=True)

    test[count_columns] = test[count_columns].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    test.drop(columns=['id'], inplace=True)
    
    print("Data loading and preprocessing complete.")
    print(f"Train data shape: {train.shape}")
    print(f"Test data shape: {test.shape}")
    
    return train, test

def generate_few_shot_examples(train_data, num_examples=20):
    """Generate few-shot examples from training data"""
    few_shot_examples = train_data.sample(num_examples, random_state=42)
    
    results = []
    for idx, row in few_shot_examples.iterrows():
        statement = f"Statement is '{row['statement']}'."
        speaker_info = f"Speaker and speaker's job title is {row['speaker']} and {row['speaker_job_title']}."
        subject_context = f"Subject and context is {row['subject']} and {row['context']}."
        label_info = f"Label: {row['label']}"
        
        summary = f"{statement} {speaker_info} {subject_context}\n{label_info}\n"
        results.append(summary)

    # Join all examples into one nicely formatted string
    formatted_few_shot_examples = "\n".join(results)
    return formatted_few_shot_examples

def load_fakenewsnet_dataset(dataset_path="data/fakenewsnet"):
    """Load FakeNewsNet dataset if available"""
    from sklearn.model_selection import train_test_split
    import os
    
    try:
        # Check if files exist
        required_files = ["gossipcop_fake.csv", "gossipcop_real.csv", 
                         "politifact_fake.csv", "politifact_real.csv"]
        
        file_path = os.path.join(dataset_path, required_files[0])
        if not os.path.exists(file_path):
            print(f"FakeNewsNet dataset not found at {dataset_path}")
            return None, None
        
        # Load and process data - adjust to actual file format
        dfs = []
        for filename in required_files:
            file_path = os.path.join(dataset_path, filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                label = "fake" if "fake" in filename else "real"
                df["label"] = label
                dfs.append(df)
        
        if not dfs:
            return None, None
            
        # Combine all files
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Split into train/test
        train, test = train_test_split(combined_df, test_size=0.2, random_state=42, stratify=combined_df["label"])
        
        print(f"Loaded FakeNewsNet dataset: {len(combined_df)} samples")
        print(f"Train: {len(train)}, Test: {len(test)}")
        
        return train, test
    
    except Exception as e:
        print(f"Error loading FakeNewsNet dataset: {str(e)}")
        return None, None