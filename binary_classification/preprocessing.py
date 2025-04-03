import ssl
import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse

# SSL and NLTK Setup
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set NLTK data directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Download NLTK resources
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
nltk.data.path.append(nltk_data_dir)

def load_fakenewsnet_csv(base_path):
    """Load FakeNewsNet dataset from CSV files"""
    csv_files = [
        ("gossipcop_fake.csv", "fake"),
        ("gossipcop_real.csv", "real"),
        ("politifact_fake.csv", "fake"),
        ("politifact_real.csv", "real")
    ]
    dfs = []
    
    for filename, label in csv_files:
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['label'] = label
                source = filename.split('_')[0]
                df['source'] = source
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"Warning: File {file_path} does not exist")
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
def enhanced_clean_text(text):
    """Enhanced text preprocessing with stopword removal and lemmatization"""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)


def extract_domain(url):
    """Extract the domain from a URL"""
    if pd.isna(url) or url is None:
        return ""
    
    try:
        if not url.startswith('http'):
            url = 'https://' + url
        
        domain = urlparse(url).netloc
        domain = re.sub(r'^www\.', '', domain)
        return domain
    except:
        return ""


def extract_features(row):
    """Extract additional features from news data"""
    features = {}
    
    title = row.get('title', '') if pd.notna(row.get('title')) else ""
    features['title_length'] = len(title)
    features['title_word_count'] = len(title.split())
    features['title_has_question_mark'] = '?' in title
    features['title_has_exclamation_mark'] = '!' in title
    features['title_uppercase_ratio'] = sum(1 for c in title if c.isupper()) / len(title) if len(title) > 0 else 0
    features['title_capital_words'] = sum(1 for word in title.split() if word.istitle())
    
    domain = row.get('domain', '') if pd.notna(row.get('domain')) else ""
    features['domain_length'] = len(domain)
    features['domain_word_count'] = len(domain.split('.'))
    
    # Add emotional language detection
    emotional_terms = ['shocking', 'amazing', 'incredible', 'bombshell', 'outrage', 'scandal', 'breaking']
    features['emotional_language_count'] = sum(1 for term in emotional_terms if term in title.lower())
    
    return features


def format_news_for_llm(row):
    """Format a news entry for the LLM with enhanced features"""
    formatted = f"Title: {row.get('title', '')}\n"
    
    # Add additional features when available
    if row.get('title_uppercase_ratio', 0) > 0.2:
        formatted += f"Note: Title contains {row.get('title_uppercase_ratio', 0):.1%} uppercase characters\n"
    
    if row.get('title_has_question_mark'):
        formatted += "Note: Title contains question mark\n"
        
    if row.get('title_has_exclamation_mark'):
        formatted += "Note: Title contains exclamation mark\n"
    
    if row.get('emotional_language_count', 0) > 0:
        formatted += f"Note: Title contains {row.get('emotional_language_count')} emotional/sensational terms\n"
    
    return formatted.strip()