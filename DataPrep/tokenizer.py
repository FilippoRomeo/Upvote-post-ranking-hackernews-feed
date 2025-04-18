# DataPrep/tokenizer.py
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (one-time download)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def tokenize_title(title):
    # Step 1: Convert to lowercase
    title = title.lower()
    
    # Step 2: Remove punctuation and non-alphabetic characters (you can modify this as needed)
    title = re.sub(r'[^a-z0-9\s]', '', title)
    
    # Step 3: Tokenize the title into words
    tokens = title.split()
    
    # Step 4: Remove stopwords (optional, can be adjusted based on your needs)
    tokens = [token for token in tokens if token not in stop_words]
    
    # Step 5: Optional - Special handling for multi-word terms like "Web 2.0"
    # (You can create custom logic here if you want to merge terms like "Web 2.0" into one token)
    
    return tokens

def tokenize_all_titles(df):
    # Apply tokenization to all titles in the DataFrame
    return df['title'].apply(tokenize_title)
