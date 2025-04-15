# src/tokenizer.py
import re
import pandas as pd

def simple_tokenizer(text):
    # Lowercase and remove non-alphabetical characters
    text = text.lower()
    return re.findall(r'\b[a-z]{2,}\b', text)

def tokenize_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df['tokens'] = df['title'].apply(simple_tokenizer)
    return df[['tokens', 'score']]

if __name__ == "__main__":
    df = tokenize_dataset("data/hn_posts.csv")
    df.to_csv("data/hn_tokenized.csv", index=False)
