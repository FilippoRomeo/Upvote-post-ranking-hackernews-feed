import json
import os
import pandas as pd
from tokenizer import tokenize_title

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    return vocab['word_to_ix']

def convert_title_to_indices(tokens, word_to_ix):
    indices = [word_to_ix[token] for token in tokens if token in word_to_ix]
    return indices

def process_dataframe(df, vocab_path):
    word_to_ix = load_vocab(vocab_path)
    df['tokens'] = df['title'].apply(tokenize_title)
    df['indices'] = df['tokens'].apply(lambda tokens: convert_title_to_indices(tokens, word_to_ix))
    
    # Remove rows with empty indices
    df = df[df['indices'].map(len) > 0]
    
    return df[['indices', 'score']]

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "data/fetch_data", "hn_2010_stories.csv")
    vocab_path = os.path.join(base_dir, "..", "data", "text8_vocab.json")

    df = pd.read_csv(data_path)
    processed_df = process_dataframe(df, vocab_path)

    out_path = os.path.join(base_dir, "..", "data", "hn_2010_indices.csv")
    processed_df.to_csv(out_path, index=False)
    print(f"✅ Saved indexed titles to {out_path}")
