import os
import sys

# Add parent directory to Python path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from train_model import UpvotePredictor, config
import pandas as pd
from DataPrep.tokenizer import tokenize_title
from DataPrep.title_to_indices import titles_to_indices

# === Paths ===
data_path = os.path.join(BASE_DIR, "data", "fetch_data", "hn_dataset.pt")
embedding_path = os.path.join(BASE_DIR, "data", "text8_embeddings.npy")
vocab_path = os.path.join(BASE_DIR, "data", "text8_vocab.json")
model_path = os.path.join(BASE_DIR, "data", "best_model.pt")
csv_path = os.path.join(BASE_DIR, "data", "fetch_data", "hn_2010_stories.csv")

def load_vocab():
    import json
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    return vocab['word_to_ix']

def predict_upvotes(model, titles, word_to_ix):
    # Tokenize and convert to indices
    tokens = [tokenize_title(title) for title in titles]
    indices = titles_to_indices(tokens, word_to_ix)
    
    # Convert to tensor and predict
    with torch.no_grad():
        preds = model(indices)
    
    return preds.numpy()

def main():
    # Load model
    embeddings = torch.from_numpy(np.load(embedding_path))
    model = UpvotePredictor(embeddings, 
                          hidden_dim=config['hidden_dim'],
                          dropout=config['dropout'])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load vocabulary
    word_to_ix = load_vocab()
    
    # Load original dataset to get titles
    df = pd.read_csv(csv_path)
    
    # Get 10 random titles from the dataset
    sample_titles = df.sample(n=10)['title'].tolist()
    
    # Make predictions
    predictions = predict_upvotes(model, sample_titles, word_to_ix)
    
    # Get actual scores for comparison
    actual_scores = df[df['title'].isin(sample_titles)]['score'].values
    
    # Display results
    print("\n=== Upvote Predictions ===")
    print("Title | Predicted Score | Actual Score")
    print("-" * 70)
    for title, pred_score, actual_score in zip(sample_titles, predictions, actual_scores):
        print(f"{title[:40]}... | {pred_score:.1f} | {actual_score}")

if __name__ == "__main__":
    main() 