import os
import sys

# Add parent directory to Python path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from DataPrep.tokenizer import tokenize_title
from DataPrep.title_to_indices import titles_to_indices

# === Configuration ===
config = {
    "batch_size": 128,
    "learning_rate": 3e-4,
    "epochs": 20,
    "embedding_dim": 300,
    "hidden_dim1": 128,  # Updated to match checkpoint
    "hidden_dim2": 64,   # Updated to match checkpoint
    "dropout": 0.3,
    "early_stopping_patience": 5
}

# === Paths ===
data_path = os.path.join(BASE_DIR, "data", "fetch_data", "hn_dataset.pt")
embedding_path = os.path.join(BASE_DIR, "data", "text8_embeddings.npy")
vocab_path = os.path.join(BASE_DIR, "data", "text8_vocab.json")
model_path = os.path.join(BASE_DIR, "data", "hn_regressor_model.pt")
csv_path = os.path.join(BASE_DIR, "data", "fetch_data", "hn_2010_stories.csv")

class UpvotePredictor(torch.nn.Module):
    def __init__(self, embedding_matrix, hidden_dim1=128, hidden_dim2=64, dropout=0.3):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.size()
        
        self.embeddings = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        
        # Fully connected layers with correct sizes
        self.fc1 = torch.nn.Linear(emb_dim, hidden_dim1)  # 300 -> 128
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)  # 128 -> 64
        self.fc3 = torch.nn.Linear(hidden_dim2, 1)  # 64 -> 1
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout)
        
        # Activation
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        # Pad sequences to same length
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        
        # Get embeddings
        embeds = self.embeddings(x)
        
        # Average pooling
        avg_embeds = embeds.mean(dim=1)
        
        # Fully connected layers
        x = self.relu(self.fc1(avg_embeds))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x.squeeze(1)

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
                          hidden_dim1=config['hidden_dim1'],
                          hidden_dim2=config['hidden_dim2'],
                          dropout=config['dropout'])
    
    # Load checkpoint and extract model state
    checkpoint = torch.load(model_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
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