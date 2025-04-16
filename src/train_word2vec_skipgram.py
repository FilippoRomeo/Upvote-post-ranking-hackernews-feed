# train_word2vec_skipgram.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from word2vec_dataset import Word2VecDataset
from word2vec_model import SkipGramModel
from text8_tokenizer import load_vocab  # Add this to your text8_tokenizer.py
import time
from tqdm import tqdm
import os
import numpy as np

# Configuration (same as CBOW)
DATA_DIR = "data"
TEXT8_PATH = os.path.join(DATA_DIR, "text8")
VOCAB_PATH = os.path.join(DATA_DIR, "text8_vocab.pkl")  # Reuse CBOW's vocab
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "text8_skipgram_model.pt")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "text8_skipgram_embeddings.npy")

# Hyperparameters (same as CBOW)
CONTEXT_SIZE = 5
EMBEDDING_DIM = 300
BATCH_SIZE = 512
LR = 0.001
EPOCHS = 10

def train_skipgram():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load existing vocabulary (from CBOW training)
    word_to_ix, ix_to_word = load_vocab(VOCAB_PATH)  # Implement this if missing
    vocab_size = len(word_to_ix)
    print(f"Loaded vocabulary size: {vocab_size:,}")

    # Load tokens (optional: reuse preprocessed tokens if available)
    tokens = preprocess_text8(TEXT8_PATH)  # Ensure this matches CBOW's preprocessing

    # Skip-gram dataset (modify Word2VecDataset to support mode="skipgram")
    dataset = Word2VecDataset(
        tokens, 
        word_to_ix, 
        ix_to_word, 
        context_size=CONTEXT_SIZE, 
        mode="skipgram"  # Critical change
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Skip-gram model
    model = SkipGramModel(vocab_size, EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop (same structure as CBOW)
    print("\nTraining Skip-gram...")
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
        for target, context in progress_bar:  # Note: (target, context) order!
            target, context = target.to(device), context.to(device)
            optimizer.zero_grad()
            output = model(target)  # Skip-gram: target -> context
            loss = loss_fn(output, context)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        print(f"Epoch {epoch} Avg Loss: {epoch_loss / len(dataloader):.4f}")

    # Save Skip-gram model and embeddings
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    embeddings = model.get_embeddings()
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Skip-gram embeddings saved to {EMBEDDINGS_PATH}")

if __name__ == "__main__":
    train_skipgram()