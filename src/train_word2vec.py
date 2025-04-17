# src/train_word2vec.py
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from word2vec_dataset import Word2VecDataset
from word2vec_model import CBOWModel
from text8_tokenizer import preprocess_text8, build_vocab, save_vocab_json
import time
from tqdm import tqdm
import os
import numpy as np  

# Configuration
DATA_DIR = "data"
TEXT8_PATH = os.path.join(DATA_DIR, "text8")
VOCAB_PATH = os.path.join(DATA_DIR, "text8_vocab.json")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "text8_cbow_model.pt")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "text8_embeddings.npy")

# Hyperparameters
CONTEXT_SIZE = 5
EMBEDDING_DIM = 300
BATCH_SIZE = 512
LR = 0.0005
EPOCHS = 5
MIN_WORD_COUNT = 10

def train():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess text8 data
    print("Loading and tokenizing text8 data...")
    tokens = preprocess_text8(TEXT8_PATH)
    print(f"Total tokens: {len(tokens):,}")

    # Build vocabulary
    print("Building vocabulary...")
    word_to_ix, ix_to_word, vocab = build_vocab(tokens, min_count=MIN_WORD_COUNT)
    vocab_size = len(word_to_ix)
    print(f"Vocabulary size: {vocab_size:,}")

    # Save vocabulary
    save_vocab_json(word_to_ix, ix_to_word, VOCAB_PATH)
    print(f"Vocabulary saved to {VOCAB_PATH}")

    # Create dataset and dataloader
    dataset = Word2VecDataset(tokens, word_to_ix, ix_to_word, context_size=CONTEXT_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = CBOWModel(vocab_size, EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")

        for context, target in progress_bar:
            context, target = context.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(context)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} complete - Avg Loss: {avg_loss:.4f}")

    # Save model and embeddings
    torch.save({
        "model_state_dict": model.state_dict(),
        "word_to_ix": word_to_ix,
        "ix_to_word": ix_to_word,
        "config": {
            "embedding_dim": EMBEDDING_DIM,
            "context_size": CONTEXT_SIZE
        }
    }, MODEL_SAVE_PATH)

    embeddings = model.get_embeddings()
    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"\nTraining completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Embeddings saved to {EMBEDDINGS_PATH}")

if __name__ == "__main__":
    train()
