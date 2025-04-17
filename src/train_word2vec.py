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
import wandb

# Configuration
DATA_DIR = "data"
TEXT8_PATH = os.path.join(DATA_DIR, "text8")
VOCAB_PATH = os.path.join(DATA_DIR, "text8_vocab.json")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "text8_cbow_model.pt")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "text8_embeddings.npy")

# Hyperparameters
config = {
    "context_size": 5,
    "embedding_dim": 300,  # Increased from 100
    "batch_size": 1024,    # Increased from 512
    "lr": 0.01,           # Increased from 0.001
    "epochs": 5,         # Word2vec needs more epochs
    "min_word_count": 5,
    "architecture": "CBOW"
}

def train():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(project="word2vec-cbow", config=config)
    
    # Load and preprocess text8 data
    print("Loading and tokenizing text8 data...")
    tokens = preprocess_text8(TEXT8_PATH)
    print(f"Total tokens: {len(tokens):,}")

    # Build vocabulary
    print("Building vocabulary...")
    word_to_ix, ix_to_word, vocab = build_vocab(tokens, min_count=config["min_word_count"])
    vocab_size = len(word_to_ix)
    print(f"Vocabulary size: {vocab_size:,}")
    wandb.config.update({"vocab_size": vocab_size})  # Log dynamic config

    # Save vocabulary
    save_vocab_json(word_to_ix, ix_to_word, VOCAB_PATH)
    print(f"Vocabulary saved to {VOCAB_PATH}")

    # Create dataset and dataloader
    dataset = Word2VecDataset(tokens, word_to_ix, ix_to_word, context_size=config["context_size"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Initialize model
    model = CBOWModel(vocab_size, config["embedding_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    # Watch model (optional)
    wandb.watch(model, loss_fn, log="all", log_freq=100)

    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    # Before training loop:
    best_loss = float('inf')
    patience = 3
    no_improvement = 0

    # Inside epoch loop (after calculating avg_loss):
    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improvement = 0
        # Save best model
        torch.save(model.state_dict(), "best_model.pt")
    else:
        no_improvement += 1
        if no_improvement >= patience:
            print(f"No improvement for {patience} epochs, early stopping")
            break

    # Add learning rate scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)
    scheduler.step(avg_loss)

    for epoch in range(1, config["epochs"] + 1):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{config['epochs']}", unit="batch")

        for batch_idx, (context, target) in enumerate(progress_bar):
            context, target = context.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(context)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if batch_idx % 100 == 0:
                wandb.log({"batch_loss": loss.item(), "epoch": epoch})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} complete - Avg Loss: {avg_loss:.4f}")

        # Logging after the epoch
        wandb.log({
            "epoch_loss": avg_loss,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch_time": (time.time() - start_time)/60
        })

        scheduler.step(avg_loss)

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"No improvement for {patience} epochs, early stopping")
                break

    # Save model and embeddings
    torch.save({
        "model_state_dict": model.state_dict(),
        "word_to_ix": word_to_ix,
        "ix_to_word": ix_to_word,
        "config": {
            "embedding_dim": config["embedding_dim"],
            "context_size": config["context_size"]
        }
    }, MODEL_SAVE_PATH)

    embeddings = model.get_embeddings()
    np.save(EMBEDDINGS_PATH, embeddings)

    # Log artifacts
    artifact = wandb.Artifact('trained-model', type='model')
    artifact.add_file(MODEL_SAVE_PATH)
    artifact.add_file(EMBEDDINGS_PATH)
    artifact.add_file(VOCAB_PATH)
    wandb.log_artifact(artifact)

    print(f"\nTraining completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Embeddings saved to {EMBEDDINGS_PATH}")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    train()