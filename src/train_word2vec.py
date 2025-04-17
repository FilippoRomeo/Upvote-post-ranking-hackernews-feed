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
import torch.nn.functional as F
import wandb

# Configuration
DATA_DIR = "data"
TEXT8_PATH = os.path.join(DATA_DIR, "text8")
VOCAB_PATH = os.path.join(DATA_DIR, "text8_vocab.json")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "text8_cbow_model.pt")
BEST_MODEL_PATH = os.path.join(DATA_DIR, "best_model.pt")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "text8_embeddings.npy")

# Hyperparameters
config = {
    "context_size": 5,
    "embedding_dim": 100,  # Increased from 100
    "batch_size": 512,    # Decreased from 1024
    "lr": 0.001,            # Increased from 0.001
    "epochs": 5,           # Word2vec needs more epochs
    "min_word_count": 7,
    "architecture": "CBOW"
}

def print_similar_words(word, model, word_to_ix, ix_to_word, top_n=10):
    if word not in word_to_ix:
        print(f"'{word}' not in vocabulary.")
        return

    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings()
        norm_embeddings = F.normalize(torch.tensor(embeddings), p=2, dim=1)

        word_idx = word_to_ix[word]
        word_embedding = norm_embeddings[word_idx].unsqueeze(0)

        cosine_similarities = F.cosine_similarity(word_embedding, norm_embeddings)

        # Exclude the word itself
        top_indices = cosine_similarities.topk(top_n + 1).indices[1:].tolist()
        similar_words = [ix_to_word[str(i)] for i in top_indices]

        print(f"\nTop {top_n} words similar to '{word}':")
        print(", ".join(similar_words))

def train():
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

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
    wandb.run.summary["vocab_size"] = vocab_size

    # Save vocabulary
    save_vocab_json(word_to_ix, ix_to_word, VOCAB_PATH)
    print(f"Vocabulary saved to {VOCAB_PATH}")

    # Create dataset and dataloader
    dataset = Word2VecDataset(tokens, word_to_ix, ix_to_word, context_size=config["context_size"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Initialize model
    model = CBOWModel(vocab_size, config["embedding_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)
    loss_fn = nn.CrossEntropyLoss()
    
    # Watch model
    wandb.watch(model, loss_fn, log="all", log_freq=100)

    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    best_loss = float('inf')
    patience = 3
    no_improvement = 0

    for epoch in range(1, config["epochs"] + 1):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{config['epochs']}", unit="batch")

        for batch_idx, (context, target) in enumerate(progress_bar):
            context, target = context.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(context)
            loss = loss_fn(output, target)
            loss.backward()

            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if batch_idx % 100 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch
                })

                # Print most similar words to a test word
                print_similar_words("apple", model, word_to_ix, ix_to_word, top_n=10)

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} complete - Avg Loss: {avg_loss:.4f}")

        # Log epoch metrics
        wandb.log({
            "epoch_loss": avg_loss,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch_time": (time.time() - start_time) / 60
        })

        # Learning rate scheduling
        scheduler.step(avg_loss)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
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
    artifact.add_file(BEST_MODEL_PATH)
    artifact.add_file(EMBEDDINGS_PATH)
    artifact.add_file(VOCAB_PATH)
    wandb.log_artifact(artifact)

    print(f"\nTraining completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Best model saved to {BEST_MODEL_PATH}")
    print(f"Embeddings saved to {EMBEDDINGS_PATH}")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    train()
