import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from word2vec_dataset import Word2VecDataset
from word2vec_model import CBOWModel
from text8_tokenizer import load_tokenized_data
import wandb

# Set paths
DATA_DIR = "data"
TOKENIZED_DATA_PATH = os.path.join(DATA_DIR, "text8")
VOCAB_PATH = os.path.join(DATA_DIR, "text8_vocab.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "text8_embeddings.npy")
BEST_MODEL_PATH = os.path.join(DATA_DIR, "best_model.pt")

# Load config
config = {
    "embedding_dim": 100,
    "context_size": 4,
    "batch_size": 512,
    "epochs": 5,
    "lr": 0.001,
}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize wandb
wandb.init(project="ml-week1-cbow", config=config)

def load_existing_model(model, optimizer, path):
    """Load model and optimizer state if checkpoint exists."""
    if os.path.exists(path):
        print(f"Found existing model at {path}. Loading checkpoint...")
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Checkpoint loaded successfully.")
        return checkpoint.get("epoch", 0), checkpoint.get("loss", float("inf"))
    else:
        print("No existing model found. Starting from scratch.")
        return 0, float("inf")

def train():
    # Load data
    print("Loading tokenized data and vocabulary...")
    tokenized_data = load_tokenized_data(TOKENIZED_DATA_PATH)
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    # Create dataset and dataloader
    dataset = Word2VecDataset(tokenized_data, config["context_size"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Initialize model
    model = CBOWModel(vocab_size, config["embedding_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5, amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    # Try to load from existing checkpoint
    start_epoch, best_loss = load_existing_model(model, optimizer, BEST_MODEL_PATH)

    # Optional fine-tuning tweak: reduce LR
    # for g in optimizer.param_groups:
    #     g["lr"] = config["lr"] * 0.1

    print(f"Starting training from epoch {start_epoch + 1} to {config['epochs']}")

    # Training loop
    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        total_loss = 0.0
        model.train()

        for context, target in dataloader:
            context, target = context.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{config['epochs']} - Loss: {avg_loss:.4f}")
        wandb.log({"loss": avg_loss, "epoch": epoch})

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, BEST_MODEL_PATH)
            print(f"Best model saved at epoch {epoch} with loss {best_loss:.4f}")

    # Save final embeddings
    embeddings = model.embeddings.weight.data.cpu().numpy()
    with open(EMBEDDINGS_PATH, "wb") as f:
        import numpy as np
        np.save(f, embeddings)
    print("Final embeddings saved.")

if __name__ == "__main__":
    train()
