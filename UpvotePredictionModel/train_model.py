import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm  # for progress bar
import wandb  # Optional: only if using Weights & Biases

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Go up one level to reach the root directory
data_path = os.path.join(BASE_DIR, "data", "fetch_data", "hn_dataset.pt")
embedding_path = os.path.join(BASE_DIR, "data", "text8_embeddings.npy")

# === Load Dataset ===
print("📦 Loading dataset...")
dataset = torch.load(data_path)
X, y = dataset['inputs'], dataset['targets']
print(f"✅ Loaded dataset with shape {X.shape} and targets shape {y.shape}")

# === Load CBOW embeddings ===
print("📥 Loading pretrained CBOW vectors...")
embeddings = torch.from_numpy(np.load(embedding_path))  # Shape: [vocab_size, embedding_dim]
embedding_dim = embeddings.shape[1]
print(f"✅ Loaded embeddings with shape {embeddings.shape}")

# === Dataset and Dataloader ===
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# === Deeper Model ===
class DeeperRegressor(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.size()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        embeds = self.embedding(x)
        avg_embeds = embeds.mean(dim=1)
        return self.mlp(avg_embeds).squeeze(1)

model = DeeperRegressor(embeddings)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# === Optional: Initialize wandb ===
use_wandb = False  # Set to True if you want to use Weights & Biases
if use_wandb:
    wandb.init(project="upvote-prediction")

# === Training Loop ===
def train(num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"🚂 Epoch {epoch+1}")

        for batch_idx, (X_batch, y_batch) in progress_bar:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Optional: wandb logging
            if use_wandb and batch_idx % 500 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch_progress": epoch + batch_idx/len(train_loader),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

        avg_loss = total_loss / len(train_loader)

        # === Validation ===
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                loss = criterion(preds, y_batch.float())
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"📉 Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        if use_wandb:
            wandb.log({"train_loss": avg_loss, "val_loss": avg_val_loss, "epoch": epoch+1})

train()
