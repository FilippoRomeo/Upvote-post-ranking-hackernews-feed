import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Go up one level to reach the root directory
data_path = os.path.join(BASE_DIR, "data", "fetch_data", "hn_dataset.pt")
embedding_path = os.path.join(BASE_DIR, "data", "text8_vectors.pt")


# === Load Dataset ===
print("📦 Loading dataset...")
dataset = torch.load(data_path)
X, y = dataset['inputs'], dataset['targets']
print(f"✅ Loaded dataset with shape {X.shape} and targets shape {y.shape}")

# === Load CBOW embeddings ===
print("📥 Loading pretrained CBOW vectors...")
embeddings = torch.load(embedding_path)  # Shape: [vocab_size, embedding_dim]
embedding_dim = embeddings.shape[1]
print(f"✅ Loaded embeddings with shape {embeddings.shape}")

# === Dataset and Dataloader ===
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# === Model ===
class AvgEmbeddingRegressor(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.size()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.linear = nn.Linear(emb_dim, 1)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embeds = self.embedding(x)  # [batch_size, seq_len, emb_dim]
        avg_embeds = embeds.mean(dim=1)  # [batch_size, emb_dim]
        out = self.linear(avg_embeds)  # [batch_size, 1]
        return out.squeeze(1)  # [batch_size]

model = AvgEmbeddingRegressor(embeddings)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# === Training Loop ===
def train(num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                loss = criterion(preds, y_batch.float())
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"📉 Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

train()
