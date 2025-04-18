import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import wandb
from sklearn.preprocessing import StandardScaler

# === Configuration ===
config = {
    "batch_size": 128,
    "learning_rate": 3e-4,
    "epochs": 20,
    "embedding_dim": 300,
    "hidden_dim": 256,
    "dropout": 0.3,
    "early_stopping_patience": 5
}

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, "data", "fetch_data", "hn_dataset.pt")
embedding_path = os.path.join(BASE_DIR, "data", "text8_embeddings.npy")

# === Load and Prepare Data ===
print("📦 Loading dataset...")
dataset = torch.load(data_path)
X, y = dataset['inputs'], dataset['targets']

# Normalize targets
scaler = StandardScaler()
y_norm = torch.FloatTensor(scaler.fit_transform(y.reshape(-1, 1))).squeeze()
print(f"✅ Loaded dataset | X: {X.shape} | y: {y.shape} | Mean score: {y.mean():.1f} ± {y.std():.1f}")

# === Enhanced Model ===
class UpvotePredictor(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=256, dropout=0.3):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.size()
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        
        # Enhanced attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        
        # Deeper MLP with residual connections
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # Pad sequences to same length
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        
        embeds = self.embedding(x)
        attn_weights = self.attention(embeds)
        weighted_avg = (attn_weights * embeds).sum(dim=1)
        
        return self.mlp(weighted_avg).squeeze(1)
# === Training Setup ===
def train_model():
    # Load embeddings
    embeddings = torch.from_numpy(np.load(embedding_path))
    model = UpvotePredictor(embeddings, 
                          hidden_dim=config['hidden_dim'],
                          dropout=config['dropout'])
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), 
                          lr=config['learning_rate'],
                          weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                   patience=2,
                                                   factor=0.5,
                                                   verbose=True)

    # Dataset split
    dataset = TensorDataset(X, y_norm)
    train_size = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_ds, 
                            batch_size=config['batch_size'],
                            shuffle=True,
                            num_workers=4)
    val_loader = DataLoader(val_ds, 
                          batch_size=config['batch_size'],
                          num_workers=4)

    # Initialize wandb
    wandb.init(project="hn-upvote-prediction", config=config)
    wandb.watch(model, log="all")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for X_batch, y_batch in progress_bar:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        # Log metrics
        wandb.log({
            "train_loss": avg_train,
            "val_loss": avg_val,
            "lr": optimizer.param_groups[0]['lr']
        })
        
        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step(avg_val)
        print(f"Epoch {epoch+1} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

    wandb.finish()

if __name__ == "__main__":
    train_model()