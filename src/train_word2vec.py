# src/tokenizer.py

import torch
from torch.utils.data import DataLoader
from word2vec_dataset import Word2VecDataset
from word2vec_model import CBOWModel
import json
import time
from tqdm import tqdm
import os

# Set correct paths (assuming script runs from project root)
DATA_DIR = "data"
TOKENS_PATH = os.path.join(DATA_DIR, "tokens.json")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "cbow_embeddings.pt")

# Load tokens
with open(TOKENS_PATH, "r") as f:
    tokens = json.load(f)

# Create dataset
context_size = 2
dataset = Word2VecDataset(tokens, context_size=context_size)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Model setup
embedding_dim = 100
 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

# Initialize model
model = CBOWModel(len(dataset.word_to_ix), embedding_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 5
start_time = time.time()

for epoch in range(epochs):
    epoch_loss = 0
    progress_bar = tqdm(dataloader, 
                       desc=f"Epoch {epoch+1}/{epochs}",
                       unit="batch",
                       ncols=100)
    
    for context_idxs, target in progress_bar:
        context_idxs = context_idxs.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(context_idxs)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"batch_loss": f"{loss.item():.2f}"})
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"\n✅ Epoch {epoch + 1} complete — Avg Loss: {avg_loss:.4f}")

end_time = time.time()
print(f"\n⏱️  Training completed in {(end_time - start_time)/60:.2f} minutes.")

# Save the model
torch.save({
    "model_state_dict": model.state_dict(),
    "word_to_ix": dataset.word_to_ix,
    "ix_to_word": dataset.ix_to_word,
}, MODEL_SAVE_PATH)
print(f"✅ Trained Word2Vec model saved to {MODEL_SAVE_PATH}")