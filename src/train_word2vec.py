# src/train_word2vec.py

import torch
from torch.utils.data import DataLoader
from word2vec_dataset import Word2VecDataset
from word2vec_model import CBOWModel
import json
import time


# Load tokens
with open("../data/tokens.json", "r") as f:
    tokens = json.load(f)

# Create dataset
context_size = 2
dataset = Word2VecDataset(tokens, context_size=context_size)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Model setup
embedding_dim = 100
 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🖥️  Using device: {device}")


# Train loop
epochs = 5

model = CBOWModel(len(dataset.word_to_ix), embedding_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loss_fn = torch.nn.CrossEntropyLoss()

start_time = time.time()

for epoch in range(epochs):
    total_loss = 0
    batch_count = 0
    for context_idxs, target in dataloader:
        context_idxs = context_idxs.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(context_idxs)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        # Print every N batches to track progress
        if batch_count % 100 == 0:
            print(f"  ↪ Batch {batch_count}, running loss: {total_loss:.2f}")

    print(f"✅ Epoch {epoch + 1}/{epochs} complete — Loss: {total_loss:.4f}")

end_time = time.time()
print(f"⏱️  Training completed in {(end_time - start_time):.2f} seconds.")


# Save the model
torch.save({
    "model_state_dict": model.state_dict(),
    "word_to_ix": dataset.word_to_ix,
    "ix_to_word": dataset.ix_to_word,
}, "data/cbow_embeddings.pt")
print("✅ Trained Word2Vec model saved to data/cbow_embeddings.pt")
