import torch
from torch.utils.data import DataLoader
from word2vec_dataset import Word2VecDataset
from word2vec_model import CBOWModel

# Hyperparameters
EMBEDDING_DIM = 50
CONTEXT_SIZE = 2
EPOCHS = 30
LR = 0.01

# Example text
text = "the quick brown fox jumps over the lazy dog"

# Dataset & DataLoader
dataset = Word2VecDataset(text, context_size=CONTEXT_SIZE)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Model, loss, optimizer
model = CBOWModel(vocab_size=len(dataset.vocab), embedding_dim=EMBEDDING_DIM)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for context_idxs, target in dataloader:
        model.zero_grad()
        output = model(context_idxs.squeeze(0))
        loss = loss_fn(output.unsqueeze(0), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
