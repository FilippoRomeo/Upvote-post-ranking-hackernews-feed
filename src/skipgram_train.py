# src/skipgram_train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import defaultdict
import numpy as np

# 1. Model Definition
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, target):
        embeds = self.embeddings(target)        # [batch_size, embedding_dim]
        out = self.linear(embeds)               # [batch_size, vocab_size]
        return out

    def get_embeddings(self):
        return self.embeddings.weight.data.cpu().numpy()

# 2. Dataset Functions
def build_vocab(text):
    vocab = set(text)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    return word_to_ix, ix_to_word

def get_skipgram_data(text, word_to_ix, window_size=2):
    data = []
    for i in range(window_size, len(text) - window_size):
        target = text[i]
        context = text[i - window_size:i] + text[i + 1:i + window_size + 1]
        for ctx in context:
            data.append((word_to_ix[target], word_to_ix[ctx]))
    return data

# 3. Training Function
def train_skipgram(model, data, vocab_size, epochs=10, lr=0.01, negative_samples=5):
    loss_fn = nn.BCEWithLogitsLoss()  # Use BCE for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(data)

        for target, context in data:
            target_tensor = torch.tensor([target], dtype=torch.long)
            context_tensor = torch.tensor([context], dtype=torch.long)

            # Negative Sampling: randomly sample negative words
            neg_samples = random.sample(range(vocab_size), negative_samples)
            neg_samples = [sample for sample in neg_samples if sample != context]  # Ensure we don't pick the context word

            # Concatenate positive and negative samples
            sampled_words = [context] + neg_samples

            # Get the output from the model for the target word
            output = model(target_tensor)  # shape: [1, vocab_size]

            # Convert to probabilities using BCEWithLogitsLoss
            sampled_indices = torch.tensor(sampled_words, dtype=torch.long)

            # Flatten output to match the target's shape for binary classification
            output = output.squeeze(0)[sampled_indices]

            # Binary classification target: 1 for positive (context), 0 for negative samples
            target_labels = torch.tensor([1] + [0] * negative_samples, dtype=torch.float32)

            # Calculate loss
            loss = loss_fn(output, target_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# 4. Run Training
if __name__ == "__main__":
    # Sample text
    text = "we are learning the skip gram model using pytorch word embeddings for ai projects".lower().split()

    # Build vocabulary and training pairs
    word_to_ix, ix_to_word = build_vocab(text)
    data = get_skipgram_data(text, word_to_ix, window_size=2)

    # Initialize and train model
    vocab_size = len(word_to_ix)
    embedding_dim = 50
    model = SkipGramModel(vocab_size, embedding_dim)

    train_skipgram(model, data, vocab_size, epochs=100, lr=5e-4)

    # Save embeddings
    embeddings = model.get_embeddings()
    np.save("skipgram_embeddings.npy", embeddings)

    # Optional: print some example word vectors
    print("\nSample word vectors:")
    for word in ["skip", "model", "ai", "pytorch"]:
        idx = word_to_ix[word]
        print(f"{word}: {embeddings[idx][:5]}...")  # print first 5 dims
