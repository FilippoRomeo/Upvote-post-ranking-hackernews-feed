# src/word2vec_model.py
import torch.nn as nn

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idxs):
        # context_idxs: [batch_size, 2*context_size]
        embeds = self.embeddings(context_idxs)
        pooled = embeds.mean(dim=1)  # [batch_size, embedding_dim]
        out = self.linear(pooled)
        return out
