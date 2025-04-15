import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idxs):
        embedded = self.embeddings(context_idxs)
        mean_embed = embedded.mean(dim=0)
        out = self.linear1(mean_embed)
        return out
