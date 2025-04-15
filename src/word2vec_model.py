import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idxs):
        # context_idxs: [batch_size, context_size * 2]
        embeds = self.embeddings(context_idxs)  # [batch_size, context_size*2, embedding_dim]
        mean_embed = embeds.mean(dim=1)         # [batch_size, embedding_dim]
        out = self.linear1(mean_embed)          # [batch_size, vocab_size]
        return out
