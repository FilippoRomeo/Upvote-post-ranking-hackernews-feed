import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.projection = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.projection.weight)
        self.projection.bias.data.zero_()

    def forward(self, context):
        # context shape: [batch_size, 2*context_size]
        embeds = self.embeddings(context)  # [batch_size, 2*context_size, embedding_dim]
        mean_embeds = embeds.mean(dim=1)   # [batch_size, embedding_dim]
        out = self.projection(mean_embeds)  # [batch_size, vocab_size]
        return out

    def get_embeddings(self):
        return self.embeddings.weight.data.cpu().numpy()
