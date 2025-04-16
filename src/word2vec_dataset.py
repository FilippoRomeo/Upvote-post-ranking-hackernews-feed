import torch
from torch.utils.data import Dataset
import numpy as np

class Word2VecDataset(Dataset):
    def __init__(self, tokens, word_to_ix, ix_to_word, context_size=5):
        """
        Args:
            tokens: List of tokenized words
            word_to_ix: Dictionary mapping words to indices
            ix_to_word: Dictionary mapping indices to words
            context_size: Half the size of the context window
        """
        self.context_size = context_size
        self.tokens = tokens
        self.word_to_ix = word_to_ix
        self.ix_to_word = ix_to_word
        self.unk_idx = word_to_ix.get('<unk>', 1)
        
        # Generate training pairs
        self.data = []
        for i in range(context_size, len(tokens) - context_size):
            context = (
                [self.word_to_ix.get(tokens[j], self.unk_idx) 
                 for j in range(i - context_size, i)] +
                [self.word_to_ix.get(tokens[j], self.unk_idx) 
                 for j in range(i + 1, i + context_size + 1)]
            )
            target = self.word_to_ix.get(tokens[i], self.unk_idx)
            self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)
# word2vec_dataset.py
class Word2VecDataset:
    def __init__(self, tokens, word_to_ix, ix_to_word, context_size=5, mode="cbow"):
        self.tokens = tokens
        self.word_to_ix = word_to_ix
        self.ix_to_word = ix_to_word
        self.context_size = context_size
        self.mode = mode  # "cbow" or "skipgram"

    def __len__(self):
        return len(self.tokens) - 2 * self.context_size

    def __getitem__(self, idx):
        center = idx + self.context_size
        context_indices = [
            idx + i for i in range(2 * self.context_size + 1) 
            if i != self.context_size
        ]
        
        if self.mode == "cbow":
            context = [self.word_to_ix[self.tokens[i]] for i in context_indices]
            target = self.word_to_ix[self.tokens[center]]
            return torch.tensor(context), torch.tensor(target)
        elif self.mode == "skipgram":
            target = self.word_to_ix[self.tokens[center]]
            context = [self.word_to_ix[self.tokens[i]] for i in context_indices]
            # Return multiple (target, context) pairs per window
            return torch.tensor(target), torch.tensor(context[0])  # Simplified: use 1 context