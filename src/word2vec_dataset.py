# src/word2vec_dataset.py
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
        self.unk_idx = word_to_ix.get('<UNK>', 0)  # Default to 0 if not found
        
        # Generate training pairs for CBOW
        self.data = []
        for i in range(context_size, len(tokens) - context_size):
            context_indices = (
                [self.word_to_ix.get(tokens[j], self.unk_idx) 
                 for j in range(i - context_size, i)] +
                [self.word_to_ix.get(tokens[j], self.unk_idx) 
                 for j in range(i + 1, i + context_size + 1)]
            )
            target = self.word_to_ix.get(tokens[i], self.unk_idx)
            self.data.append((context_indices, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)
