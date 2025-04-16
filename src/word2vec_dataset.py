# src/word2vec_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class Word2VecDataset(Dataset):
    def __init__(self, tokens, word_to_ix, ix_to_word, context_size=5, mode="cbow"):
        """
        Args:
            tokens: List of tokenized words
            word_to_ix: Dictionary mapping words to indices
            ix_to_word: Dictionary mapping indices to words
            context_size: Half the size of the context window
            mode: "cbow" or "skipgram"
        """
        self.context_size = context_size
        self.tokens = tokens
        self.word_to_ix = word_to_ix
        self.ix_to_word = ix_to_word
        self.unk_idx = word_to_ix.get('<UNK>', 0)  # Default to 0 if not found
        self.mode = mode  # Either 'cbow' or 'skipgram'
        
        # Generate training pairs
        self.data = []
        for i in range(context_size, len(tokens) - context_size):
            context_indices = (
                [self.word_to_ix.get(tokens[j], self.unk_idx) 
                 for j in range(i - context_size, i)] +
                [self.word_to_ix.get(tokens[j], self.unk_idx) 
                 for j in range(i + 1, i + context_size + 1)]
            )
            target = self.word_to_ix.get(tokens[i], self.unk_idx)
            
            if self.mode == "cbow":
                # CBOW: Input = context, Output = target
                self.data.append((context_indices, target))
            elif self.mode == "skipgram":
                # Skipgram: Input = target, Output = context (multiple context words per target)
                for context_word in context_indices:
                    self.data.append((target, context_word))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)
