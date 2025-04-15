import torch
from torch.utils.data import Dataset
from collections import Counter
import re


class Word2VecDataset(Dataset):
    def __init__(self, tokens, context_size=2):
        self.context_size = context_size
        self.tokens = tokens  # <- already tokenized!

        # Build vocab
        self.word_to_ix = {word: i for i, word in enumerate(sorted(set(tokens)))}
        self.ix_to_word = {i: word for word, i in self.word_to_ix.items()}

        # Generate training pairs
        self.data = self.create_context_target_pairs()

    def create_context_target_pairs(self):
        data = []
        for i in range(self.context_size, len(self.tokens) - self.context_size):
            context = [
                self.tokens[j]
                for j in range(i - self.context_size, i + self.context_size + 1)
                if j != i
            ]
            target = self.tokens[i]
            data.append((context, target))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_idxs = torch.tensor([self.word_to_ix[w] for w in context], dtype=torch.long)
        target_idx = torch.tensor(self.word_to_ix[target], dtype=torch.long)
        return context_idxs, target_idx