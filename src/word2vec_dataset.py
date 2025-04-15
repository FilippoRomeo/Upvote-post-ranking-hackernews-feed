import torch
from torch.utils.data import Dataset
from collections import Counter
import re


class Word2VecDataset(Dataset):
    def __init__(self, text, context_size=2):
        self.context_size = context_size
        self.vocab = self.build_vocab(text)
        self.word_to_ix = {word: i for i, word in enumerate(self.vocab)}
        self.ix_to_word = {i: word for word, i in self.word_to_ix.items()}
        self.data = self.create_context_target_pairs(text)

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.split()

    def build_vocab(self, text):
        tokens = self.preprocess(text)
        return list(set(tokens))

    def create_context_target_pairs(self, text):
        tokens = self.preprocess(text)
        data = []
        for i in range(self.context_size, len(tokens) - self.context_size):
            context = [tokens[j] for j in range(i - self.context_size, i + self.context_size + 1) if j != i]
            target = tokens[i]
            data.append((context, target))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_idxs = torch.tensor([self.word_to_ix[word] for word in context], dtype=torch.long)
        target_idx = torch.tensor(self.word_to_ix[target], dtype=torch.long)
        return context_idxs, target_idx
