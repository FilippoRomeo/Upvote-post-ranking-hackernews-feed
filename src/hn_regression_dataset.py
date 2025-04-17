#src/hn_regression_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np


class HNDataset(Dataset):
    """
    Dataset for Hacker News titles and upvote scores.
    Each sample is (avg_title_embedding, upvote_score).
    """
    def __init__(self, df, embeddings: np.ndarray, vocab: dict, tokenizer):
        self.titles = df['title'].tolist()
        self.scores = df['upvote_score'].astype(float).tolist()
        self.embeddings = embeddings
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.titles[idx].lower())
        indices = [self.vocab[t] for t in tokens if t in self.vocab]
        if indices:
            emb = np.mean(self.embeddings[indices], axis=0)
        else:
            emb = np.zeros(self.embeddings.shape[1], dtype=float)
        emb_tensor = torch.tensor(emb, dtype=torch.float32)
        score_tensor = torch.tensor(self.scores[idx], dtype=torch.float32)
        return emb_tensor, score_tensor