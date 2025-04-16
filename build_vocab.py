# build_vocab.py
import os
import pickle
from collections import Counter

TEXT8_PATH = "data/text8"
VOCAB_PATH = "data/text8_vocab.pkl"
VOCAB_SIZE = 10000  # You can change this

def preprocess_text8(path):
    with open(file_path, 'r') as f:
        text = f.read()
    tokens = text.lower().split()
    return tokens

def build_vocab(tokens, vocab_size):
    word_counts = Counter(tokens)
    most_common = word_counts.most_common(vocab_size - 1)
    
    word_to_ix = {word: i for i, (word, _) in enumerate(most_common, start=1)}
    word_to_ix["<UNK>"] = 0
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    
    return word_to_ix, ix_to_word

def save_vocab(word_to_ix, ix_to_word, path):
    with open(path, "wb") as f:
        pickle.dump((word_to_ix, ix_to_word), f)

if __name__ == "__main__":
    tokens = preprocess_text8(TEXT8_PATH)
    word_to_ix, ix_to_word = build_vocab(tokens, VOCAB_SIZE)
    save_vocab(word_to_ix, ix_to_word, VOCAB_PATH)
    print(f"Vocabulary saved to {VOCAB_PATH}")

