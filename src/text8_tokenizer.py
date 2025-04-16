import re
from collections import Counter
import numpy as np

def simple_tokenizer(text):
    """Basic tokenizer that splits on whitespace and lowercases"""
    return text.lower().split()

def preprocess_text8(file_path):
    """Load and tokenize text8 data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return simple_tokenizer(text)

def build_vocab(tokens, min_count=5):
    """Build vocabulary from tokens with minimum count threshold"""
    word_counts = Counter(tokens)
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    
    word_to_ix = {word: i+2 for i, word in enumerate(sorted(vocab))}  # 0=pad, 1=unk
    word_to_ix['<pad>'] = 0
    word_to_ix['<unk>'] = 1
    
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    
    return word_to_ix, ix_to_word, vocab

def save_vocab(word_to_ix, ix_to_word, file_path):
    """Save vocabulary to file"""
    import pickle
    with open(file_path, 'wb') as f:
        pickle.dump({
            'word_to_ix': word_to_ix,
            'ix_to_word': ix_to_word
        }, f)

def load_vocab(file_path):
    """Load vocabulary from file"""
    import pickle
    with open(file_path, 'rb') as f:
        return pickle.load(f)
