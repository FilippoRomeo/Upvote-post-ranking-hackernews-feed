import os
import pickle
from collections import Counter

TEXT8_PATH = "text8"  # Changed from glove file to text8
VOCAB_PATH = "data/vocab.pkl"  # Changed output path
VOCAB_SIZE = 10000

def preprocess_text8(path):
    """Load and tokenize text8 file"""
    with open(path, 'r') as f:  # Fixed parameter name from file_path to path
        text = f.read()
    # Simple tokenization - split on whitespace and lowercase
    tokens = text.lower().split()
    return tokens

def build_vocab(tokens, vocab_size):
    """Build vocabulary from tokens"""
    word_counts = Counter(tokens)
    most_common = word_counts.most_common(vocab_size - 1)  # Reserve 0 for UNK
    
    word_to_ix = {word: i for i, (word, _) in enumerate(most_common, start=1)}
    word_to_ix["<UNK>"] = 0  # Add unknown token
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    
    return word_to_ix, ix_to_word

def save_vocab(word_to_ix, ix_to_word, path):
    """Save vocabulary to file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Create dir if needed
    with open(path, "wb") as f:
        pickle.dump((word_to_ix, ix_to_word), f)

def load_vocab(path):
    with open(path, "rb") as f:
        word_to_ix, ix_to_word = pickle.load(f)
    return word_to_ix, ix_to_word

if __name__ == "__main__":
    # Check if text8 exists
    if not os.path.exists(TEXT8_PATH):
        print(f"Error: {TEXT8_PATH} not found.")
        print("Please download text8 first and place it in the project root.")
    else:
        print("Processing text8 file...")
        tokens = preprocess_text8(TEXT8_PATH)
        print(f"Found {len(tokens)} tokens")
        
        print("Building vocabulary...")
        word_to_ix, ix_to_word = build_vocab(tokens, VOCAB_SIZE)
        
        print("Saving vocabulary...")
        save_vocab(word_to_ix, ix_to_word, VOCAB_PATH)
        print(f"Vocabulary saved to {VOCAB_PATH}")
        print(f"Vocabulary size: {len(word_to_ix)}")