# src/tokenizer.py

from datasets import load_dataset
from collections import Counter
import re
import argparse

def clean_text(text):
    # Lowercase and remove non-alphabet characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return tokens

def limit_vocab(tokens, vocab_size):
    counter = Counter(tokens)
    most_common = counter.most_common(vocab_size)
    vocab = set(word for word, _ in most_common)
    filtered = [word for word in tokens if word in vocab]
    return filtered, vocab

def load_text8_tokens(vocab_size=None, path="text8"):
    print("Loading text8 file from local path...")
    with open(path, "r") as f:
        raw_text = f.read()

    print("Cleaning and tokenizing...")
    tokens = clean_text(raw_text)
    print(f"Original token count: {len(tokens)}")

    if vocab_size is not None and vocab_size > 0:
        tokens, vocab = limit_vocab(tokens, vocab_size)
        print(f"Reduced to top {vocab_size} words, token count now: {len(tokens)}")
    else:
        print("No vocabulary size limit applied.")

    return tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--path", type=str, default="text8")
    args = parser.parse_args()

    tokens = load_text8_tokens(vocab_size=args.vocab_size, path=args.path)

    print("Sample:", tokens[:20])

    # Save to ./data/tokens.json
    import os
    import json

    os.makedirs("data", exist_ok=True)

    with open("../data/tokens.json", "w") as f:
        json.dump(tokens, f)

    print("✅ Tokens saved to data/tokens.json")

