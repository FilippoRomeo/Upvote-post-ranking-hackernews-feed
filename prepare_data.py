import os
import pandas as pd
import torch
import json

from DataPrep.tokenizer import tokenize_all_titles
from DataPrep.title_to_indices import titles_to_indices
from DataPrep.save_dataset import save_dataset

# === Paths ===
BASE_DIR = os.path.dirname(__file__)
data_dir = os.path.join(BASE_DIR, "data")
csv_path = os.path.join(data_dir, "hn_2010_stories.csv")
vocab_path = os.path.join(data_dir, "text8_vocab.json")
output_path = os.path.join(data_dir, "hn_dataset.pt")

# === Step 1: Load raw Hacker News data ===
print("📥 Loading raw data...")
df = pd.read_csv(csv_path)
print(f"✔ Loaded {len(df)} rows")

# === Step 2: Tokenize titles ===
print("🧠 Tokenizing titles...")
df['tokens'] = tokenize_all_titles(df)
print(f"✔ Tokenized {len(df)} titles")

# === Step 3: Load CBOW vocab ===
print("📖 Loading CBOW vocabulary...")
with open(vocab_path, 'r') as f:
    vocab = json.load(f)
word_to_ix = vocab['word_to_ix']

# === Step 4: Convert tokens to indices ===
print("🔢 Converting tokens to indices...")
df['indices'] = titles_to_indices(df['tokens'], word_to_ix)
df = df[df['indices'].map(lambda x: len(x) > 0)]  # Drop rows with no valid tokens
print(f"✔ Retained {len(df)} rows after removing empty titles")

# === Step 5: Save dataset as tensor ===
print("💾 Saving dataset to PyTorch .pt file...")
save_dataset(df['indices'].tolist(), df['score'].tolist(), output_path)
print(f"✅ Dataset saved to {output_path}")
