# src/test_cbow.py
import torch
import numpy as np
import json
import torch.nn.functional as F

# Load vocab
with open('data/tokens.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    word_to_ix = data["word_to_ix"]
    ix_to_word = {int(k): v for k, v in data["ix_to_word"].items()}

# Load embeddings
embeddings = np.load('data/text8_embeddings.npy')
embeddings = torch.tensor(embeddings)

# Cosine similarity
def most_similar(word, topn=15):
    if word not in word_to_ix:
        print(f"❌ '{word}' not in vocabulary.")
        return []

    idx = word_to_ix[word]
    target_emb = embeddings[idx].unsqueeze(0)
    similarities = F.cosine_similarity(target_emb, embeddings)

    # Get top (topn + 20) in case some are missing from ix_to_word
    top_indices = torch.topk(similarities, topn + 30).indices
    results = []

    for i in top_indices:
        i = i.item()
        word_i = ix_to_word.get(i, None)
        if word_i and word_i != word:
            results.append((word_i, similarities[i].item()))
        if len(results) == topn:
            break

    if len(results) < topn:
        print(f"⚠️ Only found {len(results)} valid similar words for '{word}'")
    return results

# CLI
if __name__ == '__main__':
    print("🔍 CBOW Semantic Field Explorer")
    print("Type a word to explore related terms. Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter a word (or 'exit'): ").strip().lower()
        if user_input == 'exit':
            break

        topn_input = input("How many similar words to show? [default: 15]: ").strip()
        try:
            topn = int(topn_input) if topn_input else 15
        except ValueError:
            print("⚠️ Invalid number. Showing default 15 results.")
            topn = 15

        similar_words = most_similar(user_input, topn=topn)
        if similar_words:
            print(f"\n🔗 Words related to '{user_input}':\n")
            for w, score in similar_words:
                print(f"  {w:<15} | similarity: {score:.4f}")
            print()
