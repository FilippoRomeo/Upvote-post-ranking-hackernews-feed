import torch
import numpy as np
import pickle
import torch.nn.functional as F

# Load vocab
with open('data/text8_vocab.pkl', 'rb') as f:
    word_to_ix, ix_to_word = pickle.load(f)

# Load embeddings
embeddings = np.load('data/text8_embeddings.npy')  # shape: (vocab_size, embedding_dim)
embeddings = torch.tensor(embeddings)

# Cosine similarity
def most_similar(word, topn=5):
    if word not in word_to_ix:
        print(f"'{word}' not in vocabulary.")
        return []

    idx = word_to_ix[word]
    target_emb = embeddings[idx].unsqueeze(0)  # shape: (1, dim)
    similarities = F.cosine_similarity(target_emb, embeddings)

    top_indices = torch.topk(similarities, topn + 1).indices  # +1 to skip self
    results = []
    for i in top_indices:
        i = i.item()
        word_i = ix_to_word.get(i)
        if word_i and word_i != word:
            results.append((word_i, similarities[i].item()))
        if len(results) == topn:
            break
    return results

# CLI
if __name__ == '__main__':
    print("🔍 CBOW Similarity Explorer (type 'exit' to quit)")
    while True:
        word = input("Enter a word: ").strip().lower()
        if word == 'exit':
            break
        similar_words = most_similar(word)
        if similar_words:
            print(f"\nWords most similar to '{word}':")
            for w, score in similar_words:
                print(f"  {w:<12} | similarity: {score:.4f}")
            print()
