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
def most_similar(word, topn=15):
    if word not in word_to_ix:
        print(f"❌ '{word}' not in vocabulary.")
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
