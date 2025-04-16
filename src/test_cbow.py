# test_cbow.py
import torch
import torch.nn.functional as F
from your_model_module import CBOWModel  # Replace with your actual model file
import pickle

# Load vocab mappings
with open('word_to_ix.pkl', 'rb') as f:
    word_to_ix = pickle.load(f)
with open('ix_to_word.pkl', 'rb') as f:
    ix_to_word = pickle.load(f)

# Load trained model
embedding_dim = 100  # or whatever you used
vocab_size = len(word_to_ix)
model = CBOWModel(vocab_size, embedding_dim)
model.load_state_dict(torch.load('cbow_model.pt'))
model.eval()

# Get embedding for a word
def get_embedding(word):
    idx = word_to_ix.get(word, word_to_ix.get('<UNK>'))
    return model.embeddings(torch.tensor(idx)).detach()

# Find top-N similar words based on cosine similarity
def most_similar(word, topn=5):
    word_emb = get_embedding(word)
    all_embeddings = model.embeddings.weight.detach()
    
    similarities = F.cosine_similarity(word_emb.unsqueeze(0), all_embeddings)
    topk = torch.topk(similarities, topn + 1)  # +1 to skip the word itself
    top_words = [(ix_to_word[i.item()], similarities[i].item()) for i in topk.indices if ix_to_word[i.item()] != word]
    return top_words[:topn]

# CLI-like interface
if __name__ == '__main__':
    while True:
        word = input("Enter a word (or 'exit' to quit): ").strip().lower()
        if word == 'exit':
            break
        if word not in word_to_ix:
            print(f"'{word}' not in vocabulary.")
            continue
        print(f"Top similar words to '{word}':")
        for sim_word, score in most_similar(word):
            print(f"  {sim_word:<10} | similarity: {score:.4f}")

