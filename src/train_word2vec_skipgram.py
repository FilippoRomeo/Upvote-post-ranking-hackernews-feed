# train_word2vec_skipgram.py (similar to CBOW but with Skip-gram)
from word2vec_model import SkipGramModel

# Replace CBOWModel with SkipGramModel
model = SkipGramModel(vocab_size, EMBEDDING_DIM).to(device)

# Modify dataset to return (target, context) instead of (context, target)
dataset = Word2VecDataset(tokens, word_to_ix, ix_to_word, context_size=CONTEXT_SIZE, mode="skipgram")