import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from word2vec_dataset import Word2VecDataset
from word2vec_model import CBOWModel
from text8_tokenizer import preprocess_text8, build_vocab, save_vocab_json
import time
from tqdm import tqdm
import os
import numpy as np  
import torch.nn.functional as F
import wandb
from collections import defaultdict

# Configuration
DATA_DIR = "data"
TEXT8_PATH = os.path.join(DATA_DIR, "text8")
VOCAB_PATH = os.path.join(DATA_DIR, "text8_vocab.json")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "text8_cbow_model.pt")
BEST_MODEL_PATH = os.path.join(DATA_DIR, "best_model.pt")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "text8_embeddings.npy")

# Hyperparameters
config = {
    "context_size": 5,
    "embedding_dim": 300,  # Increased for better word representations
    "batch_size": 1024,    # Increased for more stable training
    "lr": 0.005,           # Higher learning rate for faster convergence
    "epochs": 5,           # Too many trainings
    "min_word_count": 5,   # Lower count to keep more words
    "architecture": "CBOW",
    "patience": 5,         # More tolerant early stopping
    "eval_words": ["king", "queen", "apple", "computer", "car", "human"],  # Test words
    "accumulation_steps": 2,  # Gradient accumulation steps
    "warmup_steps": 4000,     # Warmup steps for learning rate
}

def print_similar_words(word, model, word_to_ix, ix_to_word, top_n=10):
    """Print most similar words with their similarity scores"""
    if word not in word_to_ix:
        print(f"'{word}' not in vocabulary.")
        return

    model.eval()
    with torch.no_grad():
        embeddings = model.embeddings.weight.data.cpu()
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        word_idx = word_to_ix[word]
        word_vec = norm_embeddings[word_idx].unsqueeze(0)
        
        similarities = F.cosine_similarity(word_vec, norm_embeddings)
        top_values, top_indices = similarities.topk(top_n + 1)  # +1 to exclude self
        
        print(f"\nTop {top_n} words similar to '{word}':")
        for i, (score, idx) in enumerate(zip(top_values[1:], top_indices[1:]), 1):
            print(f"{i}. {ix_to_word[idx.item()]} ({score:.3f})")

def evaluate_similarity(model, word_to_ix, ix_to_word, words):
    """Evaluate and log similarity for test words"""
    results = defaultdict(dict)
    model.eval()
    with torch.no_grad():
        embeddings = model.embeddings.weight.data.cpu()
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        for word in words:
            if word in word_to_ix:
                idx = word_to_ix[word]
                word_vec = norm_embeddings[idx].unsqueeze(0)
                similarities = F.cosine_similarity(word_vec, norm_embeddings)
                top_values, top_indices = similarities.topk(6)  # Top 5 + self
                
                for val, idx in zip(top_values[1:], top_indices[1:]):  # Skip self
                    results[word][ix_to_word[idx.item()]] = val.item()
    
    # Log to wandb
    wandb.log({"word_similarities": wandb.Table(
        columns=["word"] + [f"top_{i}" for i in range(1, 6)],
        data=[[w] + list(results[w].keys()) for w in words if w in results]
    )})

def train():
    torch.manual_seed(42)
    device = get_device()

    if config.get("use_wandb", True):
        wandb.init(project="word2vec-cbow", config=config, settings=wandb.Settings(start_method="thread"))

    tokens = preprocess_text8(TEXT8_PATH)
    word_to_ix, ix_to_word, _ = build_vocab(tokens, config["min_word_count"])
    save_vocab_json(word_to_ix, ix_to_word, VOCAB_PATH)

    dataset = Word2VecDataset(tokens, word_to_ix, ix_to_word, config["context_size"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    model = CBOWModel(len(word_to_ix), config["embedding_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1., step / config["warmup_steps"]))
    loss_fn = nn.CrossEntropyLoss()

    best_loss, no_improve = float("inf"), 0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss = 0

        for i, (ctx, tgt) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            ctx, tgt = ctx.to(device), tgt.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(ctx), tgt)
            loss.backward()

            if (i + 1) % config["accumulation_steps"] == 0:
                optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        lr_scheduler.step(epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            save_model(BEST_MODEL_PATH, model, optimizer, {"epoch": epoch, "loss": best_loss})
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                break

        if epoch % 2 == 0:
            evaluate_similarity(model, word_to_ix, ix_to_word, config["eval_words"])

    save_model(MODEL_SAVE_PATH, model, optimizer)
    np.save(EMBEDDINGS_PATH, model.embeddings.weight.data.cpu().numpy())

if __name__ == "__main__":
    train()
