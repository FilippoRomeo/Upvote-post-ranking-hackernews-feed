# src/train_word2vec.py
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
import json

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
    "embedding_dim": 300,
    "batch_size": 1024,
    "lr": 0.005,
    "epochs": 5,
    "min_word_count": 5,
    "architecture": "CBOW",
    "patience": 5,
    "eval_words": ["king", "queen", "apple", "computer", "car", "human"]
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
        results = []
        for i, (score, idx) in enumerate(zip(top_values[1:], top_indices[1:]), 1):
            result = f"{i}. {ix_to_word[idx.item()]} ({score:.3f})"
            print(result)
            results.append((ix_to_word[idx.item()], float(score)))
    
    # Log to wandb in a format it can handle
    wandb.log({f"similar/{word}": wandb.Table(
        columns=["rank", "word", "score"],
        data=[[i+1, word, score] for i, (word, score) in enumerate(results)]
    )})

def train():
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(project="word2vec-cbow", config=config)
    
    # Load and preprocess text8 data
    print("Loading and tokenizing text8 data...")
    tokens = preprocess_text8(TEXT8_PATH)
    print(f"Total tokens: {len(tokens):,}")

    # Build vocabulary
    print("Building vocabulary...")
    word_to_ix, ix_to_word, vocab = build_vocab(tokens, min_count=config["min_word_count"])
    vocab_size = len(word_to_ix)
    print(f"Vocabulary size: {vocab_size:,}")
    
    # Save vocabulary in a safe format
    with open(VOCAB_PATH, 'w') as f:
        json.dump({
            'word_to_ix': word_to_ix,
            'ix_to_word': ix_to_word
        }, f)
    print(f"Vocabulary saved to {VOCAB_PATH}")

    # Create dataset and dataloader
    dataset = Word2VecDataset(tokens, word_to_ix, ix_to_word, context_size=config["context_size"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], 
                          shuffle=True, num_workers=4, pin_memory=True)

    # Initialize model
    model = CBOWModel(vocab_size, config["embedding_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], 
                                weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    loss_fn = nn.CrossEntropyLoss()
    
    # Watch model
    wandb.watch(model, log="all", log_freq=100)

    # Training setup
    print("\nStarting training...")
    start_time = time.time()
    best_loss = float('inf')
    no_improvement = 0

    for epoch in range(1, config["epochs"] + 1):
        epoch_loss = 0
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{config['epochs']}", unit="batch")

        for batch_idx, (context, target) in enumerate(progress_bar):
            context, target = context.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(context)
            loss = loss_fn(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log batch metrics
            if batch_idx % 100 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

        # Epoch evaluation
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        # Evaluate word similarities
        if epoch % 1 == 0:  # Evaluate every epoch
            print_similar_words(config["eval_words"][0], model, word_to_ix, ix_to_word)

        # Log epoch metrics
        wandb.log({
            "epoch_loss": avg_loss,
            "epoch_time": (time.time() - start_time) / 60
        })

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, BEST_MODEL_PATH)
        else:
            no_improvement += 1
            if no_improvement >= config["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'word_to_ix': word_to_ix,
        'ix_to_word': ix_to_word,
        'config': config
    }, MODEL_SAVE_PATH)

    # Save embeddings in numpy format
    embeddings = model.embeddings.weight.data.cpu().numpy()
    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"\nTraining completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Best model saved to {BEST_MODEL_PATH}")
    print(f"Embeddings saved to {EMBEDDINGS_PATH}")
    
    wandb.finish()

if __name__ == "__main__":
    train()