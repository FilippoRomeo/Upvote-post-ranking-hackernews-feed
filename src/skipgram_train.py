import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from word2vec_dataset import Word2VecDataset  # Assume you've implemented dataset class
from word2vec_model import SkipGramModel  # Implement this model based on SkipGram
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
# Hyperparameters and paths for Skipgram model
SKIPGRAM_MODEL_SAVE_PATH = os.path.join(DATA_DIR, "text8_skipgram_model.pt")
SKIPGRAM_BEST_MODEL_PATH = os.path.join(DATA_DIR, "best_skipgram_model.pt")
SKIPGRAM_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "text8_skipgram_embeddings.npy")
SKIPGRAM_VOCAB_PATH = os.path.join(DATA_DIR, "text8_skipgram_vocab.json")


# Hyperparameters
config = {
    "context_size": 5,
    "embedding_dim": 300,  # Increased for better word representations
    "batch_size": 1024,    # Increased for more stable training
    "lr": 0.005,           # Higher learning rate for faster convergence
    "epochs": 5,           # Too many trainings
    "min_word_count": 5,   # Lower count to keep more words
    "architecture": "SkipGram",
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
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(project="word2vec-skipgram", config=config, 
              settings=wandb.Settings(start_method="thread"))
    
    # Load and preprocess text8 data
    print("Loading and tokenizing text8 data...")
    tokens = preprocess_text8(TEXT8_PATH)
    print(f"Total tokens: {len(tokens):,}")

    # Build vocabulary
    print("Building vocabulary...")
    word_to_ix, ix_to_word, vocab = build_vocab(tokens, min_count=config["min_word_count"])
    vocab_size = len(word_to_ix)
    print(f"Vocabulary size: {vocab_size:,}")
    
    # Save vocabulary
    save_vocab_json(word_to_ix, ix_to_word, VOCAB_PATH)
    print(f"Vocabulary saved to {VOCAB_PATH}")

    # Create dataset and dataloader
    dataset = Word2VecDataset(tokens, word_to_ix, ix_to_word, context_size=config["context_size"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], 
                          shuffle=True, num_workers=4, pin_memory=True)

    # Initialize model
    model = SkipGramModel(vocab_size, config["embedding_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], 
                                weight_decay=1e-5, amsgrad=True)
    
    # Warmup LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1., step / config["warmup_steps"]))
    
    # ReduceLR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Watch model
    wandb.watch(model, loss_fn, log="parameters", log_freq=500)

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
            
            # Reshape target for CrossEntropyLoss (should be [batch_size, vocab_size])
            target = target.view(-1)  # Flatten target to a 1D tensor

            optimizer.zero_grad(set_to_none=True)
            output = model(context)
            loss = loss_fn(output, target)
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config["accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log batch metrics
            if batch_idx % 500 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch_progress": epoch + batch_idx/len(dataloader),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

        # Update learning rate scheduler
        lr_scheduler.step(epoch * len(dataloader) + batch_idx)
        
        # Epoch evaluation
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        # Evaluate word similarities
        if epoch % 2 == 0 or epoch == 1:
            evaluate_similarity(model, word_to_ix, ix_to_word, config["eval_words"])
            print_similar_words(config["eval_words"][0], model, word_to_ix, ix_to_word)

        # Log epoch metrics
        wandb.log({
            "epoch_loss": avg_loss,
            "epoch": epoch,
            "epoch_time": (time.time() - start_time) / 60
        })

        # Early stopping and model saving
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "epoch": epoch
            }, SKIPGRAM_BEST_MODEL_PATH)
        else:
            no_improvement += 1
            if no_improvement >= config["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Final save
    torch.save({
        "model_state_dict": model.state_dict(),
        "word_to_ix": word_to_ix,
        "ix_to_word": ix_to_word,
        "config": config,
        "optimizer_state_dict": optimizer.state_dict()
    }, SKIPGRAM_MODEL_SAVE_PATH)

    # Save embeddings
    embeddings = model.embeddings.weight.data.cpu().numpy()
    np.save(SKIPGRAM_EMBEDDINGS_PATH, embeddings)

    # Log artifacts
    artifact = wandb.Artifact('trained-model', type='model')
    artifact.add_file(SKIPGRAM_MODEL_SAVE_PATH)
    artifact.add_file(SKIPGRAM_BEST_MODEL_PATH)
    artifact.add_file(SKIPGRAM_EMBEDDINGS_PATH)
    artifact.add_file(VOCAB_PATH)
    wandb.log_artifact(artifact)

    print(f"\nTraining completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Model saved to {SKIPGRAM_MODEL_SAVE_PATH}")
    print(f"Best model saved to {SKIPGRAM_BEST_MODEL_PATH}")
    print(f"Embeddings saved to {SKIPGRAM_EMBEDDINGS_PATH}")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    train()
