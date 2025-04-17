# src/train_regressor.py
import torch
from torch.utils.data import DataLoader, random_split
from hn_regression_dataset import HNDataset
from regressor_model import RegressionModel
from text8_tokenizer import simple_tokenizer
from hn_data_loader import load_hn_data, load_embeddings
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import wandb

def main():
    # Initialize wandb
    wandb.init(project="hn-score-regression", config={
        "learning_rate": 1e-4,
        "batch_size": 64,
        "epochs": 50,
        "weight_decay": 1e-5
    })
    
    # 1. Load and prepare data
    df = load_hn_data()
    
    # Data inspection
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(df['score'], bins=50)
    plt.title("Original Score Distribution")
    
    # Normalize scores
    score_mean = df['score'].mean()
    score_std = df['score'].std()
    df['score'] = (df['score'] - score_mean) / score_std
    wandb.config.update({"score_mean": score_mean, "score_std": score_std})
    
    plt.subplot(1, 2, 2)
    plt.hist(df['score'], bins=50)
    plt.title("Normalized Score Distribution")
    plt.tight_layout()
    wandb.log({"score_distribution": wandb.Image(plt)})
    plt.close()
    
    print(f"Score normalization: mean = {score_mean:.2f}, std = {score_std:.2f}")
    
    # 2. Load embeddings
    embeddings, vocab = load_embeddings(
        embeddings_path='data/text8_embeddings.npy',
        vocab_path='data/text8_vocab.json'
    )
    print(f"Loaded embeddings: {embeddings.shape}")
    
    # 3. Create datasets
    dataset = HNDataset(df, embeddings, vocab, simple_tokenizer)
    train_size = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=4)

    # 4. Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionModel(embeddings.shape[1]).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=wandb.config.learning_rate,
                                weight_decay=wandb.config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    wandb.watch(model, criterion, log="all", log_freq=10)

    # 5. Training loop with progress bars
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(wandb.config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{wandb.config.epochs} [Train]", unit="batch")
        
        for emb_batch, score_batch in train_bar:
            emb_batch, score_batch = emb_batch.to(device), score_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(emb_batch).squeeze()
            loss = criterion(preds, score_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{wandb.config.epochs} [Val]", unit="batch")
        
        with torch.no_grad():
            for emb_batch, score_batch in val_bar:
                emb_batch, score_batch = emb_batch.to(device), score_batch.to(device)
                preds = model(emb_batch).squeeze()
                loss = criterion(preds, score_batch)
                val_loss += loss.item()
                val_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Log metrics
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'data/hn_regressor_best.pt')
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= 5:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Save final model and artifacts
    torch.save(model.state_dict(), 'data/hn_regressor_final.pt')
    torch.save({'mean': score_mean, 'std': score_std}, 'data/normalization_params.pt')
    
    # Log artifacts to wandb
    artifact = wandb.Artifact('hn-regressor', type='model')
    artifact.add_file('data/hn_regressor_best.pt')
    artifact.add_file('data/hn_regressor_final.pt')
    artifact.add_file('data/normalization_params.pt')
    wandb.log_artifact(artifact)
    
    wandb.finish()
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()