# src/train_regressor.py
import torch
from torch.utils.data import DataLoader, random_split
from hn_regression_dataset import HNDataset
from regressor_model import RegressionModel
from text8_tokenizer import simple_tokenizer
from hn_data_loader import load_hn_data, load_embeddings
import matplotlib.pyplot as plt

def main():
    # 1. Load Hacker News data
    df = load_hn_data()
    
    # Inspect score distribution before normalization
    plt.hist(df['score'], bins=50)
    plt.title("Original Score Distribution")
    plt.show()
    
    # Normalize scores
    score_mean = df['score'].mean()
    score_std = df['score'].std()
    df['score'] = (df['score'] - score_mean) / score_std
    
    print(f"Score normalization: mean = {score_mean:.2f}, std = {score_std:.2f}")
    
    # 2. Load pre-trained embeddings + vocab
    embeddings, vocab = load_embeddings(
        embeddings_path='data/text8_embeddings.npy',
        vocab_path='data/text8_vocab.json'
    )
    
    # Check embedding stats
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding stats - mean: {embeddings.mean():.4f}, std: {embeddings.std():.4f}")

    # 3. Prepare dataset
    dataset = HNDataset(df, embeddings, vocab, simple_tokenizer)

    # 4. Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # 5. Initialize model, loss, optimizer
    input_dim = embeddings.shape[1]
    model = RegressionModel(input_dim)
    criterion = torch.nn.MSELoss()
    # In train_regressor.py:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # 6. Training loop
    num_epochs = 50  # Increased epochs
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for emb_batch, score_batch in train_loader:
            optimizer.zero_grad()
            preds = model(emb_batch).squeeze()
            loss = criterion(preds, score_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for emb_batch, score_batch in val_loader:
                preds = model(emb_batch).squeeze()
                loss = criterion(preds, score_batch)
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        
        scheduler.step(avg_val)  # Update learning rate
        
        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'data/hn_regressor_model_best.pt')

        print(f"Epoch {epoch+1}/{num_epochs} — Train MSE: {avg_train:.4f}, Val MSE: {avg_val:.4f}")
        
        # Early stopping if no improvement
        if scheduler.num_bad_epochs > 5:
            print("Early stopping triggered")
            break

    # Plot training curves
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Training Curves')
    plt.show()

    # 7. Save the final trained regressor
    torch.save(model.state_dict(), 'data/hn_regressor_model_final.pt')
    print(f"Best validation MSE: {best_val_loss:.4f}")

    # Save normalization parameters
    torch.save({'mean': score_mean, 'std': score_std}, 'data/normalization_params.pt')

if __name__ == '__main__':
    main()