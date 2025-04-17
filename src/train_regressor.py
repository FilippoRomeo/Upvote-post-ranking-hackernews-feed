# src/train_regressor.py
import torch
from torch.utils.data import DataLoader, random_split
from hn_data_loader import load_hn_data, load_embeddings, tokenize
from hn_regression_dataset import HNDataset
from regressor_model import RegressionModel


def main():
    # 1. Load Hacker News data
    df = load_hn_data()

    # 2. Load pre-trained embeddinds + vocab
    embeddings, vocab = load_embeddings(
        embeddings_path='data/text8_embeddings.npy',
        vocab_path='data/text8_vocab.json'
    )

    # 3. Prepare dataset
    dataset = HNDataset(df, embeddings, vocab, tokenize)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 6. Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for emb_batch, score_batch in train_loader:
            optimizer.zero_grad()
            preds = model(emb_batch).squeeze()
            loss = criterion(preds, score_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for emb_batch, score_batch in val_loader:
                preds = model(emb_batch).squeeze()
                loss = criterion(preds, score_batch)
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} — Train MSE: {avg_train:.4f}, Val MSE: {avg_val:.4f}")

    # 7. Save the trained regressor
    torch.save(model.state_dict(), 'data/hn_regressor_model.pt')


if __name__ == '__main__':
    main()