# src/test_regressor.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from hn_regression_dataset import HNDataset
from regressor_model import RegressionModel
from text8_tokenizer import simple_tokenizer
from hn_data_loader import load_hn_data, load_embeddings
import matplotlib.pyplot as plt
from tqdm import tqdm

def test():
    # 1. Load test data (using same loader as training)
    df = load_hn_data()
    
    # Load normalization parameters
    norm_params = torch.load('data/normalization_params.pt')
    score_mean = norm_params['mean']
    score_std = norm_params['std']
    
    # 2. Load model and embeddings
    embeddings, vocab = load_embeddings(
        'data/text8_embeddings.npy',
        'data/text8_vocab.json'
    )
    
    model = RegressionModel(embeddings.shape[1])
    model.load_state_dict(torch.load('data/hn_regressor_best.pt'))
    model.eval()
    
    # 3. Prepare test dataset
    test_dataset = HNDataset(df, embeddings, vocab, simple_tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 4. Run predictions
    predictions = []
    true_scores = []
    
    with torch.no_grad():
        for emb_batch, score_batch in tqdm(test_loader, desc="Testing"):
            preds = model(emb_batch).squeeze()
            predictions.extend(preds.numpy())
            true_scores.extend(score_batch.numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_scores = np.array(true_scores)
    
    # 5. Denormalize predictions
    pred_denorm = (predictions * score_std) + score_mean
    true_denorm = (true_scores * score_std) + score_mean
    
    # 6. Calculate metrics
    mse = np.mean((pred_denorm - true_denorm)**2)
    mae = np.mean(np.abs(pred_denorm - true_denorm))
    r2 = 1 - np.sum((true_denorm - pred_denorm)**2) / np.sum((true_denorm - np.mean(true_denorm))**2)
    
    print(f"\nTest Results:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    # 7. Visualization
    plt.figure(figsize=(12, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(true_denorm, pred_denorm, alpha=0.3)
    plt.plot([true_denorm.min(), true_denorm.max()], 
             [true_denorm.min(), true_denorm.max()], 'r--')
    plt.xlabel("True Scores")
    plt.ylabel("Predicted Scores")
    plt.title("True vs Predicted Scores")
    
    # Error distribution
    plt.subplot(1, 2, 2)
    errors = pred_denorm - true_denorm
    plt.hist(errors, bins=50)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    
    plt.tight_layout()
    plt.savefig('results/prediction_analysis.png')
    plt.show()
    
    # 8. Example predictions
    print("\nSample Predictions:")
    sample_indices = np.random.choice(len(df), 5, replace=False)
    for idx in sample_indices:
        title = df.iloc[idx]['title']
        true = true_denorm[idx]
        pred = pred_denorm[idx]
        print(f"\nTitle: {title}")
        print(f"True Score: {true:.1f} | Predicted: {pred:.1f} | Error: {pred-true:.1f}")

if __name__ == '__main__':
    test()