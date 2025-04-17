# src/hn_test_regressor.py (updated)
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
    # 1. Load test data
    df = load_hn_data()
    
    # 2. Load normalization parameters (updated loading method)
    try:
        norm_params = torch.load('data/normalization_params.pt', weights_only=True)
    except:
        # Fallback to unsafe loading if needed (only if you trust the file source)
        norm_params = torch.load('data/normalization_params.pt', weights_only=False)
    
    score_mean = norm_params['mean']
    score_std = norm_params['std']
    
    # 3. Load model and embeddings
    embeddings, vocab = load_embeddings(
        'data/text8_embeddings.npy',
        'data/text8_vocab.json'
    )
    
    # Load model with proper device handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionModel(embeddings.shape[1]).to(device)
    
    # Try loading best model first, fallback to final if needed
    try:
        model.load_state_dict(torch.load('data/hn_regressor_best.pt', map_location=device))
    except FileNotFoundError:
        model.load_state_dict(torch.load('data/hn_regressor_final.pt', map_location=device))
    
    model.eval()
    
    # Rest of your test code remains the same...
    # [Include all the evaluation code from your original script]

if __name__ == '__main__':
    test()