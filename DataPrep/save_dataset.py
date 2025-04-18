import pandas as pd
import torch
import os

def pad_sequences(sequences, padding_value=0):
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [padding_value] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded, dtype=torch.long)

def load_and_save_tensor(csv_path, out_path):
    df = pd.read_csv(csv_path)
    sequences = df['indices'].apply(eval).tolist()  # Convert from string to list
    targets = df['score'].tolist()

    X = pad_sequences(sequences)
    y = torch.tensor(targets, dtype=torch.float32)

    torch.save({'inputs': X, 'targets': y}, out_path)
    print(f"✅ Saved dataset to {out_path} with shape {X.shape} and {len(y)} targets")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    input_csv = os.path.join(base_dir, "..", "data", "hn_2010_indices.csv")
    output_pt = os.path.join(base_dir, "..", "data", "hn_dataset.pt")

    load_and_save_tensor(input_csv, output_pt)
