# 🔍 Hacker News Upvote Predictor

This project builds a regression model to **predict the upvote score** of Hacker News posts based on their **titles**. It uses **Word2Vec-style embeddings** (trained with the CBOW architecture on the [text8 dataset](https://huggingface.co/datasets/ardMLX/text8)) and then feeds those embeddings into a regression model.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing Word Embeddings](#testing-word-embeddings)
- [Upvote Prediction Model](#upvote-prediction-model)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## 🔎 Overview

1. **Train embeddings**
   - Train a **CBOW Word2Vec model** on the cleaned Wikipedia corpus **text8**.
   - Produce a vocabulary (`text8_vocab.json`) and word embeddings (`text8_embeddings.npy`).

2. **Prepare Hacker News data**
   - Connect to a PostgreSQL DB with titles and upvote scores.
   - Tokenize titles → Convert tokens to indices using CBOW vocab → Save dataset as tensors.

3. **Predict upvotes**
   - Pool embeddings (average) from each title.
   - Feed into a regression model (simple feed-forward NN).
   - Train with **MSE loss** to predict Hacker News scores.

---

## 📂 Project Structure

```
project-root/
├─ data/
│  ├─ text8                         # raw text8 corpus (downloaded)
│  ├─ text8_vocab.json              # built vocab (CBOW)
│  ├─ text8_embeddings.npy          # trained embeddings (CBOW)
│  └─ hn_dataset.pt                 # processed HN dataset (titles → indices, scores)
├─ src/
│  ├─ word2vec_model.py             # CBOW model
│  ├─ word2vec_dataset.py           # CBOW dataset logic
│  ├─ text8_tokenizer.py            # text8 preprocessing + vocab utils
│  └─ test_cbow.py                  # semantic explorer for embeddings
├─ DataPrep/
│  ├─ fetch_hn_data.py              # DB fetch: titles + scores
│  ├─ tokenize_titles.py            # title normalization & tokenization
│  ├─ title_to_indices.py           # tokens → indices using CBOW vocab
│  └─ save_dataset.py               # persist processed dataset
├─ prepare_data.py                  # pipeline entry to build hn_dataset.pt
├─ train_cbow.py                    # train CBOW on text8
├─ train_regressor.py               # train upvote regressor on HN data
├─ environment.yml
├─ requirements.txt
└─ README.md
```

> **Note:** Filenames can vary; the commands below assume these defaults.

---

## ⚙️ Setup

### 1) Clone & enter the repo

```bash
git clone <your-repo-url>
cd search-engine
```

### 2) Create Conda environment

```bash
conda env create -f environment.yml -n search-engine
conda activate search-engine
```

**Or manually install requirements:**

```bash
conda activate ai-lab   # or your preferred env
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3) Verify installation

```bash
python --version
pip list
```

---

## 🗄️ Data Preparation

**Database connection**

The project connects to a PostgreSQL DB:

```
postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki
```

**Run the pipeline**

```bash
python prepare_data.py
```

This will:

- Fetch **titles + scores**
- Tokenize titles
- Map tokens → indices with **CBOW vocab**
- Save processed dataset at: `data/hn_dataset.pt`

**Edge cases handled:**

- Tokens not in vocab → filtered out
- Empty titles after filtering → skipped
- Extremely short/long titles → retained, but pooled during training

---

## 🧠 Training

### 1) Train Word2Vec (CBOW)

```bash
python train_cbow.py
```

This saves:

- `data/text8_vocab.json` (vocab mapping)
- `data/text8_embeddings.npy` (trained embeddings)
- Model checkpoints (e.g., `data/text8_cbow_model.pt`, `data/best_model.pt`)

### 2) Train the Upvote Regressor

```bash
python train_regressor.py
```

This uses the Hacker News dataset (`hn_dataset.pt`) and **CBOW embeddings** to predict scores.

---

## 🧪 Testing Word Embeddings

Explore the semantic space of words:

```bash
python src/test_cbow.py
```

**Example session**

```
Enter a word (or 'exit'): car

🔗 Words related to 'car':
  cars            | similarity: 0.5110
  vehicle         | similarity: 0.3860
  automobile      | similarity: 0.3740
  motorcycle      | similarity: 0.3350
  ...
```

---

## 📈 Upvote Prediction Model

A simple feed-forward regressor:

```python
import torch.nn as nn

class UpvoteRegressor(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: predicted score
        )

    def forward(self, x):
        return self.model(x)
```

- **Loss:** `MSELoss`
- **Optimizer:** `Adam` (e.g., `lr=1e-3`)
- **Input:** average-pooled title embeddings (`[batch, embedding_dim]`)

---

## 🛠️ Troubleshooting

### Pip / Conda issues

If you see:

```
error: externally-managed-environment
```

➡️ Use Conda:

```bash
conda activate ai-lab
python -m pip install -r requirements.txt
```

➡️ Or recreate env:

```bash
conda env create -f environment.yml -n search-engine
conda activate search-engine
```

➡️ If `pip` is broken inside Conda:

```bash
conda install pip --force-reinstall
```

### Empty Titles

Some HN titles may contain no valid tokens after preprocessing (e.g., all non-alphanumerics). These are **skipped** during dataset creation.

### macOS + Homebrew Python (PEP 668)

Homebrew’s Python may block global `pip` installs. Prefer **Conda envs** or a `venv`. If you must, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 📚 References

- **Word2Vec Paper** (Mikolov et al., 2013): *Efficient Estimation of Word Representations in Vector Space*
- **text8 dataset** (Hugging Face): https://huggingface.co/datasets/ardMLX/text8  
  Background: http://mattmahoney.net/dc/textdata.html
- **Hacker News**: https://news.ycombinator.com/
- **PEP 668** (Externally Managed Environments): https://peps.python.org/pep-0668/

---

🚀 **Happy hacking!** If you have questions or want to extend this (e.g., try Skip-gram, different pooling, or switch to RNNs/Transformers), open an issue or PR.
