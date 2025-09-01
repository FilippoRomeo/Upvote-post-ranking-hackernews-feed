# 📈 Hacker News Upvote Prediction with Word2Vec

This project explores **predicting Hacker News post upvote scores** using **word embeddings trained from scratch** on the [`text8`](https://huggingface.co/datasets/ardMLX/text8) dataset with the **Word2Vec CBOW architecture**.

We:
1. Train custom word embeddings (CBOW).
2. Test embedding quality with semantic similarity queries.
3. Prepare a dataset of Hacker News post titles + upvote scores.
4. Build a regression model that predicts upvotes from title embeddings.

---

## ⚡ Quick Start

```bash
# 1. Create and activate environment
conda env create -f environment.yml -n search-engine
conda activate search-engine

# 2. Train Word2Vec embeddings on text8
python train_cbow.py

# 3. Test embeddings interactively
python test_cbow.py

# 4. Prepare Hacker News dataset from DB
python prepare_data.py

# (Next) Train regression model on hn_dataset.pt
