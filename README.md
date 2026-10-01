# Hacker News Upvote Predictor

An NLP experiment that asks a deliberately narrow question: **how much of a Hacker News post's score can be predicted from its title alone?**

The pipeline trains CBOW word embeddings on `text8`, converts Hacker News titles into embedding indices, pools the title representation, and feeds it into a small neural regressor that predicts post score.

## Pipeline

```text
text8 corpus
    ↓
CBOW word embeddings
    ↓
Hacker News titles + scores
    ↓
tokenisation → vocabulary indices
    ↓
average-pooled title representation
    ↓
feed-forward regressor
    ↓
predicted score
```

## What is in the repository

- **CBOW training** in `src/train_word2vec.py`
- **Vocabulary and text preprocessing** under `src/`
- **Hacker News data preparation** under `DataPrep/`
- **Dataset assembly** in `prepare_data.py`
- **Score regressor** in `UpvotePredictionModel/train_model.py`
- **Prediction testing** in `UpvotePredictionModel/test_predictions.py`
- a Hacker News dataset snapshot under `data/fetch_data/`
- `text8.zip` for the embedding-training corpus

## Model design

### 1. Word representation

`src/train_word2vec.py` trains a CBOW model with PyTorch. The current configuration uses 300-dimensional embeddings and logs training diagnostics with Weights & Biases.

The resulting vocabulary and embeddings are written under `data/`.

### 2. Hacker News preparation

`prepare_data.py` reads Hacker News titles and scores, tokenises each title, maps known words into the CBOW vocabulary, removes titles with no usable tokens, and serialises the resulting dataset for PyTorch.

### 3. Score prediction

The prediction model consumes pooled title embeddings and trains a feed-forward neural network against Hacker News score using regression loss.

This is an experimental title-only model. It does not model timing, author reputation, linked domain, comments, front-page position, topic trends, or other factors that affect post performance.

## Quick start

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/FilippoRomeo/Upvote-post-ranking-hackernews-feed.git
cd Upvote-post-ranking-hackernews-feed
python -m pip install -r requirements.txt
```

Extract the included `text8` corpus into `data/`:

```bash
unzip -o text8.zip -d data
```

Train the word embeddings:

```bash
python src/train_word2vec.py
```

Prepare the Hacker News dataset:

```bash
python prepare_data.py
```

Train the score regressor:

```bash
python UpvotePredictionModel/train_model.py
```

Test predictions:

```bash
python UpvotePredictionModel/test_predictions.py
```

## Hacker News data

The repository currently includes a generated Hacker News CSV snapshot at:

```text
data/fetch_data/hn_2010_stories.csv
```

The source fetcher can regenerate data from a compatible PostgreSQL database. Database credentials are never stored in source code; provide the connection through `DATABASE_URL`:

```bash
export DATABASE_URL='postgresql://USER:PASSWORD@HOST:5432/DBNAME'
python DataPrep/fetch_hn_data.py
```

If `DATABASE_URL` is missing, the fetcher stops instead of falling back to an embedded credential.

## Repository structure

```text
.
├── DataPrep/
│   ├── fetch_hn_data.py
│   ├── save_dataset.py
│   ├── title_to_indices.py
│   └── tokenizer.py
├── UpvotePredictionModel/
│   ├── train_model.py
│   └── test_predictions.py
├── data/
│   └── fetch_data/
├── src/
│   ├── train_word2vec.py
│   ├── word2vec_dataset.py
│   ├── word2vec_model.py
│   ├── text8_tokenizer.py
│   └── test_cbow.py
├── prepare_data.py
├── requirements.txt
└── text8.zip
```

## Stack

`Python` `PyTorch` `CBOW / Word2Vec` `pandas` `PostgreSQL` `Weights & Biases`

## Scope and limitations

This repository is a learning and modelling experiment rather than a production ranking system. A post's eventual score is influenced by many variables that are intentionally excluded here, so prediction quality should be interpreted as evidence about the information contained in title text, not as a general model of Hacker News popularity.

## Security note

Database configuration is environment-based. Real credentials should stay in a local `.env` file or shell environment and must not be committed to Git.
