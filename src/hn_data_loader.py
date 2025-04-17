#src/hn_data_loader.py
import psycopg2
import pandas as pd
import numpy as np
import json
from text8_tokenizer import simple_tokenizer  # assuming tokenize() is exposed


def load_hn_data():
    """
    Connect to the Hacker News Postgres DB and fetch titles and upvote scores.
    Returns a pandas DataFrame with columns ['title', 'upvote_score'].
    """
    conn = psycopg2.connect(
        dbname="hd64m1ki",
        user="sy91dhb",
        password="g5t49ao",
        host="178.156.142.230",
        port="5432"
    )
    df = pd.read_sql("SELECT title, upvote_score FROM hacker_news.items", conn)
    conn.close()
    return df


def load_embeddings(embeddings_path: str, vocab_path: str):
    """
    Load pre-trained CBOW embeddings and vocabulary mapping.
    embeddings_path: Path to .npy file (data/text8_embeddings.npy)
    vocab_path: Path to vocab JSON (data/text8_vocab.json)
    Returns: (embeddings: np.ndarray, vocab: dict[token -> idx])
    """
    embeddings = np.load(embeddings_path)
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    return embeddings, vocab