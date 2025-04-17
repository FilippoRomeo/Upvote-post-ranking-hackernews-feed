# src/preprocess_data.py
import torch
import numpy as np
import pandas as pd
from word2vec_model import CBOWModel
from word2vec_dataset import Word2VecDataset

def load_hacker_news_data(database_connection):
    # Your database connection setup to fetch the data
    query = "SELECT title, upvote_score FROM hacker_news"
    df = pd.read_sql(query, database_connection)
    return df

def get_avg_embedding(title_tokens, word2vec_model):
    embeddings = [word2vec_model[token] for token in title_tokens if token in word2vec_model]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)  # Default to zeros if no embeddings

def preprocess_data(df, word2vec_model):
    # Tokenize the titles and get their average embeddings
    X = []
    y = []
    
    for index, row in df.iterrows():
        tokens = row['title'].split()  # Simple space-based tokenization
        avg_embedding = get_avg_embedding(tokens, word2vec_model)
        X.append(avg_embedding)
        y.append(row['upvote_score'])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y
