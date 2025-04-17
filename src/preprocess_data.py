# src/preprocess_data.py
import torch
import numpy as np
import pandas as pd
from word2vec_model import CBOWModel
from word2vec_dataset import Word2VecDataset

def load_hacker_news_data(database_connection):
    column_check_query = """
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_schema = 'hacker_news' AND table_name = 'items_by_year'
    """
    columns = pd.read_sql(column_check_query, database_connection)
    print("Available columns in hacker_news.items_by_year:")
    print(columns)
    
    query = "SELECT * FROM hacker_news.items_by_year LIMIT 1"
    sample = pd.read_sql(query, database_connection)
    print("\nSample row:")
    print(sample)
    
    query = "SELECT title, score FROM hacker_news.items_by_year"
    df = pd.read_sql(query, database_connection)
    return df

def get_avg_embedding(title_tokens, word2vec_model):
    embeddings = []
    for token in title_tokens:
        try:
            embeddings.append(word2vec_model.get_embedding(token))
        except KeyError:
            continue

    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)  # Default to zeros if no embeddings

def preprocess_data(df, word2vec_model):
    X = []
    y = []
    
    for index, row in df.iterrows():
        tokens = row['title'].split()
        avg_embedding = get_avg_embedding(tokens, word2vec_model)
        X.append(avg_embedding)
        y.append(row['score'])

    X = np.array(X)
    y = np.array(y)
    
    return X, y
