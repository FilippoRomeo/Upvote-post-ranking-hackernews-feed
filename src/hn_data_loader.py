# src/hn_data_loader.py
import pandas as pd
import numpy as np
import json
import os
import psycopg2
from datetime import datetime
from datasets import Dataset, concatenate_datasets,load_from_disk
from huggingface_hub import login

def load_hn_data():
    """Load Hacker News data from local disk"""
    data_path = './hacker_news_titles'
    if os.path.exists(data_path):
        dataset = load_from_disk(data_path)
        return dataset.to_pandas()
    else:
        raise FileNotFoundError(f"Hacker News data not found at {data_path}")

def load_embeddings(embeddings_path, vocab_path):
    """Load pre-trained embeddings and vocabulary"""
    embeddings = np.load(embeddings_path)
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    return embeddings, vocab

# ---- CONFIGURATION ---- #
DB_CONN_STRING = 'postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki'
HF_DATASET_NAME = 'besartshyti/hacker_news_score'
LOCAL_SAVE_PATH = './hacker_news_titles'
MAX_ITERATIONS = 300
BATCH_SIZE = 5000

# ---- DATABASE CONNECTION ---- #
conn = psycopg2.connect(DB_CONN_STRING)
cursor = conn.cursor()
last_time = datetime.now()
STORE = []

# ---- DATA EXTRACTION LOOP ---- #
for i in range(MAX_ITERATIONS):
    query = '''
        SELECT title, score, time, url
        FROM hacker_news.items
        WHERE type = 'story'
        AND score IS NOT NULL
        AND title IS NOT NULL
        AND time < %s
        ORDER BY time DESC
        LIMIT %s
    '''

    df = pd.read_sql_query(query, conn, params=(last_time, BATCH_SIZE))
    if df.empty:
        print(f"[{i:03d}] No more rows to fetch.")
        break

    STORE.append(Dataset.from_pandas(df))
    last_time = df['time'].iloc[-1]
    print(f"[{i:03d}] Fetched {len(df)} rows. New last_time = {last_time}")

# ---- CLOSE CONNECTION ---- #
conn.close()

# ---- CONCATENATE AND SAVE ---- #
print("Merging all batches...")
data = concatenate_datasets(STORE)
data.save_to_disk(LOCAL_SAVE_PATH)

# ---- UPLOAD TO HUGGINGFACE HUB ---- #
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    print("Logging into Hugging Face...")
    login(hf_token)
    print("Pushing dataset to hub...")
    data.push_to_hub(HF_DATASET_NAME)
    print("Upload complete.")
else:
    print("⚠️ Hugging Face token not found in environment variable 'HF_TOKEN'. Skipping upload.")
