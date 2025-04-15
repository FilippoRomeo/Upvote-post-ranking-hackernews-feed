# src/fetch_data.py
import psycopg2
import pandas as pd

conn_str = {
    "host": "REDACTED_HOST",
    "port": 5432,
    "dbname": "REDACTED_DBNAME",
    "user": "REDACTED_USER",
    "password": "REDACTED_PASSWORD"
}

def fetch_data():
    conn = psycopg2.connect(**conn_str)
    query = "SELECT title, score FROM posts;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

if __name__ == "__main__":
    df = fetch_data()
    df.to_csv("data/hn_posts.csv", index=False)
