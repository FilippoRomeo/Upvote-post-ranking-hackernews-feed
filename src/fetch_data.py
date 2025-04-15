# src/fetch_data.py
import psycopg2
import pandas as pd

conn_str = {
    "host": "178.156.142.230",
    "port": 5432,
    "dbname": "hd64m1ki",
    "user": "sy91dhb",
    "password": "g5t49ao"
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
