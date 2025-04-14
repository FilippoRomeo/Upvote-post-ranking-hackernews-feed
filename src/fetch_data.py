import psycopg2
import pandas as pd

# DB credentials
conn_str = {
    "host": "178.156.142.230",
    "port": 5432,
    "dbname": "hd64m1ki",
    "user": "sy91dhb",
    "password": "g5t49ao"
}

def fetch_data():
    conn = psycopg2.connect(**conn_str)
    query = "SELECT title, score FROM posts;"  # Adjust if table/column names are different
    df = pd.read_sql(query, conn)
    conn.close()
    return df

if __name__ == "__main__":
    df = fetch_data()
    print(df.head())
