import psycopg2
import pandas as pd

# DB credentials
conn_str = {
    "host": "REDACTED_HOST",
    "port": 5432,
    "dbname": "REDACTED_DBNAME",
    "user": "REDACTED_USER",
    "password": "REDACTED_PASSWORD"
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
