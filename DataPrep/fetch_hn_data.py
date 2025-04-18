# DataPrep/fetch_h_data.py
import psycopg2
import pandas as pd
import os

def fetch_hn_data(start_year=2015, end_year=2015, content_type='story', min_score=1):
    conn = psycopg2.connect("postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki")
    all_data = []

    try:
        with conn:
            with conn.cursor() as cur:
                for year in range(start_year, end_year + 1):
                    for month in range(1, 13):
                        table_name = f"hacker_news.items_by_month_{year:04d}_{month:02d}"
                        query = f"""
                            SELECT id, type, by, time, text, title, url, score, descendants
                            FROM {table_name}
                            WHERE type = %s
                        """
                        try:
                            cur.execute(query, (content_type,))
                            rows = cur.fetchall()

                            if rows:
                                columns = [desc[0] for desc in cur.description]
                                df = pd.DataFrame(rows, columns=columns)

                                # Drop rows with null/empty titles or null score
                                df = df.dropna(subset=['title', 'score'])
                                df = df[df['title'].str.strip() != '']
                                df = df[df['score'] >= min_score]

                                if not df.empty:
                                    all_data.append(df)
                                    print(f"✔ {len(df)} rows from {table_name}")
                        except Exception as e:
                            print(f"⚠ Skipped {table_name}: {e}")
    finally:
        conn.close()

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "..", "data/fetch_data", "hn_2010_stories.csv")
    df = fetch_hn_data(2008, 2010, 'story', min_score=1)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(df)} clean rows to hn_2010_stories.csv")
