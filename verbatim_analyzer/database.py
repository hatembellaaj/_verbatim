import os
import sqlite3
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

DATA_DIR = "verbatim_analyzer/data"
DB_PATH = os.path.join(DATA_DIR, "analyses.db")
os.makedirs(DATA_DIR, exist_ok=True)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                date TEXT,
                model TEXT,
                themes TEXT,
                nb_clusters INTEGER,
                filename TEXT,
                result_path TEXT
            )
        """)
        conn.commit()

def save_analysis_result(analysis_id, df_result):
    result_path = os.path.join(DATA_DIR, f"{analysis_id}.csv")
    df_result.to_csv(result_path, index=False)
    return result_path

def save_analysis_metadata(metadata, df_result):
    logging.info("📦 Metadata :")
    logging.info(metadata)

    result_path = save_analysis_result(metadata["id"], df_result)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analyses (id, date, model, themes, nb_clusters, filename, result_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata["id"],
            metadata["date"],
            metadata["model"],
            metadata["themes"],
            metadata["nb_clusters"],
            metadata["filename"],
            result_path
        ))
        conn.commit()



def get_analyses_history():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT id, date, filename FROM analyses ORDER BY date DESC", conn)
        df["label"] = df["date"] + " - " + df["filename"]
        return df[["label", "id"]]

def load_analysis_result(history_df, selected_label):
    row = history_df[history_df['label'] == selected_label].iloc[0]
    analysis_id = row['id']
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT result_path FROM analyses WHERE id = ?", (analysis_id,))
        result = cursor.fetchone()
        if result:
            result_path = result[0]
            return pd.read_csv(result_path)
    return pd.DataFrame()
