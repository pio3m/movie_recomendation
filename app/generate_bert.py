# app/generate_bert_embeddings.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def generate_embeddings():
    print("📥 Wczytywanie danych...")
    df = pd.read_csv("./bert_model/movies.csv")

    if "overview" not in df.columns:
        raise ValueError("Brakuje kolumny 'overview' w movies.csv")

    print(f"📦 Koduję {len(df)} overview za pomocą BERT...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['overview'].fillna("").tolist(), show_progress_bar=True)

    np.save("./bert_model/overview_embeddings.npy", embeddings)
    print("✅ Zapisano overview_embeddings.npy")

if __name__ == "__main__":
    generate_embeddings()
