import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from app.models import database
from transformers import pipeline

emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=3)


async def fetch_training_data():
    query = """
    SELECT 
        m.id, 
        m.title, 
        m.year, 
        m.rating,
        m.overview,
        COALESCE(ARRAY_AGG(g.name), '{}') AS genres
    FROM movies m
    LEFT JOIN movie_genres mg ON m.id = mg.movie_id
    LEFT JOIN genres g ON g.id = mg.genre_id
    GROUP BY m.id, m.title, m.year, m.rating, m.overview
    """
    rows = await database.fetch_all(query)
    return pd.DataFrame([dict(row._mapping) for row in rows])



async def train_and_save_model(path="models/model.pkl"):
    df = await fetch_training_data()
    if df.empty:
        raise ValueError("Brak danych treningowych")

    df.info()
    df["genres"] = df["genres"].apply(lambda x: x if isinstance(x, list) else [])

    # Upewnij się, że wszystkie są listami
    df["genres"] = df["genres"].apply(lambda g: g if isinstance(g, list) else [])

    # Zamień puste na "Unknown" (opcjonalnie)
    df["genres"] = df["genres"].apply(lambda g: g if g else ["Unknown"])

    df["genres"] = df["genres"].apply(lambda g: [x for x in g if isinstance(x, str)])
    df["genres"] = df["genres"].apply(lambda g: g if g else ["Unknown"])
    
    print("📦 Przypisywanie emocji do overview...")
    df["emotions"] = df["overview"].apply(classify_emotions_from_text)
    print("✅ Emocje dodane. Przykład:")
    print(df[["title", "emotions"]].head(5))

    # Zakoduj
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(df["genres"])
    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

    # Skaluj year i rating
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[["year", "rating"]].fillna(0))
    scaled_df = pd.DataFrame(scaled_features, columns=["year_scaled", "rating_scaled"])

    # Połącz cechy
    features = pd.concat([genres_df, scaled_df], axis=1)


    # Ucz kNN
    model = NearestNeighbors(n_neighbors=10, metric="cosine")
    model.fit(features)

    # Zapisz model i dane
    with open(path, "wb") as f:
        pickle.dump({
            "model": model,
            "features": features,
            "df": df,
            "scaler": scaler,
            "mlb": mlb
        }, f)

    return {"status": "ok", "samples": len(df), "features": features.shape[1]}

def classify_emotions_from_text(text):
    if not isinstance(text, str) or not text.strip():
        return ["Neutralna"]
    try:
        results = emotion_model(text[:512])
        return [r["label"] for r in results if r["score"] > 0.3]
    except Exception:
        return ["Neutralna"]
