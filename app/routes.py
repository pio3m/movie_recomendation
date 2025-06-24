from fastapi import APIRouter
from .models import database, movies
import pandas as pd
from fastapi import APIRouter, HTTPException
from app.utils import import_movies_from_kaggle
import os

router = APIRouter()



@router.post("/import")
async def trigger_import():
    csv_path = "data/movies_metadata.csv"

    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV file not found.")

    try:
        await import_movies_from_kaggle(csv_path)
        return {"status": "success", "message": "Movies imported successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
from .training_model import train_and_save_model

@router.post("/train-model")
async def train_model_endpoint():
    try:
        result = await train_and_save_model()
        return {"status": "trained", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fetch-data")
async def fetch_training_data_preview(limit: int = 10):
    query = """
    SELECT m.id, m.title, m.year, m.rating, ARRAY_AGG(g.name) AS genres
    FROM movies m
    JOIN movie_genres mg ON m.id = mg.movie_id
    JOIN genres g ON g.id = mg.genre_id
    GROUP BY m.id, m.title, m.year, m.rating
    ORDER BY m.rating DESC
    LIMIT :limit
    """
    rows = await database.fetch_all(query=query, values={"limit": limit})
    return rows

from fastapi import Request
import pickle
import numpy as np

from .schemas import RecommendRequest

@router.post("/recommend")
async def recommend_movies(payload: RecommendRequest):
    liked_titles = payload.liked_titles
    n = payload.n_recommendations

    if not liked_titles:
        raise HTTPException(status_code=400, detail="No liked_titles provided")

    # Załaduj model
    try:
        with open("models/model.pkl", "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model not trained yet.")

    model = data["model"]
    df = data["df"]
    features = data["features"]

    # Znajdź indeksy
    liked_indices = df[df["title"].isin(liked_titles)].index.tolist()

    if not liked_indices:
        raise HTTPException(status_code=404, detail="None of the liked titles found in dataset.")

    liked_vectors = features.iloc[liked_indices].values
    mean_vector = np.mean(liked_vectors, axis=0).reshape(1, -1)

    distances, indices = model.kneighbors(mean_vector, n_neighbors=n + len(liked_indices))
    indices = indices.flatten()

    recommended = (
        df.iloc[indices]
        .loc[~df.iloc[indices].index.isin(liked_indices)]
        .head(n)
        .to_dict(orient="records")
    )

    return recommended


