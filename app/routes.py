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
