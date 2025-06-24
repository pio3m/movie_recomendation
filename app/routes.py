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