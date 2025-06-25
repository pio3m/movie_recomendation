from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pickle
import numpy as np

router = APIRouter()

# Load model once at startup
with open("models/model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
df = model_data["df"]
features = model_data["features"]

class RecommendRequest(BaseModel):
    liked_titles: List[str]
    n_recommendations: Optional[int] = 10

@router.post("/recommend")
async def recommend_movies(payload: RecommendRequest):
    liked_titles = payload.liked_titles
    n = payload.n_recommendations

    liked_indices = df[df["title"].isin(liked_titles)].index.tolist()
    if not liked_indices:
        raise HTTPException(status_code=404, detail="Liked titles not found")

    mean_vector = np.mean(features.iloc[liked_indices].values, axis=0).reshape(1, -1)
    distances, indices = model.kneighbors(mean_vector, n_neighbors=n + len(liked_indices))

    recommended = (
        df.iloc[indices[0]]
        .loc[~df.iloc[indices[0]].index.isin(liked_indices)]
        .head(n)
        .to_dict(orient="records")
    )

    return recommended


class EmotionRequest(BaseModel):
    emotions: List[str]
    n_recommendations: Optional[int] = 5

MOCK_FILMS = [
    {
        "title": "Up",
        "year": 2009,
        "rating": 8.2,
        "emotions": ["joy", "love"],
        "feature_similarity": 0.91
    },
    {
        "title": "Inside Out",
        "year": 2015,
        "rating": 8.2,
        "emotions": ["joy", "sadness"],
        "feature_similarity": 0.87
    },
    {
        "title": "Soul",
        "year": 2020,
        "rating": 8.1,
        "emotions": ["joy", "hope"],
        "feature_similarity": 0.89
    },
    {
        "title": "Titanic",
        "year": 1997,
        "rating": 7.8,
        "emotions": ["love", "sadness"],
        "feature_similarity": 0.84
    },
    {
        "title": "Joker",
        "year": 2019,
        "rating": 8.4,
        "emotions": ["anger", "sadness"],
        "feature_similarity": 0.81
    },
]

from typing import List, Optional
import random

@router.post("/recommend-by-emotion")
async def recommend_by_emotion(payload: EmotionRequest):
    requested = set(payload.emotions)
    matching = []

    for film in MOCK_FILMS:
        matched = list(requested & set(film["emotions"]))
        if matched:
            matching.append({
                "title": film["title"],
                "year": film["year"],
                "rating": film["rating"],
                "matched_emotions": matched,
                "score": round(0.8 + 0.03 * len(matched), 2),
                "feature_similarity": film["feature_similarity"]
            })

    random.shuffle(matching)
    return matching[:payload.n_recommendations]