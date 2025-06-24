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
