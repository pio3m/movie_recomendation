from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

router = APIRouter()

bert_model_path = "./app/bert_model"
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
bert_embeddings = np.load(os.path.join(bert_model_path, "overview_embeddings.npy"))
bert_df = pd.read_csv(os.path.join(bert_model_path, "movies.csv"))

# Load classic kNN model once at startup
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


class SentenceRequest(BaseModel):
    sentence: str
    n_recommendations: Optional[int] = 10

@router.post("/recommend-by-sentence")
async def recommend_by_sentence(payload: SentenceRequest):
    query_text = payload.sentence.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query sentence cannot be empty.")

    query_vec = bert_model.encode([query_text])

    similarities = cosine_similarity(query_vec, bert_embeddings).flatten()
    top_indices = similarities.argsort()[-payload.n_recommendations:][::-1]

    results = bert_df.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]

    return results[["title", "similarity"]].to_dict(orient="records")
