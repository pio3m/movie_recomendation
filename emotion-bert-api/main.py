from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load BERT embeddings & model on startup
bert_model_path = "./bert_model"
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
bert_embeddings = np.load(os.path.join(bert_model_path, "overview_embeddings.npy"))
bert_df = pd.read_csv(os.path.join(bert_model_path, "movies.csv"))

class SentenceRequest(BaseModel):
    sentence: str
    n_recommendations: Optional[int] = 10

@app.post("/recommend-by-sentence")
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
