import asyncio
from transformers import pipeline
from app.models import database, movies

# Model do klasyfikacji emocji
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=3
)

BATCH_SIZE = 10

async def save_emotions_to_db(movie_id: int, emotion_labels: list[str]):
    for label in emotion_labels:
        # Znajdź lub dodaj emocję
        emotion = await database.fetch_one(
            "SELECT id FROM emotions WHERE name = :name", {"name": label}
        )

        if emotion:
            emotion_id = emotion["id"]
        else:
            emotion_id = await database.execute(
                "INSERT INTO emotions (name) VALUES (:name) RETURNING id",
                {"name": label}
            )

        # Wstaw relację film–emocja
        await database.execute(
            """
            INSERT INTO movie_emotions (movie_id, emotion_id)
            VALUES (:movie_id, :emotion_id)
            ON CONFLICT DO NOTHING
            """,
            {"movie_id": movie_id, "emotion_id": emotion_id}
        )

async def classify_batch(batch):
    texts = [m["overview"][:512] if m["overview"] else "" for m in batch]
    results = emotion_model(texts)

    updates = []
    for i, movie in enumerate(batch):
        try:
            labels = [r["label"] for r in results[i] if r["score"] > 0.3] or ["Neutralna"]
        except Exception:
            labels = ["Neutralna"]

        updates.append({
            "movie_id": movie["movie_id"],
            "emotions": labels
        })
    return updates

async def run_worker():
    await database.connect()

    while True:
        jobs = await database.fetch_all(
            """
            SELECT j.movie_id, m.overview
            FROM emotion_jobs j
            JOIN movies m ON m.id = j.movie_id
            WHERE j.status = 'pending'
            ORDER BY j.updated_at ASC
            LIMIT :limit
            """,
            values={"limit": BATCH_SIZE}
        )

        if not jobs:
            print("💤 No jobs pending, sleeping...")
            await asyncio.sleep(5)
            continue

        # Oznacz jako 'processing'
        ids = [j["movie_id"] for j in jobs]
        await database.execute(
            "UPDATE emotion_jobs SET status = 'processing' WHERE movie_id = ANY(:ids)",
            values={"ids": ids}
        )

        updates = await classify_batch(jobs)

        for upd in updates:
            await save_emotions_to_db(upd["movie_id"], upd["emotions"])

            await database.execute(
                "UPDATE emotion_jobs SET status = 'done', updated_at = now() WHERE movie_id = :id",
                values={"id": upd["movie_id"]}
            )

        print(f"✅ Processed batch of {len(updates)} movies.")

    await database.disconnect()
