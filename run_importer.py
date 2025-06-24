import sys
import asyncio
from pathlib import Path

from app.models import metadata, engine
# Dodaj katalog nadrzędny do sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from app.utils import import_movies_from_kaggle

if __name__ == "__main__":
    metadata.create_all(engine)
    asyncio.run(import_movies_from_kaggle("data/movies_metadata.csv"))
