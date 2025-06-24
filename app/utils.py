from .models import movies, genres, movie_genres, database
from .loader import load_kaggle_movies

async def import_movies_from_kaggle(csv_path="data/movies_metadata.csv"):
    df = load_kaggle_movies(csv_path)
    df["overview"] = df["overview"].fillna("")
    
    await database.connect()
    genre_cache = {}  # zapamiętane genre_name → genre_id

    for _, row in df.iterrows():
        # Wstaw film do tabeli movies (bez genres)
        movie_data = {
            "title": row["title"],
            "year": row["year"],
            "rating": row["rating"],
            "overview": row["overview"]
        }
        movie_id = await database.execute(movies.insert().values(**movie_data))

        # Obsługa wielu gatunków
        for genre_name in row["genres"]:
            # Jeżeli gatunek już był wstawiony, nie rób SELECT-a
            if genre_name not in genre_cache:
                # Czy gatunek już istnieje?
                select_query = genres.select().where(genres.c.name == genre_name)
                existing = await database.fetch_one(select_query)

                if existing:
                    genre_id = existing["id"]
                else:
                    genre_id = await database.execute(genres.insert().values(name=genre_name))

                genre_cache[genre_name] = genre_id
            else:
                genre_id = genre_cache[genre_name]

            # Wstaw relację movie-genre
            await database.execute(movie_genres.insert().values(movie_id=movie_id, genre_id=genre_id))

    await database.disconnect()
