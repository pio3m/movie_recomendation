import pandas as pd
import ast

def load_kaggle_movies(path="data/movies_metadata.csv"):
    df = pd.read_csv(path, low_memory=False)

    # Wybierz tylko potrzebne kolumny
    df = df[["title", "genres", "release_date", "vote_average", "overview"]].dropna(
        subset=["title", "genres", "release_date", "vote_average"]
    )

    # Przekształć pole genres (JSON string → lista nazw)
    def parse_genres(genre_str):
        try:
            genres = ast.literal_eval(genre_str)
            return [g["name"] for g in genres if "name" in g]
        except Exception:
            return []

    df["genres"] = df["genres"].apply(parse_genres)

    # Przekształć datę na rok
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    df["overview"] = df["overview"].fillna("")

    # Przemianuj kolumnę vote_average na rating
    df = df.rename(columns={"vote_average": "rating"})

    # Finalne kolumny: tytuł, lista gatunków, rok, ocena, opis
    df = df[["title", "genres", "year", "rating", "overview"]]

    return df
