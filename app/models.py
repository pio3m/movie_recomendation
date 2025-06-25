from sqlalchemy import Table, Column, Integer, String, Float, MetaData, create_engine, ForeignKey
from databases import Database

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/movies"

database = Database(DATABASE_URL)
metadata = MetaData()

movies = Table(
    "movies",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("title", String, nullable=False),
    Column("year", Integer),
    Column("rating", Float),
    Column("overview", String),
    
)

genres = Table(
    "genres",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, unique=True, nullable=False)
)

movie_genres = Table(
    "movie_genres",
    metadata,
    Column("movie_id", ForeignKey("movies.id")),
    Column("genre_id", ForeignKey("genres.id"))
)


emotions = Table(
    "emotions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, unique=True, nullable=False)
)

movie_emotions = Table(
    "movie_emotions",
    metadata,
    Column("movie_id", ForeignKey("movies.id"), primary_key=True),
    Column("emotion_id", ForeignKey("emotions.id"), primary_key=True)
)

engine = create_engine(DATABASE_URL)