from fastapi import FastAPI
from app.render_routes import router

app = FastAPI()
app.include_router(router)
