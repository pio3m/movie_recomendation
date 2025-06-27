from fastapi import FastAPI
from render_routes import router

app = FastAPI()
app.include_router(router)
