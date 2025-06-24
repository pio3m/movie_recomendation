from fastapi import FastAPI
from .routes import router  # bo routes.py jest w app/

app = FastAPI()



from app.models import database

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


app.include_router(router)
