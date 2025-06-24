from fastapi import FastAPI
from app.routes import router
from app.models import database, metadata, engine

app = FastAPI()  # <- TO JEST KLUCZ

metadata.create_all(engine)

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

app.include_router(router)  # <- TO PODPIĘCIE ENDPOINTÓW
