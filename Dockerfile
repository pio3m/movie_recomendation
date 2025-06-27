# Pobierz lekką wersję Pythona z uvicorn
FROM python:3.10-slim

# Zainstaluj podstawowe zależności
RUN apt-get update && apt-get install -y build-essential

# Ustaw katalog roboczy
WORKDIR /app

# Skopiuj pliki projektu do kontenera
COPY ./app /app

# Skopiuj plik requirements.txt (jeśli go masz) i zainstaluj zależności
# jeśli nie masz requirements.txt, użyj pip install w CMD
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# domyślny punkt startowy - uruchom aplikację na porcie PORT
CMD ["python", "run_render.py"]
