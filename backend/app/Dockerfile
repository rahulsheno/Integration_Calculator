FROM python:3.11-slim

# System libraries needed by opencv-python / onnxruntime, which pix2text and
# pix2tex depend on transitively. Missing these is the single most common
# "works on my machine but not in a fresh environment" failure for this kind
# of OCR stack (import errors like "libGL.so.1: cannot open shared object
# file").
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# --reload picks up code changes when ./backend is bind-mounted over /app
# (see docker-compose.yml), so you don't need to rebuild the image while
# iterating on Python code.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
