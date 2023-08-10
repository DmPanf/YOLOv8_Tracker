FROM python:3.8-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    sudo

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "yolo.py"]
