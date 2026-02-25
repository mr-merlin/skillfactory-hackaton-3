FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY src/ ./src/
COPY evaluate.py analytics.py ./

# Данные монтируются через volume при запуске
# docker run -v $(pwd)/data:/app/data ...
ENV DATA_DIR=/app/data

EXPOSE 8000

# По умолчанию запускаем API сервер
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
