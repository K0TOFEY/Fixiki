FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY functions.py charts.py dashboard.py ./
COPY templates ./templates
COPY static ./static
ENV PORT=8081
EXPOSE 8081
CMD ["sh", "-c", "gunicorn dashboard:app --workers 1 --threads 4 --bind 0.0.0.0:${PORT}"]
