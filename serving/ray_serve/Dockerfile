# Ray Serve container for bonus extra credit
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY run_serve.py .

EXPOSE 8000

CMD ["python", "run_serve.py"]
