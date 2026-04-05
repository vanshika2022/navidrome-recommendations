# Triton CPU Setup

## Step 1: Export ONNX model
```bash
# Build the export container (one-time)
docker build -f Dockerfile.export -t triton-export .

# Run it to generate the model file
docker run --rm -v $(pwd)/model_repository:/workspace/model_repository triton-export
```

## Step 2: Start Triton server
```bash
docker compose up -d
```

## Step 3: Wait for Triton to load the model, then check
```bash
sleep 20
curl http://localhost:8000/v2/models/song_recommender
```

## Step 4: Run benchmark
```bash
docker run --rm --network host -v $(pwd):/app python:3.11-slim bash -c "pip install requests numpy -q && python /app/benchmark_triton.py"
```

## Stop Triton
```bash
docker compose down
```
