# Triton GPU Setup

## Step 1: Export ONNX model
```bash
pip3 install torch numpy
python3 export_onnx.py
```

## Step 2: Start Triton
```bash
docker compose up -d
```

## Step 3: Test
```bash
# Check model is loaded
curl http://localhost:8000/v2/models/song_recommender

# Send a request using perf_analyzer (like the lab)
docker run --rm --network host nvcr.io/nvidia/tritonserver:24.01-py3-sdk \
  perf_analyzer -u localhost:8000 -m song_recommender --input-data input.json -b 1 --concurrency-range 1
```
