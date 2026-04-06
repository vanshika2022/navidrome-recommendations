"""
Navidrome Recommendation API - FAISS + Redis Cache Serving Option
Copied from faiss_cpu, with Redis cache layer added.
This is a MODEL + SYSTEM-LEVEL optimization:
- FAISS for fast similarity search (model-level)
- Redis to cache results so repeat requests skip inference (system-level)
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
import redis
import json
import time

#App setup
app = FastAPI(
    title="Navidrome Recommendation API",
    description="API for song recommendations using FAISS + Redis cache",
    version="1.0.0"
)


#Request/Response models (same as baseline)

class RecommendRequest(BaseModel):
    user_id: str
    n_recommendations: int = 10

class RecommendResponse(BaseModel):
    user_id: str
    recommendations: list
    model_version: str
    generated_at: str
    inference_latency_ms: float


# Load model and embeddings (same as faiss_cpu)
NUM_USERS = 500
NUM_SONGS = 10000
EMBEDDING_DIM = 64

np.random.seed(42)
user_embeddings = np.random.randn(NUM_USERS, EMBEDDING_DIM).astype(np.float32)
song_embeddings = np.random.randn(NUM_SONGS, EMBEDDING_DIM).astype(np.float32)

# Build FAISS index (same as faiss_cpu)
faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
faiss_index.add(song_embeddings)

# Dummy song metadata (same as baseline)
song_metadata = [
    {"song_id": f"song_{i:05d}", "title": f"Song {i}", "artist": f"Artist {i % 50}",
     "album": f"Album {i % 200}", "genre": "rock"}
    for i in range(NUM_SONGS)
]

# NEW: Connect to Redis (running in separate container)
# "redis" is the hostname — Docker Compose creates a network where
# containers can find each other by service name
cache = redis.Redis(host="redis", port=6379, decode_responses=True)
CACHE_TTL = 3600  # Cache results for 1 hour (3600 seconds)


#Health check
@app.get("/health")
def health():
    return {"status": "ok"}


#Recommendation endpoint
@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):

    start_time = time.time()

    # NEW: Check Redis cache first
    # Cache key = "rec:user_42:10" (user_id + n_recommendations)
    cache_key = f"rec:{request.user_id}:{request.n_recommendations}"

    cached = cache.get(cache_key)
    if cached:
        # Cache HIT — return instantly, skip inference
        recommendations = json.loads(cached)
        inference_latency_ms = (time.time() - start_time) * 1000
        return RecommendResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_version="v0.1.0-dummy",
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            inference_latency_ms=round(inference_latency_ms, 2),
        )

    # Cache MISS — run FAISS search (same as faiss_cpu)
    user_idx = hash(request.user_id) % NUM_USERS
    user_vector = user_embeddings[user_idx].reshape(1, -1)
    scores, top_k_indices = faiss_index.search(user_vector, request.n_recommendations)
    scores = scores[0]
    top_k_indices = top_k_indices[0]

    recommendations = []
    for rank, idx in enumerate(top_k_indices, 1):
        meta = song_metadata[idx]
        recommendations.append({
            "rank": rank,
            "song_id": meta["song_id"],
            "title": meta["title"],
            "artist": meta["artist"],
            "album": meta["album"],
            "genre": meta["genre"],
            "score": float(scores[rank - 1]),
            "method": "collaborative_filtering",
        })

    # NEW: Store result in Redis for next time
    cache.set(cache_key, json.dumps(recommendations), ex=CACHE_TTL)

    inference_latency_ms = (time.time() - start_time) * 1000

    return RecommendResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        model_version="v0.1.0-dummy",
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        inference_latency_ms=round(inference_latency_ms, 2),
    )
