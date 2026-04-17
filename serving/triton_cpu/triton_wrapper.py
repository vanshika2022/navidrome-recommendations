"""
FastAPI wrapper that sits in front of Triton Inference Server.

Architecture (from Ch7 §7.4.1 — separation of concerns):
    Client  →  This wrapper (app logic)  →  Triton (GRU encoder)

This service handles:
    - Input validation, OOV filtering
    - Sending prefix_item_idxs to Triton for GRU session encoding
    - Dot product: session_repr @ all_item_emb.T → scores
    - Top-k selection and vocab translation (idx → track_id)
    - Prometheus metrics

Triton handles:
    - GRU forward pass (ONNX backend, CUDA execution provider on GPU)
    - Dynamic batching (configured in config.pbtxt)
    - Model instance scaling across GPUs

This split is analogous to the lab's ONNX backend setup, where
preprocessing/postprocessing moved to Flask while Triton ran the model.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
import torch
import tritonclient.http as httpclient
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

# ─── configuration ───────────────────────────────────────────────────────
TRITON_URL = os.environ.get("TRITON_URL", "localhost:8000")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "../../artifacts/vocabs.pkl")
MODEL_ARTIFACT_PATH = os.environ.get("MODEL_ARTIFACT_PATH", "../../artifacts/best_gru4rec.pt")
MODEL_NAME = "gru4rec_encoder"
MODEL_VERSION = os.environ.get("MODEL_VERSION", "best_gru4rec-triton")
MAX_PREFIX_LEN = 200
MAX_TOP_N = 100

# ─── logging ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("serving.triton_wrapper")


def _hash_user(user_id) -> str:
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:12]


# ─── Prometheus metrics ──────────────────────────────────────────────────
REQUESTS = Counter("recommend_requests_total", "Total requests by status", ["status"])
LATENCY = Histogram("recommend_latency_seconds", "End-to-end latency",
                     buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0))
TRITON_LATENCY = Histogram("triton_encoder_latency_seconds", "Triton GRU encoder latency",
                            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1))
OOV_ITEMS = Counter("recommend_oov_items_total", "OOV items dropped")


# ─── request / response schemas ─────────────────────────────────────────
class RecommendRequest(BaseModel):
    session_id: str
    user_id: int | str
    user_idx: int = 0
    request_timestamp: str | None = None
    prefix_track_ids: list[int | str] = Field(default_factory=list)
    prefix_item_idxs: list[int] = Field(min_length=1, max_length=MAX_PREFIX_LEN)
    playratios: list[float] = Field(default_factory=list)
    exclude_item_idxs: list[int] = Field(default_factory=list)
    top_n: int = Field(default=20, ge=1, le=MAX_TOP_N)


class Recommendation(BaseModel):
    rank: int
    item_idx: int
    track_id: str
    score: float


class RecommendResponse(BaseModel):
    session_id: str
    request_id: str
    recommendations: list[Recommendation]
    model_version: str
    generated_at: str
    inference_latency_ms: float
    triton_latency_ms: float
    oov_count: int


# ─── app state ───────────────────────────────────────────────────────────
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load vocab
    log.info(f"Loading vocab from {VOCAB_PATH} ...")
    with open(VOCAB_PATH, "rb") as f:
        item2idx, user2idx = pickle.load(f)
    idx2item = {idx: str(track_id) for track_id, idx in item2idx.items()}
    state["idx2item"] = idx2item
    log.info(f"Vocab loaded: {len(item2idx)} items")

    # Precompute item embedding matrix from the .pt file
    # (only the embeddings — we don't need the full model since Triton runs the GRU)
    log.info(f"Loading item embeddings from {MODEL_ARTIFACT_PATH} ...")
    sd = torch.load(MODEL_ARTIFACT_PATH, map_location="cpu", weights_only=True)
    all_item_emb = sd["item_emb.weight"][1:].numpy()  # strip padding row 0
    state["all_item_emb"] = all_item_emb
    log.info(f"Item embeddings loaded: {all_item_emb.shape}")

    # Connect to Triton
    client = httpclient.InferenceServerClient(url=TRITON_URL)
    state["triton_client"] = client
    log.info(f"Triton client connected to {TRITON_URL}")

    yield
    log.info("Shutting down.")


app = FastAPI(title="Navidrome Recommendation API (Triton)", version="1.0.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health():
    ready = state.get("triton_client") is not None
    return {"status": "ok" if ready else "not_ready"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest, http_request: Request):
    request_id = http_request.headers.get("x-request-id") or str(uuid.uuid4())
    t_start = time.time()

    log.info(
        f"request_id={request_id} session={request.session_id} "
        f"user_hash={_hash_user(request.user_id)} prefix_len={len(request.prefix_item_idxs)}"
    )

    try:
        with LATENCY.time():
            idx2item = state["idx2item"]
            all_item_emb = state["all_item_emb"]
            triton_client = state["triton_client"]

            # Filter OOV
            clean_prefix = [i for i in request.prefix_item_idxs if i in idx2item]
            oov_count = len(request.prefix_item_idxs) - len(clean_prefix)
            if oov_count:
                OOV_ITEMS.inc(oov_count)
            if not clean_prefix:
                REQUESTS.labels(status="400").inc()
                raise HTTPException(status_code=400, detail="All prefix items are OOV.")

            # Send to Triton for GRU encoding
            # Input: truncated to real length (no padding) — see export_onnx.py comments
            input_data = np.array([clean_prefix], dtype=np.int64)
            triton_input = httpclient.InferInput("prefix_item_idxs", input_data.shape, "INT64")
            triton_input.set_data_from_numpy(input_data)
            triton_output = httpclient.InferRequestedOutput("session_repr")

            t_triton = time.time()
            result = triton_client.infer(
                model_name=MODEL_NAME,
                inputs=[triton_input],
                outputs=[triton_output],
            )
            triton_ms = (time.time() - t_triton) * 1000
            TRITON_LATENCY.observe(triton_ms / 1000)

            session_repr = result.as_numpy("session_repr")  # (1, 128)

            # Dot product against all items + top-k (runs on CPU in this service)
            scores = session_repr @ all_item_emb.T  # (1, 745352)
            scores = scores[0]

            # Exclude already-heard items
            for item_idx in request.exclude_item_idxs:
                if 1 <= item_idx <= len(scores):
                    scores[item_idx - 1] = float("-inf")

            # Top-k
            top_k = min(request.top_n, len(scores))
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

            recs = [
                Recommendation(
                    rank=rank,
                    item_idx=int(idx) + 1,  # back to 1-based
                    track_id=idx2item.get(int(idx) + 1, f"unknown_{idx+1}"),
                    score=float(scores[idx]),
                )
                for rank, idx in enumerate(top_indices, start=1)
            ]

        REQUESTS.labels(status="200").inc()
        latency_ms = (time.time() - t_start) * 1000
        return RecommendResponse(
            session_id=request.session_id,
            request_id=request_id,
            recommendations=recs,
            model_version=MODEL_VERSION,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            inference_latency_ms=round(latency_ms, 2),
            triton_latency_ms=round(triton_ms, 2),
            oov_count=oov_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        REQUESTS.labels(status="500").inc()
        log.exception(f"request_id={request_id} error: {e}")
        raise HTTPException(status_code=500, detail="internal error")
