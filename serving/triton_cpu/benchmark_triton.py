"""
Benchmark script for Triton GRU4Rec encoder.

Two modes:
  1. Direct Triton benchmark — sends prefix_item_idxs to Triton encoder,
     measures just the GRU encoding time (no dot product).
  2. End-to-end benchmark — sends full recommendation requests to the
     FastAPI wrapper (which calls Triton internally).

Usage on Chameleon:
  # Direct Triton (encoder only):
  python benchmark_triton.py --mode triton --url triton_server:8000

  # End-to-end (wrapper → Triton → dot product → response):
  python benchmark_triton.py --mode e2e --url localhost:8080
"""

import argparse
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def benchmark_triton_direct(url, num_requests, concurrency):
    """Benchmark Triton encoder directly via HTTP inference API."""
    import tritonclient.http as httpclient

    client = httpclient.InferenceServerClient(url=url)
    prefix = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)

    # Warmup
    for _ in range(10):
        inp = httpclient.InferInput("prefix_item_idxs", prefix.shape, "INT64")
        inp.set_data_from_numpy(prefix)
        out = httpclient.InferRequestedOutput("session_repr")
        client.infer(model_name="gru4rec_encoder", inputs=[inp], outputs=[out])

    def send():
        inp = httpclient.InferInput("prefix_item_idxs", prefix.shape, "INT64")
        inp.set_data_from_numpy(prefix)
        out = httpclient.InferRequestedOutput("session_repr")
        t0 = time.time()
        client.infer(model_name="gru4rec_encoder", inputs=[inp], outputs=[out])
        return time.time() - t0

    times = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(send) for _ in range(num_requests)]
        for f in as_completed(futures):
            times.append(f.result())

    return np.array(times)


def benchmark_e2e(url, num_requests, concurrency):
    """Benchmark the full recommendation pipeline via FastAPI wrapper."""
    import requests as req

    payload = {
        "session_id": "bench_001",
        "user_id": 44361,
        "user_idx": 1,
        "prefix_item_idxs": [1, 2, 3, 4, 5],
        "playratios": [0.1, 1.0, 1.0, 1.0, 0.18],
        "exclude_item_idxs": [1, 2, 3, 4, 5],
        "top_n": 20,
    }
    endpoint = f"http://{url}/recommend"

    # Warmup
    for _ in range(10):
        req.post(endpoint, json=payload)

    def send():
        t0 = time.time()
        r = req.post(endpoint, json=payload)
        elapsed = time.time() - t0
        return elapsed, r.status_code

    times = []
    errors = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(send) for _ in range(num_requests)]
        for f in as_completed(futures):
            elapsed, status = f.result()
            if status == 200:
                times.append(elapsed)
            else:
                errors += 1

    if errors:
        print(f"  Errors: {errors}/{num_requests}")
    return np.array(times)


def print_results(times, concurrency, num_requests):
    times_ms = times * 1000
    total_time = times.sum()
    print(f"\n--- Concurrency: {concurrency} ({num_requests} requests) ---")
    print(f"Median latency:   {np.median(times_ms):.2f} ms")
    print(f"p95 latency:      {np.percentile(times_ms, 95):.2f} ms")
    print(f"p99 latency:      {np.percentile(times_ms, 99):.2f} ms")
    print(f"Throughput:       {num_requests / total_time * concurrency:.1f} req/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["triton", "e2e"], default="e2e")
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--requests", type=int, default=200)
    args = parser.parse_args()

    bench_fn = benchmark_triton_direct if args.mode == "triton" else benchmark_e2e
    print(f"Benchmarking ({args.mode}) against {args.url}")

    for c in [1, 4, 8, 16]:
        times = bench_fn(args.url, args.requests, c)
        print_results(times, c, args.requests)
