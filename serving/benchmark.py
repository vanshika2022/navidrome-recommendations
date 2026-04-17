"""
Benchmark script for recommendation serving options.
This script sends requests to the FastAPI recommendation endpoint
and measures latency & throughput at different concurrency levels.

Adapted from lab's benchmarking approach:
- Lab used perf_analyzer with --concurrency-range 1, 8, 16
- We use Python threads to simulate concurrent users
"""

import requests
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

FASTAPI_URL = "http://localhost:8000/recommend"
payload = {
    "session_id": "bench_001",
    "user_id": 44361,
    "user_idx": 1,
    "prefix_track_ids": [4698874, 838286, 2588097, 455834, 2460503],
    "prefix_item_idxs": [1, 2, 3, 4, 5],
    "playratios": [0.1, 1.0, 1.0, 1.0, 0.18],
    "exclude_item_idxs": [1, 2, 3, 4, 5],
    "top_n": 20,
}


def send_request():
    """Send one request and return the response time."""
    start_time = time.time()
    response = requests.post(FASTAPI_URL, json=payload)
    elapsed = time.time() - start_time
    return elapsed, response.status_code


def run_benchmark(num_requests, concurrency):
    """
    Send num_requests total, with `concurrency` requests at a time.
    Like the lab's perf_analyzer --concurrency-range flag.
    """
    inference_times = []
    errors = 0

    # ThreadPoolExecutor sends multiple requests at the same time
    # concurrency=1 means one at a time (like our old benchmark)
    # concurrency=10 means 10 users hitting the server simultaneously
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request) for _ in range(num_requests)]
        for future in as_completed(futures):
            elapsed, status = future.result()
            if status == 200:
                inference_times.append(elapsed)
            else:
                errors += 1

    inference_times = np.array(inference_times)
    median_time = np.median(inference_times)
    percentile_95 = np.percentile(inference_times, 95)
    percentile_99 = np.percentile(inference_times, 99)
    throughput = num_requests / inference_times.sum() * concurrency

    print(f"\n--- Concurrency: {concurrency} ({num_requests} requests) ---")
    print(f"Median inference time:  {1000 * median_time:.4f} ms")
    print(f"95th percentile:        {1000 * percentile_95:.4f} ms")
    print(f"99th percentile:        {1000 * percentile_99:.4f} ms")
    print(f"Throughput:             {throughput:.2f} requests/sec")
    print(f"Error rate:             {errors / num_requests * 100:.1f}%")


# Test at different concurrency levels (like lab's --concurrency-range 1, 8, 16)
print(f"Benchmarking {FASTAPI_URL}")
for c in [1, 5, 10, 20]:
    run_benchmark(num_requests=100, concurrency=c)
