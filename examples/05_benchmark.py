"""
examples/05_benchmark.py
=========================
Benchmarking latency and throughput between pure Python execution vs. Rust-accelerated execution.
"""

import os
import time
from context_nexus import ContextNexus
from context_nexus.fallback import (
    compute_cosine_distances_fallback,
    fast_chunk_text_fallback,
    rrf_fusion_fallback,
    traverse_graph_fallback,
)
from context_nexus.hybrid_search import (
    compute_cosine_distances,
    fast_chunk_text,
    is_rust_available,
    rrf_fusion,
    traverse_graph,
)

def benchmark():
    print("=== ContextNexus Performance Benchmark ===")
    print(f"Rust Acceleration Active: {is_rust_available()}")

    # Prepare dummy data
    q_vec = [0.1 * i for i in range(128)]
    doc_vecs = [[0.05 * (i + j) for i in range(128)] for j in range(1000)]
    nodes = [f"node_{i}" for i in range(500)]
    edges = [(f"node_{i}", f"node_{i+1}") for i in range(499)]
    vec_ranks = [f"doc_{i}" for i in range(500)]
    graph_ranks = [f"doc_{499 - i}" for i in range(500)]

    # 1. Cosine Distance Benchmark
    t0 = time.perf_counter()
    for _ in range(100):
        compute_cosine_distances(q_vec, doc_vecs)
    t_rust_vec = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(100):
        compute_cosine_distances_fallback(q_vec, doc_vecs)
    t_py_vec = (time.perf_counter() - t0) * 1000

    print(f"\n[Cosine Distances (1000 docs x 100 runs)]")
    print(f"  Dispatcher Time : {t_rust_vec:.2f} ms")
    print(f"  Python Fallback : {t_py_vec:.2f} ms")

    # 2. RRF Fusion Benchmark
    t0 = time.perf_counter()
    for _ in range(1000):
        rrf_fusion(vec_ranks, graph_ranks, 60)
    t_rrf = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(1000):
        rrf_fusion_fallback(vec_ranks, graph_ranks, 60)
    t_py_rrf = (time.perf_counter() - t0) * 1000

    print(f"\n[RRF Fusion (500 items x 1000 runs)]")
    print(f"  Dispatcher Time : {t_rrf:.2f} ms")
    print(f"  Python Fallback : {t_py_rrf:.2f} ms")

if __name__ == "__main__":
    benchmark()
