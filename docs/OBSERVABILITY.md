# Observability & Span Tracing Guide

`nexus-context` includes built-in span tracing and latency analytics to monitor multi-turn agent sessions in real time.

---

## 1. Monitoring HTTP Endpoints

When running `nexus-serve` on port 9000, you can query management endpoints:

```bash
# Check server health and active session count
curl http://localhost:9000/nexus/health

# Inspect detailed stats for a session
curl http://localhost:9000/nexus/session/{session_id}/stats
```

---

## 2. Response Headers

`nexus-serve` attaches observability headers to every completions response:

- `X-Nexus-Pipeline-Latency-MS`: Wall-clock pipeline execution overhead in milliseconds.
- `X-Nexus-Rust-Accelerated`: `1` if native Rust SIMD core was invoked, `0` if Python fallback was used.
- `X-Nexus-Session-ID`: Session identifier tracking graph topology state.

---

## 3. ObservabilityTracer Python SDK

```python
from context_nexus import ContextNexus

nexus = ContextNexus()
# Ingest and query...
context = nexus.get_context("Query text", session_id="s1")

# Inspect span metrics
print(context["trace_stats"])
# Output:
# {'total_latency_ms': 12.4, 'vector_search_ms': 4.1, 'graph_traversal_ms': 1.2, ...}
```
