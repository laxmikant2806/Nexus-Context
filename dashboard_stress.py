"""
dashboard_stress.py
====================
Fires multiple concurrent sessions with multi-turn conversations,
tool-call messages, and varied content through nexus-serve to
populate every metric panel on the real-time dashboard.

Usage:
    python dashboard_stress.py
"""
from __future__ import annotations

import json
import time
from openai import OpenAI

BASE_URL = "http://localhost:9000/v1"
MODEL = "qwen2.5-coder:7b"

client = OpenAI(api_key="nexus-local", base_url=BASE_URL)


def run_session(session_id: str, turns: list[dict]) -> None:
    """Run a multi-turn conversation under a specific session ID."""
    history = []
    print(f"\n[Session {session_id}] Starting ({len(turns)} turns)")

    for i, turn in enumerate(turns):
        history.append({"role": "user", "content": turn["user"]})

        # Inject tool-call message on configured turns to trigger Feature D
        if turn.get("tool_response"):
            history.append({
                "role": "tool",
                "content": turn["tool_response"],
                "name": "execute_python",
            })

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior Python engineer. "
                            "Answer concisely and include code where relevant."
                        ),
                    },
                    *history,
                ],
                max_tokens=120,
                extra_headers={"X-Session-ID": session_id},
            )
            reply = response.choices[0].message.content or ""
            history.append({"role": "assistant", "content": reply})
            tokens = response.usage.total_tokens if response.usage else "?"
            print(f"  Turn {i+1}: {tokens} tokens | {reply[:60].strip()}...")
        except Exception as e:
            print(f"  Turn {i+1}: ERROR — {e}")

        time.sleep(0.3)


# ---------------------------------------------------------------
# Session 1 — Multi-turn coding task (populates latency, tokens,
#             KV cache, memory pool over several turns)
# ---------------------------------------------------------------
session1_turns = [
    {
        "user": "Define connection parameters for a PostgreSQL database named prod_db on host db.internal port 5432.",
    },
    {
        "user": "Now write a Python function connect_db() that uses those parameters with psycopg2.",
    },
    {
        "user": (
            "The connect_db function needs to handle connection timeouts. "
            "Add a timeout=30 parameter and retry logic with 3 attempts."
        ),
        # Large tool response to trigger Feature D compression
        "tool_response": json.dumps({
            "status": "executed",
            "output": "Connection successful",
            "db_info": {
                "host": "db.internal",
                "port": 5432,
                "database": "prod_db",
                "version": "PostgreSQL 15.2",
                "tables": [f"table_{i}" for i in range(80)],  # large array triggers compression
                "indexes": [f"idx_{i}" for i in range(60)],
                "schemas": ["public", "analytics", "staging", "archive"],
            },
            "execution_time_ms": 142,
            "rows_affected": 0,
            "warnings": [f"Warning {i}: deprecated usage pattern" for i in range(20)],
        })
    },
    {
        "user": "Great. Now add connection pooling using psycopg2.pool.ThreadedConnectionPool with min=2, max=10.",
    },
    {
        "user": "What are the best practices for closing connections in this pool setup?",
    },
]

# ---------------------------------------------------------------
# Session 2 — Different topic, builds graph topology by
#             referencing prior definitions
# ---------------------------------------------------------------
session2_turns = [
    {
        "user": "Design a FastAPI endpoint POST /users that accepts a Pydantic model with name, email, and role fields.",
    },
    {
        "user": "Add validation: email must be a valid format, role must be one of ['admin', 'viewer', 'editor'].",
    },
    {
        "user": (
            "The endpoint needs JWT authentication. Show how to add an Authorization header dependency "
            "that decodes the token and extracts user_id."
        ),
    },
    {
        "user": "Write a pytest test for the POST /users endpoint that mocks the JWT dependency.",
    },
]

# ---------------------------------------------------------------
# Session 3 — SQL-heavy session for schema graph nodes
# ---------------------------------------------------------------
session3_turns = [
    {
        "user": (
            "Write SQL to create a table orders with columns: "
            "id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), "
            "product_name VARCHAR(255), amount DECIMAL(10,2), created_at TIMESTAMP."
        ),
    },
    {
        "user": "Add an index on user_id and created_at. Also add a CHECK constraint that amount > 0.",
    },
    {
        "user": "Write a query to get the top 10 users by total order amount in the last 30 days.",
    },
]


if __name__ == "__main__":
    print("=" * 60)
    print("Nexus-Context Dashboard Stress Test")
    print("Watch your dashboard at http://localhost:9000/dashboard")
    print("=" * 60)

    # Run sessions sequentially so we can see each session appear
    run_session("nexus-s1-coding", session1_turns)
    run_session("nexus-s2-fastapi", session2_turns)
    run_session("nexus-s3-sql", session3_turns)

    print("\n" + "=" * 60)
    print("Done. All metrics should now be populated on the dashboard.")
    print("=" * 60)
