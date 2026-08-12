"""
demo_agent.py
=============
Example script demonstrating how to connect an OpenAI-compatible agent client
to the Nexus-Context middleware proxy server running at http://localhost:9000/v1.

Prerequisites:
--------------
1. Ensure your local SLM backend (vLLM / Ollama / SGLang) is running.
2. Ensure nexus-serve is running in a separate terminal:
   nexus-serve --backend-url http://localhost:11434 --backend-type ollama --port 9000
3. Run this script: python demo_agent.py [model_name]
   Example: python demo_agent.py qwen2.5-coder:7b
"""

import sys
import os
import json
import urllib.request

# Default model name for Ollama / vLLM (override via sys.argv[1] or MODEL_NAME env var)
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else os.getenv("MODEL_NAME", "qwen2.5-coder:7b")

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def run_with_openai():
    # Initialize OpenAI client pointing to Nexus-Context proxy on port 9000
    client = OpenAI(
        base_url="http://localhost:9000/v1",
        api_key="local-slm",
    )

    # Session header tracks AST graph dependencies and memory for this session
    session_headers = {"X-Session-ID": "demo-session-001"}

    messages = [
        {
            "role": "system",
            "content": "You are an autonomous Python coding agent.",
        }
    ]

    # --- Turn 1 ---
    print(f"--- Turn 1: Database Setup (Model: {MODEL_NAME}) ---")
    messages.append({
        "role": "user",
        "content": "Define connection parameters for PostgreSQL host prod.db.internal, port 5432, user nexus_admin."
    })

    try:
        response1 = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            extra_headers=session_headers,
        )
        answer1 = response1.choices[0].message.content
        print("\nAssistant Output:")
        print(answer1)
        messages.append({"role": "assistant", "content": answer1})

        # --- Turn 2 ---
        print("\n--- Turn 2: Query Execution ---")
        messages.append({
            "role": "user",
            "content": "Now write a function using those exact parameters to query active orders."
        })

        response2 = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            extra_headers=session_headers,
        )
        print("\nAssistant Output:")
        print(response2.choices[0].message.content)

    except Exception as err:
        print(f"\n[Client Error]: {err}")
        print("\nTip: Ensure model is pulled in Ollama (e.g. 'ollama pull qwen2.5-coder:7b')!")


def run_with_urllib():
    url = "http://localhost:9000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Session-ID": "demo-session-001",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an autonomous Python coding agent."},
            {"role": "user", "content": "Define connection parameters for PostgreSQL host prod.db.internal."},
        ],
        "stream": False,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    print(f"--- Sending Request via urllib (Model: {MODEL_NAME}) ---")
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("\nResponse:")
            print(data["choices"][0]["message"]["content"])
    except Exception as err:
        print(f"\n[Connection Error]: {err}")


if __name__ == "__main__":
    if HAS_OPENAI:
        run_with_openai()
    else:
        run_with_urllib()
