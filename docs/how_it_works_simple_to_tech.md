# Nexus-Context: The Complete Guide (Simple Analogies to Technical Deep Dive)

---

## Part 1: What is Nexus-Context? (The Simple Analogies)

### The Core Problem: The AI's 200-Page Notebook Problem

Imagine you are working with a brilliant programmer who has a very specific medical condition: **they forget everything every time you ask a new question**, unless you hand them a written binder containing your entire past conversation.

Every time you type a prompt, your software sends the **entire past conversation history** back to the AI model (LLM / SLM).

As your conversation grows longer, two major problems occur:

1. **The Slowness & Memory Bottleneck (Re-reading the Whole Notebook)**
   - The AI has to read 50 or 100 pages of text from page 1 every single turn.
   - On local computers or GPUs, re-processing thousands of tokens over and over makes response times crawl and drains GPU memory.

2. **The "Torn Page" Error (Referential Amnesia)**
   - To save space, standard AI tools try to "trim" or "compress" old messages by cutting out lines of text.
   - **The Disaster**: A naive compressor might cut out Line 10 (`def connect_db(): ...`) to save space, but leave Line 50 (`conn = connect_db()`). When the AI reads Line 50, it collapses with a `NameError: name 'connect_db' is not defined` because the definition was torn out of the binder!

---

### The Solution: Nexus-Context as the "Smart Library Manager"

**Nexus-Context** is a lightweight, high-performance middleware proxy (like a smart librarian or traffic controller) that sits between your client code (e.g. `demo_agent.py` or your IDE) and your local AI server (Ollama / vLLM / SGLang).

```
┌─────────────────┐       ┌────────────────────────┐       ┌───────────────────────┐
│ Your Python App │ ────> │  Nexus-Context Proxy   │ ────> │  Local SLM Server     │
│ (demo_agent.py) │       │ (localhost:9000/v1)    │       │ (Ollama / vLLM:11434) │
└─────────────────┘       └────────────────────────┘       └───────────────────────┘
```

It solves both problems using **3 simple concepts**:

#### Concept 1: The 3-Zone Context Binder (P, T, R)
Nexus-Context organizes every conversation into 3 distinct zones:

* **Zone P (Padded System Prompt)**: The immutable rulebook laminated on the front cover. Nexus pads this to exact 16-token boundaries so the AI's GPU locks it in cache (KV Cache Lock) and never has to re-read it.
* **Zone T (The Structured Middle History)**: The old conversation turns. Instead of blindly chopping lines, Nexus builds a dependency tree (Graph) of variables and functions. It guarantees that if a function call is kept, its definition is **never** torn out.
* **Zone R (Recent Focus)**: Your current user prompt, kept 100% pristine so the AI answers accurately.

#### Concept 2: The "Sticky-Note" Memory System (WWW Tuples)
When old turns get too large and must be evicted from Zone T, Nexus doesn't throw them in the trash. It distills important changes onto a 4-field sticky note called a **WWW Tuple**:
* **Who**: User, Agent, or Tool
* **What**: State change (e.g. `db_host = "prod.db.internal"`)
* **When**: Turn #1
* **Where**: Scope (e.g. `config.module`)

These sticky notes take almost zero space and are injected directly into Zone P so the AI always remembers critical parameters across hours of interaction.

#### Concept 3: The Live Control Room (Dashboard)
A real-time dashboard running at `http://localhost:9000/dashboard` showing live gauges for latency, cache hit rates, token consumption, and graph connections.

---

## Part 2: Terminal Output Walkthrough

Here is a step-by-step breakdown of what happens in your terminal during a real session.

### 1. Starting the Middleware Server

When you run:
```powershell
nexus-serve --backend-url http://localhost:11434 --backend-type ollama --port 9000 --persist
```

**Terminal Logs & What They Mean:**
```text
INFO:     Started server process [19948]
INFO:nexus_context.memory.ltkb:{"event":"ltkb_initialized","db_path":"nexus_ltkb.db"}
INFO:nexus_context.cache.middleware:{"event":"middleware_startup","backend":"http://localhost:11434","type":"ollama"}
INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
```
* `nexus-serve` launches an OpenAI-compatible proxy server listening on port `9000`.
* It opens the SQLite storage files (`nexus_sessions.db` and `nexus_ltkb.db`) to enable crash-recovery and long-term memory across restarts.

---

### 2. Running a Client Agent Request

When you execute:
```powershell
python demo_agent.py qwen2.5-coder:7b
```

**Turn 1 Console Output:**
```text
--- Turn 1: Database Setup (Model: qwen2.5-coder:7b) ---

Assistant Output:
To define the connection parameters for a PostgreSQL database:
import psycopg2

hostname = 'prod.db.internal'
port_id = 5432
user = 'nexus_admin'
database = 'prod_db'
conn_string = f"host={hostname} port={port_id} user={user} dbname={database}"
```

**What happened under the hood during Turn 1?**
1. `demo_agent.py` sends a `POST /v1/chat/completions` payload to `localhost:9000`.
2. Nexus-Context receives the request with header `X-Session-ID: demo-session-001`.
3. Zone P alignment runs: System prompt `"You are an autonomous Python coding agent."` is padded to a 16-token block boundary. Hash digest `e3b0c442...` is registered.
4. Payload is forwarded to Ollama at `localhost:11434`.
5. Ollama processes Zone P, stores the Key-Value (KV) tensors in GPU memory, and returns the generated Python code.
6. Nexus-Context adds header `X-Nexus-Context-Stats: pipeline_ms=0.3` and returns response to `demo_agent.py`.

---

**Turn 2 Console Output:**
```text
--- Turn 2: Query Execution ---

Assistant Output:
def get_active_orders():
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    cur.execute("SELECT * FROM orders WHERE status = 'active';")
    return cur.fetchall()
```

**What happened under the hood during Turn 2?**
1. Turn 2 carries both Turn 1 and Turn 2 messages.
2. Nexus-Context checks Zone P hash: `e3b0c442...` **MATCHES** Turn 1!
3. **KV Cache Lock Hit!** Ollama reuses the pre-computed system prompt KV tensors directly from GPU cache without re-reading them (Result: **100% KV Cache Hit Rate** on your dashboard).
4. `conn_string` variable defined in Turn 1 is indexed into the AST Context Graph as a dependency edge to `get_active_orders()` in Turn 2.

---

## Part 3: Technical Deep-Dive Architecture

Nexus-Context operates as a 5-stage pipeline executed on every incoming HTTP request:

```
 Incoming Request (POST /v1/chat/completions)
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │ 1. Tool Call Interception & Compress│ (Feature D)
  └─────────────────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │ 2. Block-Aligned Zone P Alignment   │ (Feature H / Zone P)
  └─────────────────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │ 3. AST Context Graph & Cross-Modal  │ (Feature A / Zone T)
  └─────────────────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │ 4. Submodular Compaction Solver     │ (Submodular Max)
  └─────────────────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │ 5. WWW Memory Injection & LTKB      │ (Feature B & I)
  └─────────────────────────────────────┘
                     │
                     ▼
 Forwarded to Backend SLM (vLLM / Ollama / SGLang)
```

---

### Stage 1: Tool Call Interception & Compression (`nexus_context.cache.tool_compressor`)
- When agents call external tools (e.g. database dumps, web searches, shell commands), the responses come back as `role == "tool"` messages containing thousands of tokens of raw JSON.
- **ToolCallCompressor** intercepts these responses *before* token budget calculation.
- Array items are truncated to the first 3 items with a `... N more items` annotation.
- Non-JSON text is truncated at sentence boundaries.
- Reduces raw tool output tokens by 60–80% without destroying structural context.

---

### Stage 2: Block-Aligned Prefix Freezing (`nexus_context.cache.block_align`)
- PagedAttention engines (vLLM, SGLang, Ollama) partition KV caches into fixed token blocks (typically 16 or 32 tokens).
- If Zone P is 15 tokens long, adding 1 token to a downstream prompt shifts all alignment boundaries and invalidates the entire KV cache.
- `BlockAligner` pads Zone P with neutral space tokens so `len(Zone_P_tokens) % block_size == 0`.
- Calculates SHA-256 digest `zone_p_hash`. Across session turns, as long as `zone_p_hash` matches, the backend achieves **100% KV cache hit rate**.

---

### Stage 3: AST Context Graph & Cross-Modal Linking (`nexus_context.guard.ast_graph`)
- Parses Zone-T code turns using AST parsers (Python `ast` stdlib & Tree-Sitter for SQL/Bash/JS).
- Generates a directed acyclic graph $G = (V, E)$:
  - **Vertices $V$**: `AST_FUNCDEF`, `AST_ASSIGNMENT`, `AST_IMPORT`, `TOOL_JSON_FIELD`, `NL_SENTENCE`.
  - **Edges $E$**: Directed dependencies $u \to v$ (e.g. `func_def_to_call`, `import_to_use`, `tool_field_to_code_ref`).

---

### Stage 4: Submodular Compaction Solver (`nexus_context.guard.submodular`)
- When Zone T token count exceeds the allocated budget $B_T$, standard truncation would cause dangling references.
- Nexus-Context formulates compaction as **Submodular Function Maximization**:
  $$\max_{S \subseteq V, c(S) \le B_T} f(S) = \text{Relevance}(S) + \beta \text{Coverage}(S) - \gamma \text{DanglingPenalty}(S)$$
- Setting $\gamma \to \infty$ guarantees that a reference node $v$ **can never be included** in selected set $S$ unless its antecedent definition node $u$ is also included.

---

### Stage 5: WWW Memory Governance & LTKB (`nexus_context.memory`)
- Evicted turns are parsed into 4-tuples $\langle \text{Who}, \text{What}, \text{When}, \text{Where} \rangle$.
- Temporal decay updates retention weight $W(t_i, s_i) = \exp(-\lambda (T - t_i)) \cdot (1 + \eta \cdot \text{AST\_Depth}(s_i))$.
- **Long-Term Knowledge Base (`ltkb.py`)**: At session end, nodes with $W > 0.8$ and in-degree $\ge 2$ are saved to SQLite (`nexus_ltkb.db`). On new session startup, relevant facts are retrieved via TF-IDF and automatically injected into Zone P.

---

## Summary of Dashboard Metrics

| Metric | Simple Explanation | Technical Source |
|---|---|---|
| **Active Sessions** | Number of unique client agent connections active right now | `len(app.state.sessions)` |
| **Pipeline Latency** | Overhead added by Nexus-Context in milliseconds (typically 0.2ms–1.5ms) | `time.perf_counter()` delta |
| **Token Budget Used** | Tokens in current payload vs maximum budget (e.g. 611 / 4096) | `tokens_in / total_budget` |
| **KV Cache Hit Rate** | Percentage of turns that reused the frozen GPU KV-cache (100% = optimal) | `zone_p_hash` match ratio |
| **Memory Pool** | Active Who/What/When/Where sticky notes tracking variable state | `len(session.memory_pool)` |
| **Graph Topology** | Number of code symbols and dependency links built in memory | `len(graph.nodes)` & `len(graph.edges)` |
