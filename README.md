# Craster HR Policy RAG

A production-grade multi-agent chatbot that answers employee questions about Craster HR policies with verified citations and page numbers. Built on a LangGraph pipeline where eight specialised agents handle security, routing, retrieval, evaluation, generation, and verification — so answers are grounded in real policy documents, never fabricated.

---

## What it does

An employee asks: *"How much maternity leave do I get?"*

The system routes the question to the right category of documents, rewrites it into an effective search query, retrieves the most relevant chunks from 50+ HR policy PDFs, scores context quality, generates an answer, and returns it with citations including page numbers — or refuses gracefully when the evidence is insufficient.

Repeated questions return instantly from cache. Prompt injection attempts are blocked before reaching any agent.

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| LLM | Claude Sonnet (`claude-sonnet-4-20250514`) |
| Query rewriting | Ollama — Qwen 2.5 7B (local, free) |
| Embeddings | BAAI/bge-base-en-v1.5 — 768-dim (local, free) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 (local, free) |
| Vector database | Supabase + pgvector |
| Orchestration | LangGraph 0.2.28 |
| Document loading | LangChain 0.3.7 |
| API | FastAPI 0.115.0 + uvicorn |
| Caching | Redis 7 (24-hour TTL) |
| Rate limiting | SlowAPI — 10 req/min per IP |
| Monitoring | Langfuse (query tracing) + MLflow (experiment tracking) |
| Containerisation | Docker + Docker Compose |
| Testing | pytest — unit + integration |
| CI/CD | GitHub Actions |

---

## Pipeline

Eight agents run in sequence inside a LangGraph state machine. Each reads from and writes to a shared `RAGState`.

```
User question
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  GUARD                                                  │
│  Layer 1 — 30+ regex patterns (injection, jailbreaks)  │
│  Layer 2 — Claude classifies: legitimate HR topic?      │
└──────────────────────────┬──────────────────────────────┘
                           │ blocked → VERIFIER (refuse)
                           │ safe ↓
┌─────────────────────────────────────────────────────────┐
│  ROUTER                                                 │
│  Classifies question into one of 9 categories           │
│  leave_family / health_safety / employment / etc.       │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  QUERY REWRITER                                         │
│  Ollama Qwen rewrites vague questions for search        │
│  "what about my leave?" → "employee annual leave        │
│   entitlement days UK policy"                           │
│  Falls back to original if Ollama not running           │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  RETRIEVER                                              │
│  Hybrid search: vector (0.7) + BM25 (0.3) via RRF      │
│  Filtered to detected category on first pass            │
│  Fetches 3× top_k to give reranker room to work         │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  RERANKER                                               │
│  Cross-encoder scores (question, chunk) pairs           │
│  More accurate than vector search alone                 │
│  Narrows to top_k (default 5) chunks                    │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  CONTEXT EVALUATOR                                      │
│  score = (avg_similarity × 0.7) + (quality × 0.3)      │
│  + 0.1 category bonus                                   │
│                                                         │
│  score ≥ 0.8  → high   → generate directly             │
│  score ≥ 0.5  → medium → generate + caveat             │
│  score < 0.5  → low    → retry (max 2, broaden scope)  │
│  exhausted    → none   → refuse                         │
└──────────────────────────┬──────────────────────────────┘
          retry ◄──────────┤ low + retries left
                           │ high / medium ↓
┌─────────────────────────────────────────────────────────┐
│  ANSWER GENERATOR                                       │
│  Claude builds answer from numbered context blocks      │
│  Medium confidence adds: "confirm with HR directly"     │
│  Logs latency + token counts to MLflow                  │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  CITATION VERIFIER                                      │
│  Deduplicates sources by (title, page_number)           │
│  Appends formatted references to final answer           │
│  Returns fallback + HR contact if cannot answer         │
└──────────────────────────┬──────────────────────────────┘
                           ↓
           Final answer + sources + page numbers
```

---

## Quick start (Docker)

The fastest path to a running system. All four services start together.

```bash
# 1. Copy and fill in the required secrets
cp .env.example .env

# Edit .env — minimum required:
# ANTHROPIC_API_KEY=sk-ant-...
# SUPABASE_URL=https://...
# SUPABASE_KEY=...

# 2. Start all services
docker compose -f docker/docker-compose.yml up -d

# 3. Pull the Qwen model into the Ollama container
docker exec craster-rag-ollama ollama pull qwen2.5:7b

# 4. Ingest the HR policy documents
docker exec craster-rag-app python scripts/ingest.py

# 5. Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How much maternity leave do I get?", "user_id": "emp_123"}'
```

**Service ports:**

| Service | URL |
|---|---|
| API + frontend | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Ollama | http://localhost:11434 |
| Redis | localhost:6379 |

---

## Local setup (without Docker)

```bash
# Python 3.11 required
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your keys

# Run Redis locally (or skip — caching auto-disables)
redis-server

# Run Ollama locally (or skip — query rewriting falls back gracefully)
ollama serve
ollama pull qwen2.5:7b

# Ingest documents
python scripts/ingest.py

# Start API
uvicorn craster_rag.api.app:app --reload --host 0.0.0.0 --port 8000
```

---

## Environment variables

All settings load from `.env` via pydantic-settings. The app refuses to start if required values are missing.

```bash
# ── Environment ──────────────────────────────────────
ENVIRONMENT=local              # local | staging | production
APP_VERSION=0.1.0
DEBUG=false

# ── Anthropic (required) ─────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514
ANTHROPIC_MAX_TOKENS=2048

# ── Supabase (required) ──────────────────────────────
SUPABASE_URL=https://...
SUPABASE_KEY=...
SUPABASE_DB_URL=postgresql://...

# ── Embeddings ───────────────────────────────────────
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768
EMBEDDING_BATCH_SIZE=32

# ── RAG core ─────────────────────────────────────────
CHUNK_SIZE=500                 # tokens per chunk
CHUNK_OVERLAP=50
TOP_K_RESULTS=5                # final chunks after reranking
MIN_CONFIDENCE=0.5             # below this, refuse to answer
HIGH_CONFIDENCE=0.8            # above this, answer directly

# ── Hybrid search ─────────────────────────────────────
ENABLE_HYBRID_SEARCH=true
VECTOR_WEIGHT=0.7
BM25_WEIGHT=0.3

# ── Redis caching ─────────────────────────────────────
REDIS_URL=redis://localhost:6379/0
ENABLE_CACHING=true
CACHE_TTL_SECONDS=86400        # 24 hours

# ── Ollama (query rewriting) ──────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# ── API ───────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=change-me-in-production

# ── Langfuse (query monitoring) ───────────────────────
LANGFUSE_PUBLIC_KEY=pk_...
LANGFUSE_SECRET_KEY=sk_...
LANGFUSE_HOST=https://cloud.langfuse.com

# ── MLflow (experiment tracking) ──────────────────────
MLFLOW_TRACKING_URI=mlflow/experiments
MLFLOW_EXPERIMENT_NAME=craster-rag

# ── Feature flags ─────────────────────────────────────
ENABLE_RERANKING=true
ENABLE_QUERY_REWRITING=true
ENABLE_MONITORING=true

# ── Logging ───────────────────────────────────────────
LOG_LEVEL=INFO
LOG_FORMAT=json
```

---

## API

### `POST /api/v1/chat`

Rate limit: 10 requests per minute per IP.

**Request**
```json
{
  "question": "Can I work from home?",
  "user_id": "emp_456"
}
```

**Response**
```json
{
  "final_answer": "Craster operates a Connected Working Policy...\n\n---\n**Sources:**\n[1] Connected Working Policy — Page 3\n[2] Flexible Working Policy — Page 1",
  "answer": "Craster operates a Connected Working Policy...",
  "sources": ["Connected Working Policy", "Flexible Working Policy"],
  "citations": [
    {
      "index": 1,
      "title": "Connected Working Policy",
      "page_number": 3,
      "category": "employment",
      "excerpt": "Employees may request connected working arrangements...",
      "score": 0.91,
      "formatted": "[1] Connected Working Policy — Page 3"
    }
  ],
  "category": "employment",
  "confidence_level": "high",
  "can_answer": true,
  "question": "Can I work from home?"
}
```

### Other endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/api/v1/admin/stats` | Vector store stats (chunks per category) |
| `GET` | `/api/v1/admin/cache-stats` | Cache hit/miss rates |

---

## Document categories

The router classifies every question into one of these categories before retrieval.

| Category | Documents |
|---|---|
| `leave_family` | Maternity, Paternity, Shared Parental, Compassionate, Carers, Neonatal, Parental Bereavement (9 docs) |
| `health_safety` | Lone Worker Policy, H&S Handbook, H&S Policy (3 docs) |
| `employment` | Redundancy, Sickness & Absence, Flexible Working, Connected Working, Disciplinary, Grievance, Drugs & Alcohol, Harassment Procedure (8 docs) |
| `data_compliance` | Whistleblowing, Data Retention, Data Breach, DSAR, Data Protection, Privacy Notice (6 docs) |
| `conduct` | Sexual Harassment Prevention, Harassment & Bullying, Anti-Slavery, Anti-Bribery, Communication & Email, IT Use (6 docs) |
| `rewards_benefits` | Total Reward, Employee Assistance, Healthcare Cash Plan, Expenses, Season Ticket Loan, Recommend a Friend, Pride Award, Pension (8 docs) |
| `equality_diversity` | Equal Opportunities Procedure, Equality & Diversity Policy (2 docs) |
| `company_general` | Employee Handbook, Meeting Etiquette, Environmental Policy, Business Ethics, Travel Policy, Payslip Guide (6 docs) |

Place PDFs in `data/documents/hr_policy/`. The `DOCUMENT_CATEGORIES` mapping in `config.py` assigns each file to its category — add new documents there before re-running ingestion.

---

## Ingestion

```bash
# First-time ingest
python scripts/ingest.py

# Re-ingest after adding or updating documents
# This clears the Redis cache so stale answers are invalidated
python scripts/reindex.py
```

The ingestion pipeline: loads PDFs → splits into 500-token chunks with 50-token overlap → embeds with BAAI/bge-base-en-v1.5 → stores in Supabase with metadata (title, page number, category).

---

## Caching

Redis caches pipeline results keyed by a SHA-256 hash of the normalised question. TTL is 24 hours by default.

- Only successful answers (`can_answer=True`) are cached — "I don't know" responses are never stored
- The cache is checked **before** any agent runs — a hit costs zero Claude tokens and returns in milliseconds
- `scripts/reindex.py` clears all `rag_answer:*` keys after a document update
- If Redis is unreachable on startup, caching silently disables; the pipeline continues normally

---

## Monitoring

### Langfuse

Every query is traced automatically via LangChain's callback system. Each agent node records input, output, and latency. Custom spans also log `context_score`, `confidence_level`, `can_answer`, and citation validity.

Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`. View traces at https://cloud.langfuse.com. If keys are not set, monitoring is skipped silently.

### MLflow

The answer generator logs latency and token counts to MLflow after each generation. View the experiment UI:

```bash
# Local
mlflow ui --backend-store-uri mlflow/experiments --port 5000

# Docker (already running)
open http://localhost:5000
```

---

## Testing

```bash
# All tests
pytest

# Unit only
pytest tests/unit/

# Integration only (requires .env with real keys)
pytest tests/integration/

# With coverage
pytest --cov=craster_rag --cov-report=html
```

---

## Project structure

```
craster-rag/
├── src/craster_rag/
│   ├── agents/
│   │   ├── graph.py              # LangGraph pipeline definition
│   │   ├── state.py              # RAGState TypedDict
│   │   ├── guard_agent.py        # Prompt injection defence
│   │   ├── router_agent.py       # Question categorisation
│   │   ├── query_rewriter.py     # Ollama query rewriting
│   │   ├── retriever_agent.py    # Hybrid search
│   │   ├── reranker.py           # Cross-encoder reranking
│   │   ├── context_evaluator.py  # Confidence scoring + retry logic
│   │   ├── answer_generator.py   # Claude answer generation
│   │   └── citation_verifier.py  # Source extraction + formatting
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── retrieval/
│   │   └── vector_store.py       # Supabase + hybrid search
│   ├── cache/
│   │   └── cache_client.py       # Redis cache wrapper
│   ├── api/
│   │   ├── app.py                # FastAPI app + CORS + rate limiting
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   └── admin.py
│   │   ├── middleware/
│   │   │   └── rate_limiter.py
│   │   └── models/
│   │       ├── request.py
│   │       └── response.py
│   └── monitoring/
│       ├── langfuse_client.py
│       └── mlflow_client.py
├── scripts/
│   ├── ingest.py
│   └── reindex.py
├── tests/
│   ├── unit/
│   └── integration/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.dev
│   └── docker-compose.yml
├── data/documents/hr_policy/    # 50 HR policy PDFs
├── config.py                    # All settings via pydantic-settings
├── requirements.txt
└── .env.example
```

---

## Cost profile

Three of the five most expensive operations run locally for free:

| Operation | Model | Cost |
|---|---|---|
| Embeddings | BAAI/bge-base-en-v1.5 | Free (local) |
| Query rewriting | Ollama Qwen 2.5 7B | Free (local) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Free (local) |
| Routing + guard | Claude Sonnet | ~$0.001/query |
| Answer generation | Claude Sonnet | ~$0.005/query |
| Cached answer | Redis | $0.00 |

At typical usage patterns, caching eliminates 60–80% of Claude API calls for repeated questions.

---

## CI/CD

Three GitHub Actions workflows run on every push:

| Workflow | Trigger | What it does |
|---|---|---|
| `test.yml` | push / PR | pytest unit + integration |
| `lint.yml` | push / PR | ruff + black + mypy |
| `deploy.yml` | push to main | Deploy to production |
