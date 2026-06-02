# AI Customer Service AI Service

Python FastAPI service for AI orchestration.

The first version is deterministic and rule-based. It gives the Go gateway a stable API before a real LLM, RAG retrieval, and Milvus indexing are connected.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Test

```bash
python -m pytest
```

## Run

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Milvus is optional in the first phase. When it is not configured, `/healthz` reports
`"milvus": "disabled"`. To verify the reserved vector database connection locally:

```bash
python -m uvicorn app.main:app --env-file .env --reload --port 8000
```

## API

```bash
curl http://localhost:8000/healthz
```

```bash
curl -X POST http://localhost:8000/v1/ai/reply \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "session-local",
    "user_id": "demo-user-001",
    "message": "密码错误太多怎么办",
    "history": [],
    "business_context": {}
  }'
```
