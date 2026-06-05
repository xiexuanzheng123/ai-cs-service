# AI Customer Service AI Service

Python FastAPI service for model-side capabilities: LLM reply, embedding, Milvus vector search, OpenSearch keyword search, and Qwen rerank.

## Setup

推荐从 workspace 根目录按统一顺序启动，见 `../README.md`。

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

Local development expects devops MySQL, Redis, Milvus, and OpenSearch to be running first.

```bash
python -m uvicorn app.main:app --env-file .env --reload --port 8000
```

Key environment variables:

```env
MILVUS_URI=http://localhost:19530
OPENSEARCH_URL=http://localhost:9200
AI_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
AI_EMBEDDING_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
AI_RERANK_URL=https://dashscope.aliyuncs.com/compatible-api/v1/reranks
AI_API_KEY=
AI_CHAT_MODEL=qwen3.7-plus
AI_CHAT_FALLBACK_MODEL=qwen-turbo
AI_EMBEDDING_MODEL=text-embedding-v4
AI_RERANK_MODEL=qwen3-rerank
```

## API

```bash
curl http://localhost:8000/healthz
```

Expected local health:

```json
{"status":"ok","milvus":"ok","opensearch":"ok"}
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

## RAG Internals

- `/vector/chunks/upsert`: chunk text -> DashScope embedding -> Milvus.
- `/vector/search`: user query -> embedding -> Milvus vector recall.
- `/keyword/chunks/upsert`: chunk text -> OpenSearch BM25 index.
- `/keyword/search`: user query -> OpenSearch keyword recall.
- `/rerank`: Qwen rerank for top candidates; Gateway falls back to local rerank on failure.
