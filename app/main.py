from fastapi import FastAPI

from app.config import load_settings
from app.embedding import DashScopeEmbeddingClient
from app.llm import DashScopeChatClient
from app.orchestrator import AIOrchestrator
from app.schemas import (
    AIReplyRequest,
    AIReplyResponse,
    EmbeddingBatchRequest,
    EmbeddingBatchResponse,
    EmbeddingTextRequest,
    EmbeddingTextResponse,
    VectorSearchRequest,
    VectorSearchResponse,
    VectorUpsertRequest,
    VectorUpsertResponse,
)
from app.vector_store import KnowledgeVectorStore, MilvusHealthChecker

app = FastAPI(title="AI Customer Service AI Service", version="0.1.0")
settings = load_settings()

# Python 服务只承接模型相关能力：LLM、embedding、Milvus；业务编排仍由 Go Gateway 控制。
chat_client = (
    DashScopeChatClient(
        api_url=settings.ai_api_url,
        api_key=settings.ai_api_key,
        model=settings.ai_chat_model,
    )
    if settings.ai_api_url and settings.ai_api_key
    else None
)
orchestrator = AIOrchestrator(chat_client=chat_client)
milvus_checker = (
    MilvusHealthChecker(settings.milvus_uri, settings.milvus_token)
    if settings.milvus_uri
    else None
)
if settings.milvus_uri:
    app.state.vector_store = KnowledgeVectorStore(
        uri=settings.milvus_uri,
        token=settings.milvus_token,
        collection_name=settings.milvus_collection,
        dimension=settings.embedding_dimension,
    )
else:
    app.state.vector_store = None
if settings.ai_embedding_url and settings.ai_api_key:
    app.state.embedding_client = DashScopeEmbeddingClient(
        api_url=settings.ai_embedding_url,
        api_key=settings.ai_api_key,
        model=settings.ai_embedding_model,
    )
else:
    app.state.embedding_client = None


@app.get("/healthz")
def healthz() -> dict[str, str]:
    milvus_status = "disabled"
    if milvus_checker is not None:
        try:
            milvus_checker.ping()
            milvus_status = "ok"
        except Exception:
            milvus_status = "error"

    return {
        "status": "error" if milvus_status == "error" else "ok",
        "milvus": milvus_status,
    }


@app.post("/v1/ai/reply", response_model=AIReplyResponse)
def reply(request: AIReplyRequest) -> AIReplyResponse:
    # Gateway 在 RAG 低置信或需要模型兜底时调用这里。
    return orchestrator.reply(request)


@app.post("/embedding/text", response_model=EmbeddingTextResponse)
def embedding_text(request: EmbeddingTextRequest) -> EmbeddingTextResponse:
    embedding_client = get_embedding_client()
    embedding = embedding_client.embed_text(request.text)
    return EmbeddingTextResponse(
        embedding=embedding,
        dimension=len(embedding),
        model=embedding_client.model,
    )


@app.post("/embedding/batch", response_model=EmbeddingBatchResponse)
def embedding_batch(request: EmbeddingBatchRequest) -> EmbeddingBatchResponse:
    embedding_client = get_embedding_client()
    embeddings = embedding_client.embed_batch(request.texts)
    dimension = len(embeddings[0]) if embeddings else 0
    return EmbeddingBatchResponse(
        embeddings=embeddings,
        dimension=dimension,
        model=embedding_client.model,
    )


@app.post("/vector/chunks/upsert", response_model=VectorUpsertResponse)
def vector_chunks_upsert(request: VectorUpsertRequest) -> VectorUpsertResponse:
    embedding_client = get_embedding_client()
    vector_store = get_vector_store()
    chunks = [chunk.model_dump() for chunk in request.chunks]
    # 写入链路：chunk 文本 -> embedding -> Milvus，vector_id 再由 Gateway 回写 MySQL。
    embeddings = embedding_client.embed_batch([chunk["chunk_text"] for chunk in chunks])
    items = vector_store.upsert_chunks(chunks, embeddings)
    dimension = len(embeddings[0]) if embeddings else 0
    return VectorUpsertResponse(
        items=items,
        dimension=dimension,
        model=embedding_client.model,
    )


@app.post("/vector/search", response_model=VectorSearchResponse)
def vector_search(request: VectorSearchRequest) -> VectorSearchResponse:
    embedding_client = get_embedding_client()
    vector_store = get_vector_store()
    # 检索链路：用户问题先转 embedding，再用 Milvus 找最相似的知识 chunk。
    embedding = embedding_client.embed_text(request.query)
    items = vector_store.search(embedding, request.top_k)
    return VectorSearchResponse(
        items=items,
        dimension=len(embedding),
        model=embedding_client.model,
    )


def get_embedding_client() -> DashScopeEmbeddingClient:
    embedding_client = app.state.embedding_client
    if embedding_client is None:
        raise RuntimeError("embedding client is not configured")
    return embedding_client


def get_vector_store() -> KnowledgeVectorStore:
    vector_store = app.state.vector_store
    if vector_store is None:
        raise RuntimeError("vector store is not configured")
    return vector_store
