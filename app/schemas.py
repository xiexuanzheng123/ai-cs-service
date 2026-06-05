from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)


class AIReplyRequest(BaseModel):
    session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    history: list[ChatHistoryItem] = Field(default_factory=list)
    business_context: dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    doc_id: str
    title: str
    score: float


class AIReplyResponse(BaseModel):
    reply: str
    intent: str
    route: str = "ai_reply"
    risk_level: Literal["low", "medium", "high"]
    transfer_to_human: bool
    retrieved_docs: list[RetrievedDocument] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0


class EmbeddingTextRequest(BaseModel):
    text: str = Field(min_length=1)


class EmbeddingTextResponse(BaseModel):
    embedding: list[float]
    dimension: int
    model: str


class EmbeddingBatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=10)


class EmbeddingBatchResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int
    model: str


class VectorChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    knowledge_id: str = Field(min_length=1)
    chunk_text: str = Field(min_length=1)


class VectorUpsertRequest(BaseModel):
    chunks: list[VectorChunk] = Field(min_length=1, max_length=10)


class VectorUpsertItem(BaseModel):
    chunk_id: str
    vector_id: str


class VectorUpsertResponse(BaseModel):
    items: list[VectorUpsertItem]
    dimension: int
    model: str


class VectorSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class VectorSearchItem(BaseModel):
    chunk_id: str
    knowledge_id: str
    score: float
    chunk_text: str = ""


class VectorSearchResponse(BaseModel):
    items: list[VectorSearchItem]
    dimension: int
    model: str


class KeywordUpsertRequest(BaseModel):
    chunks: list[VectorChunk] = Field(min_length=1, max_length=50)


class KeywordUpsertItem(BaseModel):
    chunk_id: str
    keyword_id: str


class KeywordUpsertResponse(BaseModel):
    items: list[KeywordUpsertItem]
    index: str


class KeywordSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class KeywordSearchItem(BaseModel):
    chunk_id: str
    knowledge_id: str
    score: float
    chunk_text: str = ""


class KeywordSearchResponse(BaseModel):
    items: list[KeywordSearchItem]
    index: str


class RerankDocument(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    documents: list[RerankDocument] = Field(min_length=1, max_length=10)
    top_n: int = Field(default=5, ge=1, le=10)


class RerankItem(BaseModel):
    id: str
    index: int
    score: float


class RerankResponse(BaseModel):
    items: list[RerankItem]
    model: str
