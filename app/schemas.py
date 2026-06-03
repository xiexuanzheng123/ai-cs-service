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
    risk_level: Literal["low", "medium", "high"]
    transfer_to_human: bool
    retrieved_docs: list[RetrievedDocument] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


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
