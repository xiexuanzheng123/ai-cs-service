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

