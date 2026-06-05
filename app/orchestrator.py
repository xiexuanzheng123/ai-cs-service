from typing import Any

from app.rules import classify_message
from app.schemas import AIReplyRequest, AIReplyResponse, RetrievedDocument


class AIOrchestrator:
    def __init__(self, chat_client=None) -> None:
        self.chat_client = chat_client

    def reply(self, request: AIReplyRequest) -> AIReplyResponse:
        result = classify_message(request.message)
        passages = _extract_passages(request.business_context)

        if result.transfer_to_human:
            return AIReplyResponse(
                reply=result.reply,
                intent=result.intent,
                route="rule_handoff",
                risk_level=result.risk_level,
                transfer_to_human=True,
                retrieved_docs=[],
                suggestions=result.suggestions,
            )

        if passages:
            if self.chat_client is None:
                raise RuntimeError("chat client is not configured for rag reply")
            reply = self.chat_client.reply_with_knowledge(
                request.message,
                passages,
                history=[item.model_dump() for item in request.history],
            )
            return AIReplyResponse(
                reply=reply,
                intent="rag_llm",
                route="rag_llm",
                risk_level="low",
                transfer_to_human=False,
                retrieved_docs=_passages_to_docs(passages),
                suggestions=["有用", "没用", "转人工"],
            )

        if self.chat_client is not None:
            reply = self.chat_client.reply(
                request.message,
                history=[item.model_dump() for item in request.history],
            )
            return AIReplyResponse(
                reply=reply,
                intent=result.intent,
                route="ai_reply",
                risk_level=result.risk_level,
                transfer_to_human=result.transfer_to_human,
                retrieved_docs=[],
                suggestions=result.suggestions,
            )

        return AIReplyResponse(
            reply=result.reply,
            intent=result.intent,
            route="rule_only",
            risk_level=result.risk_level,
            transfer_to_human=result.transfer_to_human,
            retrieved_docs=[],
            suggestions=result.suggestions,
        )


def _extract_passages(business_context: dict[str, Any]) -> list[dict[str, Any]]:
    raw = business_context.get("retrieved_passages")
    if not isinstance(raw, list):
        return []

    passages: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        passages.append(item)
    return passages


def _passages_to_docs(passages: list[dict[str, Any]]) -> list[RetrievedDocument]:
    docs: list[RetrievedDocument] = []
    for passage in passages:
        knowledge_id = str(passage.get("knowledge_id") or passage.get("chunk_id") or "").strip()
        title = str(passage.get("title") or knowledge_id or "knowledge").strip()
        score = passage.get("score")
        if not isinstance(score, (int, float)):
            score = 0.0
        if not knowledge_id:
            continue
        docs.append(
            RetrievedDocument(
                doc_id=knowledge_id,
                title=title,
                score=float(score),
            )
        )
    return docs
