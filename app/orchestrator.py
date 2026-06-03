from app.rules import classify_message
from app.schemas import AIReplyRequest, AIReplyResponse


class AIOrchestrator:
    def __init__(self, chat_client=None) -> None:
        self.chat_client = chat_client

    def reply(self, request: AIReplyRequest) -> AIReplyResponse:
        result = classify_message(request.message)
        reply = result.reply
        if self.chat_client is not None and not result.transfer_to_human:
            reply = self.chat_client.reply(request.message)

        return AIReplyResponse(
            reply=reply,
            intent=result.intent,
            risk_level=result.risk_level,
            transfer_to_human=result.transfer_to_human,
            retrieved_docs=[],
            suggestions=result.suggestions,
        )
