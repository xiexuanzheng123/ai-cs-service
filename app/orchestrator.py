from app.rules import classify_message
from app.schemas import AIReplyRequest, AIReplyResponse


class AIOrchestrator:
    def reply(self, request: AIReplyRequest) -> AIReplyResponse:
        result = classify_message(request.message)

        return AIReplyResponse(
            reply=result.reply,
            intent=result.intent,
            risk_level=result.risk_level,
            transfer_to_human=result.transfer_to_human,
            retrieved_docs=[],
            suggestions=result.suggestions,
        )

