from fastapi import FastAPI

from app.orchestrator import AIOrchestrator
from app.schemas import AIReplyRequest, AIReplyResponse

app = FastAPI(title="AI Customer Service AI Service", version="0.1.0")
orchestrator = AIOrchestrator()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/ai/reply", response_model=AIReplyResponse)
def reply(request: AIReplyRequest) -> AIReplyResponse:
    return orchestrator.reply(request)

