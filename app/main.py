from fastapi import FastAPI

from app.config import load_settings
from app.orchestrator import AIOrchestrator
from app.schemas import AIReplyRequest, AIReplyResponse
from app.vector_store import MilvusHealthChecker

app = FastAPI(title="AI Customer Service AI Service", version="0.1.0")
orchestrator = AIOrchestrator()
settings = load_settings()
milvus_checker = (
    MilvusHealthChecker(settings.milvus_uri, settings.milvus_token)
    if settings.milvus_uri
    else None
)


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
    return orchestrator.reply(request)
