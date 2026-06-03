import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    milvus_uri: str | None
    milvus_token: str | None
    milvus_collection: str
    embedding_dimension: int
    ai_api_url: str | None
    ai_embedding_url: str | None
    ai_api_key: str | None
    ai_chat_model: str
    ai_embedding_model: str


def load_settings() -> Settings:
    load_dotenv()

    ai_api_url = os.getenv("AI_API_URL") or None
    return Settings(
        milvus_uri=os.getenv("MILVUS_URI") or None,
        milvus_token=os.getenv("MILVUS_TOKEN") or None,
        milvus_collection=os.getenv("MILVUS_COLLECTION") or "ai_cs_knowledge_chunk",
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION") or "1024"),
        ai_api_url=ai_api_url,
        ai_embedding_url=os.getenv("AI_EMBEDDING_URL") or infer_embedding_url(ai_api_url),
        ai_api_key=os.getenv("AI_API_KEY") or None,
        ai_chat_model=os.getenv("AI_CHAT_MODEL") or "qwen-plus",
        ai_embedding_model=os.getenv("AI_EMBEDDING_MODEL") or "text-embedding-v4",
    )


def infer_embedding_url(api_url: str | None) -> str | None:
    if not api_url:
        return None
    if api_url.endswith("/chat/completions"):
        return api_url[: -len("/chat/completions")] + "/embeddings"
    return api_url
