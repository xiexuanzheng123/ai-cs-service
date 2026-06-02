import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    milvus_uri: str | None
    milvus_token: str | None


def load_settings() -> Settings:
    load_dotenv()

    return Settings(
        milvus_uri=os.getenv("MILVUS_URI") or None,
        milvus_token=os.getenv("MILVUS_TOKEN") or None,
    )
