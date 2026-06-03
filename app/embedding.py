from collections.abc import Callable
from typing import Any

import httpx


EmbeddingTransport = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]


class DashScopeEmbeddingClient:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        transport: EmbeddingTransport | None = None,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.transport = transport or self._post

    def embed_text(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self.transport(self.api_url, headers, payload)
        return self._parse_embeddings(response)

    @staticmethod
    def _post(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _parse_embeddings(response: dict[str, Any]) -> list[list[float]]:
        data = response.get("data")
        if not isinstance(data, list):
            raise ValueError("embedding response missing data")

        embeddings: list[list[float]] = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("invalid embedding item")
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise ValueError("embedding item missing vector")
            embeddings.append([float(value) for value in embedding])

        if not embeddings:
            raise ValueError("embedding response is empty")
        return embeddings
