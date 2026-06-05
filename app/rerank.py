from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

RerankTransport = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]


class DashScopeRerankClient:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        transport: RerankTransport | None = None,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.transport = transport or self._post

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict[str, Any]]:
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
            "return_documents": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self.transport(self.api_url, headers, payload)
        return self._parse_results(response)

    @staticmethod
    def _post(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        # Rerank 在客服主链路内，只给较短超时；失败时 Gateway 会使用本地轻量排序降级。
        with httpx.Client(timeout=3.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _parse_results(response: dict[str, Any]) -> list[dict[str, Any]]:
        results = response.get("results")
        if not isinstance(results, list):
            raise ValueError("rerank response missing results")

        parsed: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            relevance_score = item.get("relevance_score")
            if not isinstance(index, int):
                continue
            parsed.append(
                {
                    "index": index,
                    "score": float(relevance_score or 0),
                }
            )
        return parsed
