from __future__ import annotations

from typing import Any

import httpx


class OpenSearchHealthChecker:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def ping(self) -> None:
        response = httpx.get(self.base_url, timeout=3)
        response.raise_for_status()


class KnowledgeKeywordStore:
    def __init__(self, base_url: str, index_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.index_name = index_name
        self._ensure_index()

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
        operations: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])
            operations.append({"index": {"_index": self.index_name, "_id": chunk_id}})
            operations.append(
                {
                    "chunk_id": chunk_id,
                    "knowledge_id": str(chunk["knowledge_id"]),
                    "chunk_text": str(chunk["chunk_text"]),
                }
            )

        if operations:
            lines = "\n".join(self._json_line(operation) for operation in operations) + "\n"
            response = httpx.post(
                f"{self.base_url}/_bulk?refresh=true",
                content=lines,
                headers={"Content-Type": "application/x-ndjson"},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("errors"):
                raise RuntimeError("opensearch bulk upsert failed")

        return [{"chunk_id": str(chunk["chunk_id"]), "keyword_id": str(chunk["chunk_id"])} for chunk in chunks]

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        body = {
            "size": top_k,
            "query": {
                "match": {
                    "chunk_text": {
                        "query": query,
                        "operator": "or",
                    }
                }
            },
        }
        response = httpx.post(f"{self.base_url}/{self.index_name}/_search", json=body, timeout=5)
        response.raise_for_status()
        hits = response.json().get("hits", {}).get("hits", [])

        items: list[dict[str, Any]] = []
        for hit in hits:
            source = hit.get("_source") or {}
            items.append(
                {
                    "chunk_id": source.get("chunk_id") or hit.get("_id", ""),
                    "knowledge_id": source.get("knowledge_id") or "",
                    "score": float(hit.get("_score") or 0),
                    "chunk_text": source.get("chunk_text") or "",
                }
            )
        return items

    def _ensure_index(self) -> None:
        response = httpx.head(f"{self.base_url}/{self.index_name}", timeout=3)
        if response.status_code == 200:
            return
        if response.status_code != 404:
            response.raise_for_status()

        mapping = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "max_ngram_diff": 4,
                },
                "analysis": {
                    "tokenizer": {
                        "cjk_ngram_tokenizer": {
                            "type": "ngram",
                            "min_gram": 1,
                            "max_gram": 5,
                            "token_chars": ["letter", "digit"],
                        }
                    },
                    "analyzer": {
                        "cjk_ngram_analyzer": {
                            "type": "custom",
                            "tokenizer": "cjk_ngram_tokenizer",
                            "filter": ["lowercase"],
                        }
                    },
                }
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "knowledge_id": {"type": "keyword"},
                    "chunk_text": {"type": "text", "analyzer": "cjk_ngram_analyzer"},
                }
            },
        }
        create_response = httpx.put(f"{self.base_url}/{self.index_name}", json=mapping, timeout=5)
        create_response.raise_for_status()

    @staticmethod
    def _json_line(value: dict[str, Any]) -> str:
        import json

        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
