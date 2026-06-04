class MilvusHealthChecker:
    def __init__(self, uri: str, token: str | None = None) -> None:
        from pymilvus import MilvusClient

        self._client = MilvusClient(uri=uri, token=token)

    def ping(self) -> None:
        self._client.list_collections()


class KnowledgeVectorStore:
    def __init__(
        self,
        uri: str,
        token: str | None,
        collection_name: str,
        dimension: int,
        client=None,
    ) -> None:
        if client is None:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=uri, token=token)
        self._client = client
        self.collection_name = collection_name
        self.dimension = dimension
        self._ensure_collection()

    def upsert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> list[dict]:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        # 使用 chunk_id 作为 Milvus 主键，重复同步同一知识时会覆盖旧向量。
        data = []
        items = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            vector_id = chunk["chunk_id"]
            data.append(
                {
                    "id": vector_id,
                    "chunk_id": chunk["chunk_id"],
                    "knowledge_id": chunk["knowledge_id"],
                    "chunk_text": chunk["chunk_text"],
                    "embedding": embedding,
                }
            )
            items.append({"chunk_id": chunk["chunk_id"], "vector_id": vector_id})

        if data:
            self._client.upsert(collection_name=self.collection_name, data=data)
        return items

    def search(self, embedding: list[float], top_k: int) -> list[dict]:
        # 只返回召回标识和分数；完整答案内容由 Gateway 再回查 MySQL。
        results = self._client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=top_k,
            output_fields=["chunk_id", "knowledge_id", "chunk_text"],
        )
        if not results:
            return []

        matches = []
        for item in results[0]:
            entity = item.get("entity", {})
            matches.append(
                {
                    "chunk_id": entity.get("chunk_id") or item.get("id", ""),
                    "knowledge_id": entity.get("knowledge_id", ""),
                    "score": float(item.get("distance", 0)),
                    "chunk_text": entity.get("chunk_text", ""),
                }
            )
        return matches

    def _ensure_collection(self) -> None:
        if self._client.has_collection(self.collection_name):
            return

        from pymilvus import DataType

        # 首次启动时自动创建 collection，维度必须与 embedding 模型输出一致。
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="knowledge_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="chunk_text", datatype=DataType.VARCHAR, max_length=8192)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
