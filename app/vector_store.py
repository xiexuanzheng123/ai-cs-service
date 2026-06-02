class MilvusHealthChecker:
    def __init__(self, uri: str, token: str | None = None) -> None:
        from pymilvus import MilvusClient

        self._client = MilvusClient(uri=uri, token=token)

    def ping(self) -> None:
        self._client.list_collections()
