from collections.abc import Callable
from typing import Any

import httpx


ChatTransport = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]


class DashScopeChatClient:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        transport: ChatTransport | None = None,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.transport = transport or self._post

    def reply(self, message: str) -> str:
        # 这里是纯模型兜底回复；转人工、RAG 直答等路由决策已经在 Gateway/Orchestrator 前置处理。
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是唱吧智能客服。请用简洁、准确的中文回答用户问题；涉及投诉、退款、账号删除、实名修改等高风险操作时，提示需要转人工。",
                },
                {"role": "user", "content": message},
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self.transport(self.api_url, headers, payload)
        return self._parse_reply(response)

    @staticmethod
    def _post(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _parse_reply(response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("chat response missing choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError("chat response missing message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("chat response missing content")
        return content.strip()
