from collections.abc import Callable
from typing import Any

import httpx

from app.rag_prompt import format_retrieved_passages

ChatTransport = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]

RAG_SYSTEM_PROMPT = """你是智能客服助手。请仅根据下面「知识库参考」回答用户问题。
- 只能使用参考中的事实，不要编造客服电话、工作时间、退款政策、处理流程等。
- 参考不足以回答时，明确说明暂未查到相关信息，并建议用户补充描述或转人工。
- 涉及投诉、退款、账号删除、实名修改等高风险操作时，提示需要转人工。
- 用简洁、准确的中文回答。"""


class DashScopeChatClient:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        fallback_model: str | None = None,
        transport: ChatTransport | None = None,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.fallback_model = fallback_model
        self.transport = transport or self._post

    def reply_with_knowledge(
        self,
        message: str,
        passages: list[dict[str, Any]],
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        knowledge_block = format_retrieved_passages(passages)
        if not knowledge_block.strip():
            raise ValueError("retrieved passages are empty")

        system_content = f"{RAG_SYSTEM_PROMPT}\n\n知识库参考：\n{knowledge_block}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                *_valid_history(history),
                {"role": "user", "content": message},
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response, model_used = self._call_with_fallback(headers, payload)
        return self._parse_reply(response, model_used)

    def reply(self, message: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是智能客服助手。请用简洁、准确的中文回答用户问题；涉及投诉、退款、账号删除、实名修改等高风险操作时，提示需要转人工。",
                },
                *_valid_history(history),
                {"role": "user", "content": message},
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response, model_used = self._call_with_fallback(headers, payload)
        return self._parse_reply(response, model_used)

    def _call_with_fallback(self, headers: dict[str, str], payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        try:
            return self.transport(self.api_url, headers, payload), str(payload["model"])
        except Exception:
            if not self.fallback_model or self.fallback_model == payload.get("model"):
                raise
            fallback_payload = dict(payload)
            fallback_payload["model"] = self.fallback_model
            return self.transport(self.api_url, headers, fallback_payload), self.fallback_model

    @staticmethod
    def _post(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _parse_reply(response: dict[str, Any], model_used: str) -> dict[str, Any]:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("chat response missing choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError("chat response missing message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("chat response missing content")
        usage = response.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        return {
            "reply": content.strip(),
            "model": model_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": estimate_model_cost(model_used, input_tokens, output_tokens),
        }


def _valid_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    if not history:
        return []
    messages: list[dict[str, str]] = []
    for item in history[-6:]:
        role = item.get("role")
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def estimate_model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    # 先做观测用粗估，后续接正式价格表时只替换这里。
    model = model.lower()
    if "turbo" in model:
        input_per_1k = 0.0003
        output_per_1k = 0.0006
    elif "plus" in model:
        input_per_1k = 0.0008
        output_per_1k = 0.002
    else:
        input_per_1k = 0.001
        output_per_1k = 0.002
    return round(input_tokens / 1000 * input_per_1k + output_tokens / 1000 * output_per_1k, 8)
