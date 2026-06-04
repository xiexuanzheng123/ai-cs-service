from typing import Any


def format_retrieved_passages(passages: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, passage in enumerate(passages, start=1):
        title = str(passage.get("title") or passage.get("knowledge_id") or f"资料{index}").strip()
        text = str(passage.get("text") or "").strip()
        score = passage.get("score")
        score_text = f"相关度:{score:.3f}" if isinstance(score, (int, float)) else ""
        if not text:
            continue
        blocks.append(f"[{index}] {title} {score_text}\n{text}")
    return "\n\n".join(blocks)
