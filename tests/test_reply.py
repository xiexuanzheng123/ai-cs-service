from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def post_message(message: str) -> dict:
    response = client.post(
        "/v1/ai/reply",
        json={
            "session_id": "session-test",
            "user_id": "demo-user-001",
            "message": message,
            "history": [],
            "business_context": {},
        },
    )
    assert response.status_code == 200
    return response.json()


def test_refund_requires_handoff() -> None:
    payload = post_message("我要退款")

    assert payload["intent"] == "refund"
    assert payload["risk_level"] == "high"
    assert payload["transfer_to_human"] is True


def test_complaint_requires_handoff() -> None:
    payload = post_message("我要投诉")

    assert payload["intent"] == "complaint"
    assert payload["risk_level"] == "high"
    assert payload["transfer_to_human"] is True


def test_password_problem_returns_guided_answer() -> None:
    payload = post_message("密码错误太多怎么办")

    assert payload["intent"] == "account_problem"
    assert payload["risk_level"] == "medium"
    assert payload["transfer_to_human"] is False
    assert "找回账号密码" in payload["suggestions"]


def test_unknown_question_returns_fallback() -> None:
    payload = post_message("这个活动什么时候开始")

    assert payload["intent"] == "unknown"
    assert payload["risk_level"] == "medium"
    assert payload["transfer_to_human"] is False
    assert payload["reply"]


def test_healthz_reports_milvus_status() -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["milvus"] in {"disabled", "ok"}
