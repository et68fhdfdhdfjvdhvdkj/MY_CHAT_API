import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from main import app


class FakeMessage:
    content = "hi from fake model"


class FakeChoices:
    def __init__(self):
        self.message = FakeMessage()


class FakeCompletions:
    def create(self, *args, **kwargs):
        return type("Response", (), {"choices": [FakeChoices()]})()


class FakeOpenAIClient:
    def __init__(self, *args, **kwargs):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


def test_chat_accepts_gh_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("USE_MOCK_MODE", raising=False)
    monkeypatch.setenv("GH_TOKEN", "fake-token")
    monkeypatch.setattr("main.OpenAI", FakeOpenAIClient)

    client = TestClient(app)
    response = client.post("/api/chat", json={"message": "hello"})

    assert response.status_code == 200, response.text
    assert response.json() == {"reply": "hi from fake model"}


def test_chat_falls_back_to_mock_when_no_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_MODELS_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("USE_MOCK_MODE", raising=False)

    client = TestClient(app)
    response = client.post("/api/chat", json={"message": "hello"})

    assert response.status_code == 200, response.text
    assert "Offline mock response" in response.json()["reply"]


def test_chat_page_is_available():
    response = TestClient(app).get("/chat")

    assert response.status_code == 200
    assert "Coding Tutor" in response.text
    assert "/api/chat/stream" in response.text


def test_chat_get_returns_endpoint_info():
    response = TestClient(app).get("/api/chat")

    assert response.status_code == 200
    assert response.json()["method"] == "POST"
    assert response.json()["streaming_endpoint"] == "/api/chat/stream"


def test_favicon_does_not_return_not_found():
    response = TestClient(app).get("/favicon.ico")

    assert response.status_code == 204


def test_streaming_mock_response_ends_cleanly(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_MODELS_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("USE_MOCK_MODE", raising=False)

    response = TestClient(app).post(
        "/api/chat/stream",
        json={"message": "hello", "history": [{"role": "assistant", "content": "old"}]},
    )

    assert response.status_code == 200
    assert "data: Offline " in response.text
    assert "data: response: " in response.text
    assert "data: [DONE]" in response.text


def test_chat_forwards_history_to_model(monkeypatch):
    captured = {}

    class RecordingCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"choices": [FakeChoices()]})()

    class RecordingClient:
        def __init__(self, *args, **kwargs):
            self.chat = type(
                "Chat", (), {"completions": RecordingCompletions()}
            )()

    monkeypatch.delenv("USE_MOCK_MODE", raising=False)
    monkeypatch.setenv("GH_TOKEN", "fake-token")
    monkeypatch.setattr("main.OpenAI", RecordingClient)

    response = TestClient(app).post(
        "/api/chat",
        json={
            "message": "continue",
            "history": [{"role": "user", "content": "previous"}],
        },
    )

    assert response.status_code == 200
    assert captured["messages"][-2:] == [
        {"role": "user", "content": "previous"},
        {"role": "user", "content": "continue"},
    ]
