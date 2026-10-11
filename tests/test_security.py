import os
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-key-for-tests")
os.environ["ADMIN_API_KEY"] = "ci-test-key"

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_is_up():
    assert client.get("/").status_code == 200

def test_rejects_non_pdf_upload():
    r = client.post("/upload", data={"user_id": "ci1"},
                    files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400

def test_rejects_fake_pdf():
    r = client.post("/upload", data={"user_id": "ci1"},
                    files={"file": ("a.pdf", b"not really a pdf", "application/pdf")})
    assert r.status_code == 400

def test_rejects_path_traversal_user_id():
    r = client.post("/ask", json={"user_id": "../hack", "query": "hello"})
    assert r.status_code == 400

def test_rejects_empty_query():
    r = client.post("/ask", json={"user_id": "ci1", "query": ""})
    assert r.status_code == 422

def test_tickets_are_locked():
    assert client.get("/tickets").status_code == 401
    assert client.get("/tickets", headers={"x-api-key": "wrong"}).status_code == 401
    assert client.get("/tickets", headers={"x-api-key": "ci-test-key"}).status_code == 200

def test_human_request_escalates_without_calling_llm():
    r = client.post("/ask", json={"user_id": "ci1", "query": "I want to talk to a human"})
    assert r.status_code == 200
    assert r.json()["source_used"] == "escalated"