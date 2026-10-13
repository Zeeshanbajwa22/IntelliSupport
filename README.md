# IntelliSupport

An AI support agent for uploaded documents. Each user gets a private knowledge base, and the agent decides per question whether to search that user's document, search the web, or answer directly. When it can't answer reliably, or the user asks for a person, it escalates to a human by creating a support ticket.

Built as a portfolio project covering NLP/LLM engineering and MLOps/AI engineering.

## Features

- **Multi-user isolation:** every user's PDF is stored in a separate vector store. The first upload returns a secret access token (shown once, stored only as a SHA-256 hash), and `/ask` and further uploads require it.
- **Agent routing:** a small Claude model chooses between `document`, `web` and `direct`.
- **RAG:** PDF loading, chunking, local embeddings (`all-MiniLM-L6-v2`), and ChromaDB search.
- **Human escalation:** two triggers (the user asks for a person, or the answer isn't in the sources). Tickets are stored in SQLite.
- **Prompt-injection hardening:** rules live in the system prompt, and retrieved text is wrapped in a randomly named tag and treated as untrusted data.
- **Security:** PDF-only uploads with a size cap and a magic-byte check, `user_id` validation against path traversal, rate limiting, and an admin key on `/tickets`, `/metrics` and `/dashboard`.
- **Monitoring:** every request logs the route, latency, token usage and status. A protected dashboard shows totals, p95 latency, escalation rate and errors.
- **Evaluation:** a 13-question test set scores routing and answers, and an injection test uses a poisoned PDF.
- **CI and Docker:** GitHub Actions runs the test suite on every push, and a Dockerfile packages the app.

## Architecture

```
Client ──► FastAPI ──► rate limit + validation + token check
                │
                ├─► escalation check ("talk to a human") ──► ticket (SQLite)
                │
                ├─► router (Claude Haiku) ──► document | web | direct
                │        ├─ document: ChromaDB (per-user folder) ──► top 8 chunks
                │        ├─ web: ddgs search snippets
                │        └─ direct: no context
                │
                └─► answer (Claude Sonnet) with hardened system prompt
                         └─ "ESCALATE" ──► ticket (SQLite)

Every /ask is logged to SQLite and shown on /dashboard.
```

## Tech stack

Python, FastAPI, LangChain, ChromaDB, sentence-transformers, Anthropic Claude API, SQLite, slowapi, pytest, Docker, GitHub Actions.

## Run locally

1. Create and activate a virtual environment, then install dependencies:
```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
```
2. Create a `.env` file (never commit it):
```
   ANTHROPIC_API_KEY=your-key
   ADMIN_API_KEY=a-long-random-string
```
3. Start the server:
```bash
   uvicorn main:app --reload
```
4. Open `http://127.0.0.1:8000/docs`. Upload a PDF as a new `user_id` and save the returned `access_token`. Send it in the `x-user-token` header for `/ask`.

## Run with Docker

```bash
docker build -t intellisupport .
docker run --rm -p 8000:8000 --env-file .env intellisupport
```

## API

| Endpoint | Access | Purpose |
|---|---|---|
| `POST /upload` | token (existing users) | Upload a PDF, create the user's knowledge base |
| `POST /ask` | user token | Ask a question |
| `GET /tickets` | admin key | List escalation tickets |
| `GET /metrics` | admin key | Usage metrics as JSON |
| `GET /dashboard` | page is open, data needs the admin key | Monitoring dashboard |

## Tests and evaluation

```bash
pytest -v                      # 9 security and behavior tests (no API key needed)
python eval.py                 # 13-question routing and answer evaluation
python eval_injection.py       # prompt-injection test with a poisoned PDF
```

Latest results: 9/9 tests passed, 13/13 on the evaluation set, and the injection test passed once its keyword check was corrected. The model did not obey the planted instructions, although it sometimes mentioned them in its answer.

## Limitations

- The evaluation checks keywords on a small set of questions, so it is a regression test and not proof of correctness.
- Retrieval uses a small embedding model. Details on title pages can rank low, and I raised the chunk count to 8 to compensate.
- Prompt-injection hardening reduces the risk but cannot remove it.
- Data (documents, tickets, metrics) lives on local disk and SQLite, so a container needs a mounted volume to keep it.
- Tickets are only stored. There is no email or chat alert.
- Web search returns snippets, so live facts like weather are not reliable.
- The app is not deployed to a public server yet. The Docker image was built and tested locally, and CI runs the tests but does not build the image.