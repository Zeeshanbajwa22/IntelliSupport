import re
import uuid
import secrets
import hashlib
from fastapi.responses import HTMLResponse
from fastapi import Request, HTTPException, Header, Depends
from pydantic import Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from typing import Optional
import anthropic
from ddgs import DDGS
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

app = FastAPI(title="IntelliSupport API")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

def validate_user_id(user_id: str):
    if not USER_ID_PATTERN.match(user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id (letters, numbers, - and _ only, max 32).")

def require_admin(x_api_key: str = Header(default="")):
    admin_key = os.getenv("ADMIN_API_KEY", "")
    if not admin_key or not secrets.compare_digest(x_api_key.encode(), admin_key.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")

# Base folder where each user's vector store will live
BASE_VECTORSTORE_DIR = Path("./user_data")
BASE_VECTORSTORE_DIR.mkdir(exist_ok=True)

# Local embedding model (runs on your own computer, no API credits needed)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Claude client
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

ROUTER_MODEL = "claude-haiku-5-5"   # cheap and fast, for the one-word routing decision
ANSWER_MODEL = "claude-sonnet-5-5"  # stronger, for the final answers

def call_claude(prompt: str, model: str, max_tokens: int = 1000, usage: dict = None) -> str:
    response = claude_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    if usage is not None:
        usage["in"] += response.usage.input_tokens
        usage["out"] += response.usage.output_tokens
    return response.content[0].text.strip()

def get_user_vectorstore_path(user_id: str) -> Path:
    validate_user_id(user_id)
    return BASE_VECTORSTORE_DIR / user_id

@app.get("/")
def root():
    return {"status": "IntelliSupport API is running"}
@app.post("/upload")
@limiter.limit("5/minute")
async def upload_document(request: Request, user_id: str = Form(...), file: UploadFile = File(...), x_user_token: str = Header(default="")):
    validate_user_id(user_id)
    is_new = not user_exists(user_id)
    if not is_new and not verify_token(user_id, x_user_token):
        raise HTTPException(status_code=401, detail="Invalid or missing user token.")

    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File is not a valid PDF.")

    temp_path = Path(f"./temp_{uuid.uuid4().hex}.pdf")  # random name, never the user's filename
    temp_path.write_bytes(content)
    try:
        pages = PyPDFLoader(str(temp_path)).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        chunks = splitter.split_documents(pages)
        chunks = [c for c in chunks if len(c.page_content.strip()) > 100]
        if not chunks:
            raise HTTPException(status_code=400, detail="No readable text found in this PDF.")

        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=str(get_user_vectorstore_path(user_id))
        )
    finally:
        temp_path.unlink(missing_ok=True)

    result = {"status": "success", "user_id": user_id, "chunks_stored": len(chunks)}
    if is_new:
        result["access_token"] = create_user(user_id)
        result["note"] = "Save this token now. It is shown only once."
    return result
def search_web(query: str) -> str:
    """Search the web and return a summary of results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No web results found."
        return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Web search failed: {str(e)}"

def decide_action(query: str, usage: dict = None) -> str:
    """Ask the LLM to decide: 'document', 'web', or 'direct'."""
    decision_prompt = f"""You are deciding how to answer a user's question. 
Respond with ONLY ONE WORD: "document", "web", or "direct".

- Say "document" if the question is likely about specific content in an uploaded file (e.g., project details, personal documents, specific reports).
- Say "web" if the question needs current/live information (e.g., news, prices, recent events, general facts not in a personal document).
- Say "direct" if it's a simple general knowledge question you can answer without searching anything (e.g., "what is 2+2", "define photosynthesis").

Question: {query}

Answer with one word only:"""
    decision = call_claude(decision_prompt, ROUTER_MODEL, max_tokens=10, usage=usage).lower().strip(' .\'"')
    
    # Safety fallback in case the model says something unexpected
    if decision not in ["document", "web", "direct"]:
        decision = "document"
    return decision

# ---------- Ticket database ----------
DB_PATH = "tickets.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        question TEXT,
        reason TEXT,
        status TEXT,
        created_at TEXT
    )""")
    conn.commit()
    conn.close()

init_db()

def init_metrics():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT, user_id TEXT, source_used TEXT,
        latency_ms INTEGER, in_tokens INTEGER, out_tokens INTEGER, status TEXT)""")
    conn.commit()
    conn.close()

init_metrics()

def log_request(user_id, source, latency_ms, usage, status):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO requests (created_at, user_id, source_used, latency_ms, in_tokens, out_tokens, status) VALUES (?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), user_id, source, latency_ms, usage["in"], usage["out"], status)
    )
    conn.commit()
    conn.close()
def init_users():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY, token_hash TEXT, created_at TEXT)""")
    conn.commit()
    conn.close()

init_users()

def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

def user_exists(user_id: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return row is not None

def create_user(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO users (user_id, token_hash, created_at) VALUES (?, ?, ?)",
                 (user_id, _hash_token(token), datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()
    return token

def verify_token(user_id: str, token: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT token_hash FROM users WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    if not row or not token:
        return False
    return secrets.compare_digest(row[0], _hash_token(token))

def create_ticket(user_id: str, question: str, reason: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "INSERT INTO tickets (user_id, question, reason, status, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, question, reason, "open", datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    ticket_id = cur.lastrowid
    conn.close()
    return ticket_id

def wants_human(query: str) -> bool:
    """Trigger 1: the user explicitly asks for a person."""
    phrases = ["human", "real person", "representative", "customer service",
               "support staff", "talk to someone", "speak to someone", "speak to a person"]
    q = query.lower()
    return any(p in q for p in phrases)

# ---------- Ask endpoint ----------
class Question(BaseModel):
    user_id: str
    query: str = Field(min_length=1, max_length=1000)

class Answer(BaseModel):
    answer: str
    source_used: str
    ticket_id: Optional[int] = None

def handle_question(question: Question, usage: dict) -> Answer:
    validate_user_id(question.user_id)
    # Trigger 1: explicit request for a human
    if wants_human(question.query):
        tid = create_ticket(question.user_id, question.query, "user_requested_human")
        return Answer(
            answer=f"I've passed your request to our support team (ticket #{tid}). Someone will follow up with you.",
            source_used="escalated",
            ticket_id=tid
        )

    action = decide_action(question.query, usage)
    context = ""

    if action == "document":
        user_path = get_user_vectorstore_path(question.user_id)
        if not user_path.exists():
            return Answer(
                answer="You haven't uploaded any document yet. Please upload one first.",
                source_used="none"
            )
        vectorstore = Chroma(
            persist_directory=str(user_path),
            embedding_function=embedding_model
        )
        results = vectorstore.similarity_search(question.query, k=8)
        context = "\n\n".join([doc.page_content for doc in results])
    elif action == "web":
        context = search_web(question.query)

    if context:
        prompt = f"""You are a helpful support assistant. Use ONLY the context below to answer.
If the context does not contain enough information to answer the question,
reply with exactly one word: ESCALATE

Context:
{context}

Question: {question.query}

Answer:"""
    else:
        prompt = f"Answer this question directly and concisely:\n\n{question.query}"

    answer_text = call_claude(prompt, ANSWER_MODEL, usage=usage)

    # Trigger 2: the agent can't answer from what it found
    if answer_text.upper().startswith("ESCALATE"):
        tid = create_ticket(question.user_id, question.query, "low_confidence")
        return Answer(
            answer=f"I couldn't find a reliable answer, so I've created a support ticket (#{tid}) for a human to review.",
            source_used="escalated",
            ticket_id=tid
        )

    return Answer(answer=answer_text, source_used=action)

@app.get("/tickets", dependencies=[Depends(require_admin)])
def list_tickets():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, user_id, question, reason, status, created_at FROM tickets ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "user_id": r[1], "question": r[2], "reason": r[3], "status": r[4], "created_at": r[5]}
        for r in rows
    ]
@app.post("/ask", response_model=Answer)
@limiter.limit("30/minute")
def ask(request: Request, question: Question, x_user_token: str = Header(default="")):
    validate_user_id(question.user_id)
    if not verify_token(question.user_id, x_user_token):
        raise HTTPException(status_code=401, detail="Invalid or missing user token.")

    start = time.perf_counter()
    usage = {"in": 0, "out": 0}
    try:
        answer = handle_question(question, usage)
    except HTTPException:
        raise
    except Exception:
        log_request(question.user_id, "error", int((time.perf_counter() - start) * 1000), usage, "error")
        raise
    log_request(question.user_id, answer.source_used, int((time.perf_counter() - start) * 1000), usage, "ok")
    return answer


PRICE_IN_PER_MTOK = 0.0   # USD per million input tokens of your ANSWER model (fill from Anthropic's pricing page)
PRICE_OUT_PER_MTOK = 0.0  # USD per million output tokens

@app.get("/metrics", dependencies=[Depends(require_admin)])
def metrics():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT created_at, user_id, source_used, latency_ms, in_tokens, out_tokens, status FROM requests ORDER BY id DESC"
    ).fetchall()
    conn.close()

    total = len(rows)
    by_source = {}
    for r in rows:
        by_source[r[2]] = by_source.get(r[2], 0) + 1
    lat = sorted(r[3] for r in rows)
    in_tok = sum(r[4] for r in rows)
    out_tok = sum(r[5] for r in rows)
    return {
        "total": total,
        "avg_latency_ms": round(sum(lat) / total) if total else 0,
        "p95_latency_ms": lat[int(0.95 * (len(lat) - 1))] if lat else 0,
        "escalation_rate_pct": round(100 * by_source.get("escalated", 0) / total, 1) if total else 0,
        "errors": sum(1 for r in rows if r[6] == "error"),
        "in_tokens": in_tok,
        "out_tokens": out_tok,
        "est_cost_usd": round(in_tok / 1e6 * PRICE_IN_PER_MTOK + out_tok / 1e6 * PRICE_OUT_PER_MTOK, 4),
        "by_source": by_source,
        "recent": [
            {"created_at": r[0], "user_id": r[1], "source_used": r[2], "latency_ms": r[3], "status": r[6]}
            for r in rows[:10]
        ],
    }

DASHBOARD_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>IntelliSupport Dashboard</title>
<style>body{font-family:sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}
.card{display:inline-block;background:#f3f4f6;border-radius:8px;padding:12px 16px;margin:6px}
table{border-collapse:collapse;width:100%;margin-top:1rem}td,th{border-bottom:1px solid #ddd;padding:6px;text-align:left}</style></head>
<body><h1>IntelliSupport Dashboard</h1>
<input id="key" type="password" placeholder="Admin key"> <button onclick="load()">Load</button>
<div id="cards"></div><table id="tbl"></table>
<script>
async function load(){
  const r = await fetch('/metrics', {headers: {'x-api-key': document.getElementById('key').value}});
  const cards = document.getElementById('cards'); const tbl = document.getElementById('tbl');
  cards.textContent=''; tbl.textContent='';
  if(!r.ok){cards.textContent='Wrong key or server error ('+r.status+')'; return;}
  const m = await r.json();
  const add=(k,v)=>{const d=document.createElement('div'); d.className='card'; d.textContent=k+': '+v; cards.appendChild(d);};
  add('Total requests', m.total); add('Avg latency (ms)', m.avg_latency_ms); add('p95 latency (ms)', m.p95_latency_ms);
  add('Escalation rate', m.escalation_rate_pct+'%'); add('Errors', m.errors);
  add('Input tokens', m.in_tokens); add('Output tokens', m.out_tokens); add('Est. cost (USD)', m.est_cost_usd);
  for (const [k,v] of Object.entries(m.by_source)) add('Source '+k, v);
  const head = tbl.insertRow();
  for (const h of ['Time','User','Source','ms','Status']) {const th=document.createElement('th'); th.textContent=h; head.appendChild(th);}
  for (const x of m.recent) {const row=tbl.insertRow(); for (const c of [x.created_at,x.user_id,x.source_used,x.latency_ms,x.status]) row.insertCell().textContent=c;}
}
</script></body></html>"""

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML
