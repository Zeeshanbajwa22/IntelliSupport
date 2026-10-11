import re
import uuid
import secrets
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

def call_claude(prompt: str, model: str, max_tokens: int = 1000) -> str:
    response = claude_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

def get_user_vectorstore_path(user_id: str) -> Path:
    validate_user_id(user_id)
    return BASE_VECTORSTORE_DIR / user_id

@app.get("/")
def root():
    return {"status": "IntelliSupport API is running"}
@app.post("/upload")
@limiter.limit("5/minute")
async def upload_document(request: Request, user_id: str = Form(...), file: UploadFile = File(...)):
    validate_user_id(user_id)

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

    return {"status": "success", "user_id": user_id, "chunks_stored": len(chunks)}
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

def decide_action(query: str) -> str:
    """Ask the LLM to decide: 'document', 'web', or 'direct'."""
    decision_prompt = f"""You are deciding how to answer a user's question. 
Respond with ONLY ONE WORD: "document", "web", or "direct".

- Say "document" if the question is likely about specific content in an uploaded file (e.g., project details, personal documents, specific reports).
- Say "web" if the question needs current/live information (e.g., news, prices, recent events, general facts not in a personal document).
- Say "direct" if it's a simple general knowledge question you can answer without searching anything (e.g., "what is 2+2", "define photosynthesis").

Question: {query}

Answer with one word only:"""
    decision = call_claude(decision_prompt, ROUTER_MODEL, max_tokens=10).lower().strip(' .\'"')
    
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

@app.post("/ask", response_model=Answer)
@limiter.limit("30/minute")
def ask(request: Request, question: Question):
    validate_user_id(question.user_id)
    # Trigger 1: explicit request for a human
    if wants_human(question.query):
        tid = create_ticket(question.user_id, question.query, "user_requested_human")
        return Answer(
            answer=f"I've passed your request to our support team (ticket #{tid}). Someone will follow up with you.",
            source_used="escalated",
            ticket_id=tid
        )

    action = decide_action(question.query)
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

    answer_text = call_claude(prompt, ANSWER_MODEL)

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

