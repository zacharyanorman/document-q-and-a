# api/main.py
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import hashlib
import shutil
import uuid
import os

from .vectorstore import collection, query_similar
from .indexer import index_file, embed_texts
from openai import OpenAI

app = FastAPI(title="Doc QnA", version="0.1.0")

# ---- Paths / Folders ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---- Health ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---- Upload ------------------------------------------------------------------
class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    bytes: int
    sha256: str
    path: str


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix
    doc_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{doc_id}{ext}"

    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    digest = sha256_of_file(dest)
    size = dest.stat().st_size

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        bytes=size,
        sha256=digest,
        path=str(dest),
    )


# ---- Index -------------------------------------------------------------------
class IndexRequest(BaseModel):
    path: str
    title: str


class IndexResponse(BaseModel):
    chunks_indexed: int
    title: str
    path: str


@app.post("/api/index", response_model=IndexResponse)
def index_doc(body: IndexRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Set OPENAI_API_KEY environment variable.")

    path = Path(body.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    n = index_file(collection, path, title=body.title, api_key=api_key)
    return IndexResponse(chunks_indexed=n, title=body.title, path=str(path))


# ---- Ask (Q&A) ---------------------------------------------------------------
class AskRequest(BaseModel):
    query: str
    titles: Optional[List[str]] = None  # limit search to specific document titles
    k: int = 6                           # top-k chunks to retrieve


class Citation(BaseModel):
    title: str
    page: int
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]


def llm_answer(query: str, contexts: List[dict]) -> str:
    """
    Calls the chat model with the query and selected context chunks.
    Answers only from context and asks the model to cite inline.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system = (
        "You are a helpful assistant. Answer ONLY from the provided context. "
        "If the answer is not in the context, say you don't know. "
        "Cite with (Title p.Page) inline when relevant."
    )

    # Build a compact context block
    blocks = []
    for i, c in enumerate(contexts, start=1):
        meta = c.get("metadata", {})
        title = str(meta.get("title", "Document"))
        page = str(meta.get("page", "?"))
        # keep each chunk bounded so the prompt stays small
        text = c.get("text", "")[:1200]
        blocks.append(f"[{i}] {title} p.{page}\n{text}")
    context_text = "\n\n".join(blocks) if blocks else "(no context)"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context_text}"},
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()


@app.post("/api/ask", response_model=AskResponse)
def ask(body: AskRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Set OPENAI_API_KEY environment variable.")

    # 1) Embed the question
    q_emb = embed_texts([body.query], api_key=api_key)[0]

    # 2) Optional filter by titles
    where = {"title": {"$in": body.titles}} if body.titles else None

    # 3) Retrieve similar chunks from Chroma
    hits = query_similar(collection, q_emb, k=body.k, where=where)

    # Optionally filter weak matches (smaller distance is better in Chroma)
    strong = [h for h in hits if h.get("distance", 1.0) <= 0.3] or hits

    # 4) Ask the LLM with context
    answer_text = llm_answer(body.query, strong)

    # 5) Build citations (cap a few)
    cits: List[Citation] = []
    for h in strong[:4]:
        meta = h.get("metadata", {})
        cits.append(Citation(
            title=str(meta.get("title", "Document")),
            page=int(meta.get("page", 0) or 0),
            snippet=(h.get("text", "")[:200].replace("\n", " ")),
        ))

    return AskResponse(answer=answer_text, citations=cits)
