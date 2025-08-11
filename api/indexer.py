# api/indexer.py
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
import docx2txt
import tiktoken
import numpy as np
from openai import OpenAI

# Use a low-cost embedding model
EMBED_MODEL = "text-embedding-3-small"

def extract_text_and_pages(file_path: Path) -> List[Dict[str, Any]]:
    """
    Returns a list of {'page': int, 'text': str} for PDF,
    or a single-item list for DOCX/TXT.
    """
    suffix = file_path.suffix.lower()
    parts: List[Dict[str, Any]] = []

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        for i, page in enumerate(reader.pages, start=1):
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append({"page": i, "text": txt})
    elif suffix in (".docx",):
        txt = docx2txt.process(str(file_path)) or ""
        if txt.strip():
            parts.append({"page": 1, "text": txt})
    elif suffix in (".txt",):
        txt = file_path.read_text(encoding="utf-8", errors="ignore")
        if txt.strip():
            parts.append({"page": 1, "text": txt})
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return parts

def _token_chunks(text: str, max_tokens: int = 800, overlap: int = 100) -> List[str]:
    """
    Simple token-aware splitter using tiktoken. Targets ~800 tokens/chunk with 100 overlap.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    chunks = []
    i = 0
    while i < len(toks):
        window = toks[i : i + max_tokens]
        chunks.append(enc.decode(window))
        i += max_tokens - overlap
    return chunks

def chunk_pages(pages: List[Dict[str, Any]], title: str) -> List[Dict[str, Any]]:
    """
    Convert page texts into overlapping token chunks with metadata.
    """
    chunks: List[Dict[str, Any]] = []
    for part in pages:
        page_no = part["page"]
        for c in _token_chunks(part["text"]):
            if c.strip():
                chunks.append({
                    "text": c,
                    "metadata": {
                        "title": title,
                        "page": page_no,
                    },
                })
    return chunks

def embed_texts(texts: List[str], api_key: str) -> List[List[float]]:
    client = OpenAI(api_key=api_key)
    # OpenAI embeds up to 2048 items per request; we’ll keep it simple
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def index_file(collection, file_path: Path, title: str, api_key: str) -> int:
    """
    Extract -> chunk -> embed -> upsert to Chroma.
    Returns number of chunks indexed.
    """
    pages = extract_text_and_pages(file_path)
    chunks = chunk_pages(pages, title=title)
    if not chunks:
        return 0

    embeddings = embed_texts([c["text"] for c in chunks], api_key=api_key)

    # stable ids: title|page|idx
    ids = [f"{title}|{c['metadata']['page']}|{i}" for i, c in enumerate(chunks)]
    metadatas = [c["metadata"] for c in chunks]
    documents = [c["text"] for c in chunks]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(chunks)
