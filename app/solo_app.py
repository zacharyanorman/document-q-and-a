# app/solo_app.py
from __future__ import annotations

# --- Make repo root importable (for Streamlit Cloud) --------------------------
from pathlib import Path
import sys as _sys

ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in _sys.path:
    _sys.path.append(str(ROOT))

# --- Ensure modern SQLite for Chroma on Streamlit Cloud -----------------------
# Chroma may require features missing in the default sqlite3 on Cloud.
# pysqlite3-binary provides a compatible sqlite3 module.
try:
    import pysqlite3 as _pysqlite3  # type: ignore
    _sys.modules["sqlite3"] = _pysqlite3
except Exception:
    # Local runs can ignore this if system sqlite3 is fine.
    pass

import os
import uuid
import tempfile
from typing import List, Dict, Any, Optional

import streamlit as st

# Reuse backend logic
from api.indexer import index_file, embed_texts

# --- UI config ----------------------------------------------------------------
st.set_page_config(page_title="Doc Q and A", page_icon="📄")

# --- OpenAI key retrieval -----------------------------------------------------
def get_openai_key() -> Optional[str]:
    try:
        return st.secrets["OPENAI_API_KEY"]  # Streamlit Cloud Secrets
    except Exception:
        return os.getenv("OPENAI_API_KEY")   # local fallback

OPENAI_API_KEY = get_openai_key()

# Optional helper message
if not OPENAI_API_KEY:
    st.sidebar.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets.")

# --- Writable data dirs (Cloud code dir is read-only) -------------------------
DATA_ROOT = Path(st.secrets.get("DATA_DIR", os.getenv("DATA_DIR", tempfile.gettempdir()))) / "docqa"
UPLOAD_DIR = DATA_ROOT / "uploads"
CHROMA_DIR = DATA_ROOT / "chroma"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --- Chroma client and collection --------------------------------------------
import chromadb

try:
    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
except Exception:
    st.warning("Using in-memory vector store (data resets on app restart).")
    _client = chromadb.EphemeralClient()

collection = _client.get_or_create_collection(name="docs", metadata={"hnsw:space": "cosine"})

def query_similar(collection, query_embedding: List[float], k: int = 6, where: Optional[Dict[str, Any]] = None):
    """Return a list of dicts: {text, metadata, distance}."""
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return [{"text": t, "metadata": m, "distance": float(d)} for t, m, d in zip(docs, metas, dists)]

# --- UI -----------------------------------------------------------------------
st.title("Intelligent Document Q and A")
st.caption("Upload documents, index them, and ask questions with grounded citations.")

# Track titles in this Streamlit session (Option 1: limit Q&A to current session uploads)
if "titles" not in st.session_state:
    st.session_state["titles"] = []

with st.sidebar:
    if st.button("New session"):
        st.session_state["titles"] = []
        st.success("Session cleared. Upload to start fresh.")

# ---- Upload & Index ----------------------------------------------------------
st.subheader("Upload and index")
uploaded = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
title = st.text_input("Title to index under")

can_index = uploaded is not None and bool(title)
if st.button("Upload and index", disabled=not can_index):
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets.")
    else:
        try:
            dest = UPLOAD_DIR / f"{uuid.uuid4()}{Path(uploaded.name).suffix}"
            dest.write_bytes(uploaded.getvalue())

            with st.spinner("Indexing…"):
                n = index_file(
                    collection=collection,
                    file_path=dest,
                    title=title,
                    api_key=OPENAI_API_KEY,
                )

            st.success(f"Indexed {n} chunks for {title}")
            if title not in st.session_state["titles"]:
                st.session_state["titles"].append(title)

        except Exception as e:
            st.error(f"Indexing failed: {e}")

# ---- Ask with citations (limited to this session’s titles) -------------------
st.subheader("Ask your documents")

q = st.text_area("Question")
known_titles = st.session_state.get("titles", [])
sel_titles = st.multiselect("Limit search to titles (optional)", known_titles)

ask_disabled = not q or not known_titles
if st.button("Ask", disabled=ask_disabled):
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets.")
    else:
        try:
            with st.spinner("Retrieving…"):
                # 1) Embed the query
                q_emb = embed_texts([q], api_key=OPENAI_API_KEY)[0]

                # 2) Filter to this session's titles by default
                titles_filter = sel_titles or known_titles
                where = {"title": {"$in": titles_filter}} if titles_filter else None

                # 3) Vector search + simple distance gate
                hits = query_similar(collection, q_emb, k=6, where=where)
                strong = [h for h in hits if h.get("distance", 1.0) <= 0.3] or hits

            # 4) Build prompt & ask the LLM
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            system = (
                "You answer ONLY from the provided context. "
                "If the answer is not in the context, say you don't know. "
                "Cite inline as (Title p.Page)."
            )

            blocks = []
            for i, c in enumerate(strong, start=1):
                meta = c.get("metadata", {})
                t = str(meta.get("title", "Document"))
                p = str(meta.get("page", "?"))
                text = (c.get("text", "") or "")[:1200]
                blocks.append(f"[{i}] {t} p.{p}\n{text}")
            context_text = "\n\n".join(blocks) if blocks else "(no context)"

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Question: {q}\n\nContext:\n{context_text}"},
                ],
            )
            answer = resp.choices[0].message.content.strip()

            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Citations")
            if not strong:
                st.info("No citations returned.")
            else:
                for h in strong[:4]:
                    meta = h.get("metadata", {})
                    t = str(meta.get("title", "Document"))
                    p = str(meta.get("page", "?"))
                    snip = (h.get("text", "") or "").replace("\n", " ")[:200]
                    with st.expander(f"{t} p.{p}"):
                        st.write(snip)

        except Exception as e:
            st.error(f"Ask failed: {e}")
