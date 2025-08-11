# app/solo_app.py
from __future__ import annotations

# --- make repo root importable (Streamlit Cloud) ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import os
import uuid
import tempfile
import streamlit as st
from typing import List, Dict, Any, Optional

# Reuse your backend logic
from api.indexer import index_file, embed_texts

# Use our own Chroma client here so we can pick a writable path on Cloud
import chromadb

# --- Config / Secrets ---
st.set_page_config(page_title="Doc Q&A", page_icon="📄")

def get_openai_key() -> Optional[str]:
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = get_openai_key()

# --- Writable data dirs (Cloud code dir is read-only) ---
DATA_ROOT = Path(st.secrets.get("DATA_DIR", os.getenv("DATA_DIR", tempfile.gettempdir()))) / "docqa"
UPLOAD_DIR = DATA_ROOT / "uploads"
CHROMA_DIR = DATA_ROOT / "chroma"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --- Chroma persistent client & collection ---
_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = _client.get_or_create_collection(name="docs", metadata={"hnsw:space": "cosine"})

def query_similar(collection, query_embedding: List[float], k: int = 6, where: Optional[Dict[str, Any]] = None):
    """Return a list of dicts: {text, metadata, distance}"""
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    out = []
    for text, meta, dist in zip(docs, metas, dists):
        out.append({"text": text, "metadata": meta, "distance": float(dist)})
    return out

st.title("📄 Intelligent Document Q&A")
st.caption("Upload documents, index them, and ask questions with grounded citations.")

# Track uploaded titles in this Streamlit session (Option 1 behavior)
if "titles" not in st.session_state:
    st.session_state["titles"] = []

# Quick UI reset (does not delete vectors; just clears the filter list)
with st.sidebar:
    if st.button("New session"):
        st.session_state["titles"] = []
        st.success("Session cleared. Upload to start fresh.")

# ---- Upload & Index ----
st.subheader("Upload and index")
uploaded = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
title = st.text_input("Title to index under")

can_index = uploaded is not None and bool(title)
if st.button("Upload and index", disabled=not can_index):
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets (or env).")
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

# ---- Ask with citations (limited to this session’s titles) ----
st.subheader("Ask your documents")

q = st.text_area("Question")
known_titles = st.session_state.get("titles", [])
sel_titles = st.multiselect("Limit search to titles (optional)", known_titles)

ask_disabled = not q or not known_titles
if st.button("Ask", disabled=ask_disabled):
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets (or env).")
    else:
        try:
            with st.spinner("Retrieving…"):
                # 1) Embed the query
                q_emb = embed_texts([q], api_key=OPENAI_API_KEY)[0]

                # 2) Filter to this session's titles by default
                titles_filter = sel_titles or known_titles
                where = {"title": {"$in": titles_filter}}

                # 3) Vector search + simple distance filter
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
