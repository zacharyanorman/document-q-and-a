# app/solo_app.py
import os
from pathlib import Path
import uuid
import requests  # optional (not used but handy if you later add remote calls)
import streamlit as st

# Reuse your existing backend modules directly
from api.indexer import index_file, embed_texts
from api.vectorstore import collection, query_similar

st.set_page_config(page_title="Doc Q&A", page_icon="📄")

# --- config / secrets ---
def get_api_key() -> str | None:
    # Prefer Streamlit secrets on Cloud; fallback to env var for local runs
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = get_api_key()

# --- folders ---
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.title("📄 Intelligent Document Q&A")
st.caption("Index documents and ask questions with grounded citations.")

# Session state
if "titles" not in st.session_state:
    st.session_state["titles"] = []

# Quick reset for this UI session (does not delete vectors, just the selection)
if st.sidebar.button("New session"):
    st.session_state["titles"] = []
    st.success("Session cleared. Upload to start fresh.")

# ---- Upload & Index ----
st.subheader("Upload and index")

uploaded = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
title = st.text_input("Title to index under")

can_index = uploaded is not None and bool(title)
if st.button("Upload and index", disabled=not can_index):
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets or as an env var.")
    else:
        try:
            # Save file to disk for the indexer
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

# ---- Ask with citations (limited to current session’s titles) ----
st.subheader("Ask your documents")

q = st.text_area("Question")
known_titles = st.session_state.get("titles", [])
sel_titles = st.multiselect("Limit search to titles (optional)", known_titles)

ask_disabled = not q or not known_titles
if st.button("Ask", disabled=ask_disabled):
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets or as an env var.")
    else:
        try:
            with st.spinner("Retrieving…"):
                # 1) embed the query
                q_emb = embed_texts([q], api_key=OPENAI_API_KEY)[0]

                # 2) filter to titles from this session if none selected explicitly
                titles_filter = sel_titles or known_titles
                where = {"title": {"$in": titles_filter}}

                # 3) vector search
                hits = query_similar(collection, q_emb, k=6, where=where)
                strong = [h for h in hits if h.get("distance", 1.0) <= 0.3] or hits

            # 4) build a compact context and call the chat model
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)

            system = (
                "You answer ONLY from the provided context. "
                "If the answer is not in the context, say you don't know. "
                "Cite inline like (Title p.Page)."
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
