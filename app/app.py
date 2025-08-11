# app/app.py
import os
import requests
import streamlit as st

st.set_page_config(page_title="Doc Q and A", page_icon="📄")

def get_base_url() -> str:
    env = os.getenv("API_BASE_URL")
    if env:
        return env
    try:
        return st.secrets["api_base"]
    except Exception:
        return "http://127.0.0.1:8000"

BASE_URL = get_base_url()

st.title("📄 Intelligent Document Q and A")
st.caption("Index documents and ask questions with grounded citations.")

# server health
with st.sidebar:
    st.subheader("Server")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.ok and r.json().get("status") == "ok":
            st.success("API is healthy")
        else:
            st.error("API not healthy")
    except Exception as e:
        st.error(f"Cannot reach API: {e}")

# keep the current session's titles only
if "titles" not in st.session_state:
    st.session_state["titles"] = []

# upload and index
st.subheader("Upload and index")

uploaded = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
title = st.text_input("Title to index under")

if st.button("Upload and index", disabled=not uploaded or not title):
    try:
        with st.spinner("Uploading…"):
            files = {
                "file": (
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type or "application/octet-stream",
                )
            }
            resp_upload = requests.post(
                f"{BASE_URL}/api/upload", files=files, timeout=120
            )
            resp_upload.raise_for_status()
            path = resp_upload.json()["path"]

        with st.spinner("Indexing…"):
            resp_index = requests.post(
                f"{BASE_URL}/api/index",
                json={"path": path, "title": title},
                timeout=600,
            )
            resp_index.raise_for_status()
            info = resp_index.json()

        st.success(f"Indexed {info.get('chunks_indexed', 0)} chunks for {info.get('title', title)}")

        if title not in st.session_state["titles"]:
            st.session_state["titles"].append(title)

    except requests.HTTPError as e:
        try:
            st.error(f"HTTP error: {resp_index.json() if 'resp_index' in locals() else resp_upload.json()}")
        except Exception:
            st.error(f"HTTP error: {e}")
    except Exception as e:
        st.error(f"Failed: {e}")

# ask with citations
st.subheader("Ask your documents")

q = st.text_area("Question")
known_titles = st.session_state.get("titles", [])
sel_titles = st.multiselect("Limit search to titles optional", known_titles)

ask_disabled = not q or not known_titles

if st.button("Ask", disabled=ask_disabled):
    try:
        with st.spinner("Thinking…"):
            # Option 1 behavior: if user did not pick any titles, use all titles from this session
            payload = {"query": q, "titles": (sel_titles or known_titles), "k": 6}
            resp = requests.post(f"{BASE_URL}/api/ask", json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()

        st.markdown("### Answer")
        st.write(data.get("answer", ""))

        st.markdown("### Citations")
        cits = data.get("citations", []) or []
        if not cits:
            st.info("No citations returned.")
        for c in cits:
            t = c.get("title", "Document")
            p = c.get("page", "?")
            snippet = c.get("snippet", "").replace("\n", " ")
            with st.expander(f"{t} p.{p}"):
                st.write(snippet)

    except requests.HTTPError as e:
        try:
            st.error(f"HTTP error: {resp.json()}")
        except Exception:
            st.error(f"HTTP error: {e}")
    except Exception as e:
        st.error(f"Failed: {e}")
