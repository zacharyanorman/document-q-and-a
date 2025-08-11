# api/vectorstore.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
import chromadb

# Persistent Chroma location
PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Create / load the collection
_client = chromadb.PersistentClient(path=str(PERSIST_DIR))
collection = _client.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"}  # cosine distance works well with embeddings
)

def query_similar(collection, query_embedding: List[float], k: int = 6, where: Optional[Dict[str, Any]] = None):
    """
    Return a list of dicts: {text, metadata, distance}
    """
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
