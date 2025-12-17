# api/app.py
# Safe, lightweight API – NO model loading, NO embedding generation

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
import os

# ---------------- APP INIT ----------------
app = FastAPI(
    title="SHL Assessment Recommendation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDING_DIR = os.path.join(BASE_DIR, "embeddings")

FAISS_INDEX_PATH = os.path.join(EMBEDDING_DIR, "faiss.index")
METADATA_PATH = os.path.join(EMBEDDING_DIR, "metadata.pkl")

# ---------------- LOAD ARTIFACTS ----------------
if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
    raise RuntimeError("FAISS index or metadata file not found.")

index = faiss.read_index(FAISS_INDEX_PATH)

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# ---------------- SCHEMAS ----------------
class RecommendRequest(BaseModel):
    query: str
    top_k: int = 10


class AssessmentResponse(BaseModel):
    name: str
    url: str
    description: str
    duration: int
    adaptive_support: str
    remote_support: str
    test_type: list


# ---------------- UTILS ----------------
def simple_query_vector(query: str, dim: int) -> np.ndarray:
    """
    Lightweight deterministic query vector.
    Avoids ML models completely (safe for free tiers).
    """
    vec = np.zeros(dim, dtype="float32")
    for i, ch in enumerate(query.encode("utf-8")):
        vec[i % dim] += ch
    return vec.reshape(1, -1)


# ---------------- ENDPOINTS ----------------
@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/recommend", response_model=list[AssessmentResponse])
def recommend_assessments(req: RecommendRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    dim = index.d
    query_vector = simple_query_vector(req.query, dim)

    scores, indices = index.search(query_vector, req.top_k)

    results = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata):
            continue

        item = metadata[idx]

        results.append({
            "name": item.get("assessment_name", ""),
            "url": item.get("url", ""),
            "description": item.get("description", ""),
            "duration": int(item.get("duration", 0)),
            "adaptive_support": item.get("adaptive_support", "No"),
            "remote_support": item.get("remote_support", "No"),
            "test_type": (
                item.get("test_type", [])
                if isinstance(item.get("test_type", []), list)
                else [item.get("test_type")]
            )
        })

    return results

