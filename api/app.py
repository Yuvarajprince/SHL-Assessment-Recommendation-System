# api/app.py
# Safe, lightweight API with FAISS retrieval + correct query embeddings (CPU-only)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# ---------------- APP INIT ----------------
app = FastAPI(
    title="SHL Assessment Recommendation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_DIR = os.path.join(BASE_DIR, "embeddings")

FAISS_INDEX_PATH = os.path.join(EMBEDDING_DIR, "faiss.index")
METADATA_PATH = os.path.join(EMBEDDING_DIR, "metadata.pkl")

# ---------------- LOAD ARTIFACTS ----------------
if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
    raise RuntimeError("FAISS index or metadata file not found.")

index = faiss.read_index(FAISS_INDEX_PATH)

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# ---------------- LOAD ENCODER (CPU ONLY) ----------------
encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

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
    test_type: list[str]


# ---------------- ENDPOINTS ----------------
@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=list[AssessmentResponse])
def recommend(request: RecommendRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Encode query (MATCHES FAISS INDEX SPACE)
    query_vector = encoder.encode(
        request.query,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).reshape(1, -1)

    # FAISS search
    distances, indices = index.search(query_vector, request.top_k)

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
                item.get("test_type")
                if isinstance(item.get("test_type"), list)
                else [item.get("test_type", "")]
            ),
        })

    return results
