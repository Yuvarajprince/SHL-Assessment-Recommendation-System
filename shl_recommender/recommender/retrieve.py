import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
EMBED_DIR = "embeddings"
FAISS_INDEX_PATH = f"{EMBED_DIR}/faiss.index"
META_PATH = f"{EMBED_DIR}/metadata.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# ---------------------------------------


class SHLRecommender:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

        self.index = faiss.read_index(FAISS_INDEX_PATH)

        with open(META_PATH, "rb") as f:
            self.metadata = pickle.load(f)

    def embed_query(self, query: str):
        vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return vec

    def retrieve(self, query: str, top_k: int = 20):
        query_vec = self.embed_query(query)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            item = self.metadata[idx].copy()
            item["score"] = float(score)
            results.append(item)

        return results
