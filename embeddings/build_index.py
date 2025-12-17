import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
DATA_PATH = "data/shl_catalog.csv"
EMBED_DIR = "embeddings"
FAISS_INDEX_PATH = os.path.join(EMBED_DIR, "faiss.index")
META_PATH = os.path.join(EMBED_DIR, "metadata.pkl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# ---------------------------------------


def prepare_text(row):
    """
    Build semantically rich text for embeddings
    """
    parts = [
        row["assessment_name"],
        f"Test Type: {row['test_type']}",
        f"Remote Support: {row['remote_support']}",
        f"Adaptive Support: {row['adaptive_support']}"
    ]
    return " | ".join(parts)


def main():
    print("🔹 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    df["embedding_text"] = df.apply(prepare_text, axis=1)

    texts = df["embedding_text"].tolist()

    print("🔹 Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("🔹 Generating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    print("🔹 Building FAISS index...")
    index = faiss.IndexFlatIP(dimension)  # cosine similarity
    index.add(embeddings)

    os.makedirs(EMBED_DIR, exist_ok=True)

    print("🔹 Saving FAISS index...")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print("🔹 Saving metadata...")
    metadata = df.drop(columns=["embedding_text"]).to_dict(orient="records")
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("\n✅ EMBEDDING PIPELINE COMPLETE")
    print(f"🔢 Total vectors indexed: {index.ntotal}")
    print(f"📁 FAISS index: {FAISS_INDEX_PATH}")
    print(f"📁 Metadata: {META_PATH}")


if __name__ == "__main__":
    main()
