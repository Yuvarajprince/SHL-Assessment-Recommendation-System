import pandas as pd
from recommender.retrieve import SHLRecommender
from recommender.rerank import rerank

# ---------------- CONFIG ----------------
CATALOG_PATH = "data/shl_catalog.csv"
OUTPUT_CSV = "evaluation/predictions.csv"
TOP_K = 10
# ---------------------------------------


def build_ground_truth(df):
    """
    Simulated ground truth:
    For each assessment, we assume it should retrieve
    similar assessments based on test_type.
    """
    ground_truth = {}

    for _, row in df.iterrows():
        key = row["assessment_name"]
        test_type = row["test_type"]

        relevant = df[df["test_type"] == test_type]["assessment_name"].tolist()
        ground_truth[key] = set(relevant)

    return ground_truth


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    hit_count = len(set(recommended[:k]) & relevant)
    return hit_count / len(relevant)


def main():
    print("🔹 Loading catalog...")
    df = pd.read_csv(CATALOG_PATH)

    print("🔹 Building simulated ground truth...")
    ground_truth = build_ground_truth(df)

    engine = SHLRecommender()

    results = []
    recalls = []

    print("🔹 Running evaluation...")
    for _, row in df.iterrows():
        query = row["assessment_name"]

        raw = engine.retrieve(query, top_k=20)
        ranked = rerank(raw, query, final_k=TOP_K)

        recommended_names = [r["assessment_name"] for r in ranked]
        relevant = ground_truth.get(query, set())

        recall = recall_at_k(recommended_names, relevant, TOP_K)
        recalls.append(recall)

        for r in ranked:
            results.append({
                "query": query,
                "recommended_assessment": r["assessment_name"],
                "test_type": r["test_type"],
                "score": r["score"]
            })

    avg_recall = sum(recalls) / len(recalls)

    print(f"\n✅ Average Recall@{TOP_K}: {avg_recall:.4f}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"📁 Predictions saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
