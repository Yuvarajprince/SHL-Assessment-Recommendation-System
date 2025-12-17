import pandas as pd
from recommender.retrieve import SHLRecommender
from recommender.rerank import rerank

OUTPUT = "evaluation/submission.csv"

engine = SHLRecommender()

queries = [
    "Java developer",
    "Python developer",
    "Business analyst",
    "Leadership role",
    "Software engineer"
]

rows = []

for q in queries:
    raw = engine.retrieve(q, top_k=20)
    ranked = rerank(raw, q, final_k=10)

    for r in ranked:
        rows.append({
            "Query": q,
            "Assessment_url": r["url"]
        })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)

print(f"✅ Submission CSV saved to {OUTPUT}")
