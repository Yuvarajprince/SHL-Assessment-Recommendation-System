from recommender.retrieve import SHLRecommender
from recommender.rerank import rerank

engine = SHLRecommender()

query = "Looking for a java developer who can collaborate with business teams"

raw_results = engine.retrieve(query, top_k=20)
final_results = rerank(raw_results, query, final_k=10)

for r in final_results:
    print(r["assessment_name"], "|", r["test_type"])
