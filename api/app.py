from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from recommender.retrieve import SHLRecommender
from recommender.rerank import rerank

# ---------------- APP INIT ----------------
app = FastAPI(
    title="SHL Assessment Recommendation API",
    version="1.0"
)

engine = SHLRecommender()
# -----------------------------------------


# ---------------- SCHEMAS ----------------
class HealthResponse(BaseModel):
    status: str


class RecommendRequest(BaseModel):
    query: str


class RecommendedAssessment(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]


class RecommendResponse(BaseModel):
    recommended_assessments: List[RecommendedAssessment]
# -----------------------------------------


# ---------------- CONSTANTS ----------------
TEST_TYPE_MAP = {
    "K": "Knowledge & Skills",
    "P": "Personality & Behaviour",
    "B": "Personality & Behaviour",
    "D": "Development"
}
# ------------------------------------------


# ---------------- ENDPOINTS ----------------
@app.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend_assessments(req: RecommendRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Step 1: semantic retrieval
    raw_results = engine.retrieve(req.query, top_k=20)

    # Step 2: intent-aware reranking
    final_results = rerank(raw_results, req.query, final_k=10)

    formatted_results = []

    for r in final_results:
        test_code = r.get("test_type")

        formatted_results.append({
            "url": r.get("url", ""),
            "adaptive_support": r.get("adaptive_support", "No"),
            "description": "",          # Not scraped (allowed)
            "duration": 0,              # Not scraped (allowed)
            "remote_support": r.get("remote_support", "No"),
            "test_type": [
                TEST_TYPE_MAP.get(test_code, test_code)
            ] if test_code else []
        })

    return {
        "recommended_assessments": formatted_results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000)
