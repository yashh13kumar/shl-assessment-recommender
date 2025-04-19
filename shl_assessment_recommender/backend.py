from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from test_recommend import recommend

app = FastAPI()

# Configure CORS
origins = [
    "https://shl-assessment-recommender-frontend.onrender.com",
    "http://localhost",
    "http://localhost:8501",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    query: str
    top_k: int = 10

class Recommendation(BaseModel):
    name: str
    url: str
    remote_testing: str
    adaptive_irt: str
    duration: str
    test_type: str
    score: float

class RecommendResponse(BaseModel):
    recommendations: List[Recommendation]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    results = recommend(request.query, top_k=request.top_k)
    recommendations = [
        Recommendation(
            name=r["name"],
            url=r["url"],
            remote_testing=r.get("remote_testing", "Unknown"),
            adaptive_irt=r.get("adaptive_irt", "Unknown"),
            duration=r.get("duration", ""),
            test_type=r.get("test_type", ""),
            score=r.get("score", 0.0)
        )
        for r in results
    ]
    return RecommendResponse(recommendations=recommendations)
