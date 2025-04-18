from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List
import json
import uvicorn
from sentence_transformers import SentenceTransformer, util
import torch

app = FastAPI(title="SHL Assessment Recommendation API")

@app.get("/debug_assessments")
def debug_assessments():
    # Return first 3 assessments to check data format
    return assessments[:3]

# Load assessment data
with open("shl_assessments.json", "r") as f:
    assessments = json.load(f)

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare data for embeddings including description
assessment_texts = [
    f"{a['name']} {a.get('description', '')} {a.get('test_type', '')} {a.get('duration', '')} {a.get('remote_testing', '')} {a.get('adaptive_irt', '')}"
    for a in assessments
]

assessment_embeddings = model.encode(assessment_texts, convert_to_tensor=True)

class RecommendRequest(BaseModel):
    query: str

class Assessment(BaseModel):
    name: str
    url: str
    remote_testing: str
    adaptive_irt: str
    duration: str
    test_type: str

class RecommendResponse(BaseModel):
    recommendations: List[Assessment]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend_assessments(request: RecommendRequest):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, assessment_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(10, len(assessments)))

    recommendations = []
    for score, idx in zip(top_results.values, top_results.indices):
        a = assessments[idx]
        duration_value = a.get("duration", "")
        if not duration_value:
            duration_value = "Unknown"
        recommendations.append(
            Assessment(
                name=a["name"],
                url=a["url"],
                remote_testing=a.get("remote_testing", "Unknown"),
                adaptive_irt=a.get("adaptive_irt", "Unknown"),
                duration=duration_value,
                test_type=a.get("test_type", "")
            )
        )

    if len(recommendations) == 0:
        raise HTTPException(status_code=404, detail="No recommendations found")

    # Return response using JSONResponse with jsonable_encoder for proper JSON serialization
    response = RecommendResponse(recommendations=recommendations)
    return JSONResponse(content=jsonable_encoder(response))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
