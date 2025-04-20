from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List
import json
import uvicorn
import os
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
model = SentenceTransformer('all-mpnet-base-v2')

# Prepare data for embeddings including description, keywords, and structured attributes
assessment_texts = [
    f"{a['name']} {a.get('description', '')} {a.get('test_type', '')} {a.get('duration', '')} {a.get('remote_testing', '')} {a.get('adaptive_irt', '')} {' '.join(a.get('keywords', []))} {a.get('job_level', '')} {a.get('industry', '')} {a.get('language', '')} {a.get('job_family', '')}"
    for a in assessments
]

assessment_embeddings = model.encode(assessment_texts, convert_to_tensor=True)

class RecommendRequest(BaseModel):
    query: str
    top_k: int = 10

class Assessment(BaseModel):
    name: str
    url: str
    remote_testing: str
    adaptive_irt: str
    duration: str
    test_type: str
    score: float

class RecommendResponse(BaseModel):
    recommendations: List[Assessment]

def extract_attributes_from_query(query):
    """Extract structured attributes from the query using simple keyword matching."""
    query_lower = query.lower()
    job_levels = ["entry-level", "mid-level", "senior", "manager", "director", "executive"]
    industries = ["finance", "healthcare", "technology", "retail", "manufacturing", "education", "hospitality"]
    languages = ["english", "spanish", "french", "german", "chinese"]
    job_families = ["engineering", "sales", "marketing", "human resources", "operations", "customer service"]

    duration_patterns = [
        (r'(\d+)\s*minutes', 1),
        (r'(\d+)\s*min', 1),
        (r'(\d+)\s*hour', 60),
        (r'(\d+)\s*hr', 60),
    ]

    extracted = {
        "job_level": "",
        "industry": "",
        "language": "",
        "job_family": "",
        "max_duration": None
    }

    import re

    for jl in job_levels:
        if jl in query_lower:
            extracted["job_level"] = jl
            break
    for ind in industries:
        if ind in query_lower:
            extracted["industry"] = ind
            break
    for lang in languages:
        if lang in query_lower:
            extracted["language"] = lang
            break
    for jf in job_families:
        if jf in query_lower:
            extracted["job_family"] = jf
            break

    # Extract max duration if mentioned
    for pattern, multiplier in duration_patterns:
        match = re.search(pattern, query_lower)
        if match:
            extracted["max_duration"] = int(match.group(1)) * multiplier
            break

    return extracted

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend_assessments(request: RecommendRequest):
    query = request.query
    top_k = request.top_k
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    extracted_attrs = extract_attributes_from_query(query)
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, assessment_embeddings)[0]

    # Boost scores for assessments matching extracted attributes
    boosted_scores = []
    for idx, score in enumerate(cos_scores):
        a = assessments[idx]
        boost = 0.0
        if extracted_attrs["job_level"] and extracted_attrs["job_level"] == a.get("job_level", ""):
            boost += 0.5
        if extracted_attrs["industry"] and extracted_attrs["industry"] == a.get("industry", ""):
            boost += 0.5
        if extracted_attrs["language"] and extracted_attrs["language"] == a.get("language", ""):
            boost += 0.3
        if extracted_attrs["job_family"] and extracted_attrs["job_family"] == a.get("job_family", ""):
            boost += 0.5

        # Filter by max duration if specified
        if extracted_attrs["max_duration"] is not None:
            try:
                duration_str = a.get("duration", "")
                if duration_str:
                    # Extract numeric duration in minutes from string if possible
                    import re
                    match = re.search(r'(\d+)', duration_str)
                    if match:
                        duration_val = int(match.group(1))
                        if duration_val > extracted_attrs["max_duration"]:
                            # Skip this assessment by setting boost to very low
                            boost = -1000.0
            except Exception:
                pass

        boosted_scores.append(score + boost)

    boosted_scores_tensor = torch.tensor(boosted_scores)
    top_results = torch.topk(boosted_scores_tensor, k=min(top_k, len(assessments)))

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
                test_type=a.get("test_type", ""),
                score=score.item()
            )
        )

    if len(recommendations) == 0:
        raise HTTPException(status_code=404, detail="No recommendations found")

    response = RecommendResponse(recommendations=recommendations)
    return JSONResponse(content=jsonable_encoder(response))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
