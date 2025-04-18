import json
from sentence_transformers import SentenceTransformer, util
import torch

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

def recommend(query, top_k=10):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, assessment_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(assessments)))
    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        a = assessments[idx]
        results.append({
            "name": a["name"],
            "url": a["url"],
            "remote_testing": a.get("remote_testing", "Unknown"),
            "adaptive_irt": a.get("adaptive_irt", "Unknown"),
            "duration": a.get("duration", ""),
            "test_type": a.get("test_type", ""),
            "score": score.item()
        })
    return results

if __name__ == "__main__":
    queries = [
        "computer"
    ]
    for q in queries:
        print(f"Query: {q}")
        recs = recommend(q)
        for i, r in enumerate(recs, 1):
            print(f"{i}. {r['name']} (Score: {r['score']:.4f})")
            print(f"   URL: {r['url']}")
            print(f"   Remote Testing: {r['remote_testing']}, Adaptive/IRT: {r['adaptive_irt']}, Duration: {r['duration']}, Test Type: {r['test_type']}")
        print("\n" + "="*50 + "\n")
