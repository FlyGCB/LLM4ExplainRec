"""
FastAPI serving layer.
GET  /rec/{user_id}          — get recommendations (cache-first)
POST /rec/{user_id}/feedback — conversational update
"""
import os
from fastapi import FastAPI, HTTPException
from service.schemas import RecRequest, RecResponse, RecItem
from service.cache import RecommendationCache
from models.ranker_explainer import RankerExplainer
from pipeline import build_pipeline

app    = FastAPI(title="LLM4ExplainRec")
cache  = RecommendationCache()
llm    = RankerExplainer(api_key=os.environ["OPENAI_API_KEY"])
pipes  = build_pipeline()   # loads recall models


@app.get("/rec/{user_id}", response_model=RecResponse)
def get_recommendations(user_id: str):
    # 1. Try cache
    cached_items = cache.get_rec(user_id)
    cached_intent = cache.get_intent(user_id)
    if cached_items and cached_intent:
        return RecResponse(user_id=user_id, intent=cached_intent,
                           items=[RecItem(**i) for i in cached_items], cached=True)

    # 2. Recall
    history, candidates = pipes.recall(user_id)
    if not history:
        raise HTTPException(404, "User not found or no history")

    # 3. LLM intent + rerank
    intent = llm.infer_intent(history)
    ranked = llm.rerank_and_explain(history, intent, candidates)

    # 4. Cache and return
    cache.set_intent(user_id, intent)
    cache.set_rec(user_id, ranked)

    return RecResponse(user_id=user_id, intent=intent,
                       items=[RecItem(**i) for i in ranked], cached=False)


@app.post("/rec/{user_id}/feedback")
def handle_feedback(user_id: str, req: RecRequest):
    current_intent = cache.get_intent(user_id) or {}
    result = llm.handle_feedback(current_intent, req.feedback)
    cache.set_intent(user_id, result["updated_intent"])
    cache.invalidate(user_id)   # force re-rank on next request
    return {"status": "updated", "new_intent": result["updated_intent"],
            "filters": result.get("new_filters", [])}
