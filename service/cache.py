"""
Redis cache for pre-generated explanations and user intent.
Keys:
  intent:{user_id}  → JSON string of causal intent
  rec:{user_id}     → JSON string of top-10 ranked items + explanations
"""
import json
import redis

class RecommendationCache:
    def __init__(self, host: str = "localhost", port: int = 6379,
                 ttl_seconds: int = 86400):
        self.r   = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl_seconds

    def get_intent(self, user_id: str) -> dict | None:
        val = self.r.get(f"intent:{user_id}")
        return json.loads(val) if val else None

    def set_intent(self, user_id: str, intent: dict):
        self.r.setex(f"intent:{user_id}", self.ttl, json.dumps(intent))

    def get_rec(self, user_id: str) -> list | None:
        val = self.r.get(f"rec:{user_id}")
        return json.loads(val) if val else None

    def set_rec(self, user_id: str, items: list):
        self.r.setex(f"rec:{user_id}", self.ttl, json.dumps(items))

    def invalidate(self, user_id: str):
        self.r.delete(f"intent:{user_id}", f"rec:{user_id}")
