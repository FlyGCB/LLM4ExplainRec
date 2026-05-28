from pydantic import BaseModel
from typing import Optional

class RecRequest(BaseModel):
    user_id: str
    feedback: Optional[str] = None   # conversational turn

class RecItem(BaseModel):
    item_id: str
    title: str
    rank: int
    explanation: str
    grounding: str

class RecResponse(BaseModel):
    user_id: str
    intent: dict
    items: list[RecItem]
    cached: bool = False
