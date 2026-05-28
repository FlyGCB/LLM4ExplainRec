"""
LLM-based ranker and explainer.
Handles: causal intent inference, faithful reranking, conversational feedback.
"""
import json
import re
from pathlib import Path
from openai import OpenAI

PROMPT_DIR = Path("prompt")


def _load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text()


def _call_llm(client: OpenAI, prompt: str, model: str = "gpt-4o-mini",
              temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def _parse_json(text: str) -> dict | list:
    """Strip markdown fences if present and parse JSON."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
    return json.loads(text)


class RankerExplainer:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model  = model
        self._cot_tmpl    = _load_prompt("causal_cot.txt")
        self._faith_tmpl  = _load_prompt("faithfulness.txt")
        self._conv_tmpl   = _load_prompt("conversational.txt")

    def infer_intent(self, user_history: list[str]) -> dict:
        """
        Step 1: Causal CoT — infer Goal / Motivation / Constraint.
        user_history: list of item titles in chronological order.
        """
        history_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(user_history))
        prompt = self._cot_tmpl.replace("{user_history}", history_str)
        raw = _call_llm(self.client, prompt, self.model)
        return _parse_json(raw)

    def rerank_and_explain(self, user_history: list[str],
                           user_intent: dict,
                           candidates: list[dict],
                           topk: int = 10) -> list[dict]:
        """
        Step 2: Faithful reranking + explanation generation.
        candidates: [{"item_id": ..., "title": ..., "category": ..., "price": ...}]
        """
        history_str    = "\n".join(f"{i+1}. {item}" for i, item in enumerate(user_history))
        intent_str     = json.dumps(user_intent, ensure_ascii=False)
        candidates_str = "\n".join(
            f"{c['item_id']} | {c['title']} | {c['category']} | ${c.get('price', '?')}"
            for c in candidates
        )
        prompt = (self._faith_tmpl
                  .replace("{user_intent}", intent_str)
                  .replace("{user_history}", history_str)
                  .replace("{candidates}", candidates_str))
        raw = _call_llm(self.client, prompt, self.model)
        results = _parse_json(raw)
        return results[:topk]

    def handle_feedback(self, current_intent: dict, user_feedback: str) -> dict:
        """
        Step 3 (conversational): Parse user feedback and update intent.
        Returns updated intent + any new filters.
        """
        intent_str = json.dumps(current_intent, ensure_ascii=False)
        prompt = (self._conv_tmpl
                  .replace("{current_intent}", intent_str)
                  .replace("{user_feedback}", user_feedback))
        raw = _call_llm(self.client, prompt, self.model)
        return _parse_json(raw)
