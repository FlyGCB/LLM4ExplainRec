"""
End-to-end pipeline test with real item titles.
Usage: python test_pipeline.py
"""
import os
import json
import re
import torch
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "deepseek-ai/DeepSeek-V3"
BASE_URL      = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY       = os.environ["SILICONFLOW_API_KEY"]
RECALL_TOPK   = 20
TEST_USER_IDX = 5

# ── Load data & mappings ──────────────────────────────────────────────────────
print("Loading data...")
df       = pd.read_parquet("data/processed/interactions.parquet")
test_df  = pd.read_parquet("data/processed/test.parquet")
meta_raw = json.load(open("data/processed/meta.json"))
num_users, num_items = meta_raw["num_users"], meta_raw["num_items"]

print("Loading item title mappings...")
int_to_asin   = {int(k): v for k, v in json.load(open("data/processed/int_to_asin.json")).items()}
meta_df       = pd.read_parquet("data/raw/meta.parquet")[["parent_asin", "title", "price"]].dropna(subset=["title"])
asin_to_title = dict(zip(meta_df["parent_asin"], meta_df["title"]))
asin_to_price = dict(zip(meta_df["parent_asin"], meta_df["price"]))

def item_to_title(item_id: int) -> str:
    asin = int_to_asin.get(item_id)
    if asin is None:
        return f"item_{item_id}"
    return asin_to_title.get(asin, f"item_{item_id} ({asin})")

def item_to_price(item_id: int) -> str:
    asin = int_to_asin.get(item_id)
    if asin is None:
        return "?"
    p = asin_to_price.get(asin)
    return f"${p}" if p else "?"

print("Building user sequences...")
df_sorted = df.sort_values(["user_id", "timestamp"])
user_seqs = df_sorted.groupby("user_id")["item_id"].apply(list).to_dict()

test_row  = test_df.iloc[TEST_USER_IDX]
test_uid  = int(test_row["user_id"])
true_item = int(test_row["item_id"])
user_seq  = user_seqs.get(test_uid, [])[-10:]

print(f"\n{'='*60}")
print(f"Test user ID : {test_uid}")
print(f"History (last {len(user_seq)} items):")
for i, iid in enumerate(user_seq):
    print(f"  {i+1}. {item_to_title(iid)}")
print(f"True next item: {item_to_title(true_item)}")
print(f"{'='*60}\n")

# ── SASRec Recall ─────────────────────────────────────────────────────────────
print("Running SASRec recall...")
from models.sasrec.model import SASRec

sasrec = SASRec(num_items=num_items, hidden_size=256, num_heads=4, num_layers=2, max_seq_len=50)
ckpt   = "checkpoints/sasrec_best.pt"
sasrec.load_state_dict(torch.load(ckpt, map_location="cpu"))
sasrec.eval()

MAX_LEN    = 50
seq_tensor = torch.tensor(user_seq[-MAX_LEN:], dtype=torch.long).unsqueeze(0)
pad_len    = MAX_LEN - seq_tensor.shape[1]
if pad_len > 0:
    seq_tensor = torch.cat([torch.zeros(1, pad_len, dtype=torch.long), seq_tensor], dim=1)

with torch.no_grad():
    all_items = torch.arange(1, num_items + 1)
    scores    = sasrec.predict(seq_tensor, all_items).squeeze(0)
    topk_ids  = scores.topk(RECALL_TOPK).indices.tolist()
    topk_ids  = [i + 1 for i in topk_ids]

print(f"  Recalled {len(topk_ids)} candidates")
print(f"  True item in candidates: {true_item in topk_ids}")
print("  Top 5 candidates:")
for iid in topk_ids[:5]:
    print(f"    - {item_to_title(iid)}")

# ── Format prompts ────────────────────────────────────────────────────────────
history_str = "\n".join(
    f"{i+1}. {item_to_title(iid)}" for i, iid in enumerate(user_seq)
)
candidates_str = "\n".join(
    f"ID:{iid} | {item_to_title(iid)} | {item_to_price(iid)}"
    for iid in topk_ids
)

# ── LLM: Causal CoT Intent ────────────────────────────────────────────────────
print("\nRunning Causal CoT intent inference...")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

cot_prompt = f"""You are a professional e-commerce user behavior analyst.

Given a user's recent purchase/browse sequence, infer their causal shopping intent.

User behavior sequence (chronological):
{history_str}

Output ONLY valid JSON (no markdown, no extra text):
{{"goal": "one sentence", "motivation": "one sentence", "constraint": "one sentence", "confidence": "high|medium|low", "reasoning": "2-3 sentences"}}"""

resp = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": cot_prompt}],
    temperature=0.2,
    max_tokens=300,
)
intent_raw = resp.choices[0].message.content.strip()
print(f"  LLM output:\n{intent_raw}\n")

intent_clean = re.sub(r"```(?:json)?|```", "", intent_raw).strip()
try:
    intent = json.loads(intent_clean)
    print(f"  Goal       : {intent.get('goal')}")
    print(f"  Motivation : {intent.get('motivation')}")
    print(f"  Constraint : {intent.get('constraint')}")
except json.JSONDecodeError:
    intent = {"goal": intent_raw, "motivation": "", "constraint": ""}
    print("  WARNING: Could not parse JSON")

# ── LLM: Faithful Rerank + Explain ───────────────────────────────────────────
print("\nRunning faithful reranking + explanation...")

rerank_prompt = f"""You are a faithful e-commerce recommendation engine.

Rules:
1. Every explanation MUST be grounded in the user's actual behavior history.
2. NEVER invent reasons not in the history.
3. Be specific — reference actual items from history.

User intent: {json.dumps(intent, ensure_ascii=False)}

User history:
{history_str}

Candidates:
{candidates_str}

Select top 5 items that best fit the user intent.
Output ONLY a valid JSON array (no markdown):
[{{"rank": 1, "item_id": 123, "title": "...", "explanation": "...", "grounding": "based on your purchase of ..."}}, ...]"""

resp2 = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": rerank_prompt}],
    temperature=0.2,
    max_tokens=600,
)
rerank_raw   = resp2.choices[0].message.content.strip()
rerank_clean = re.sub(r"```(?:json)?|```", "", rerank_raw).strip()

try:
    ranked = json.loads(rerank_clean)
    print("\n=== Final Recommendations ===")
    for item in ranked:
        print(f"  #{item['rank']} {item.get('title', item.get('item_id'))}")
        print(f"     Explanation : {item.get('explanation')}")
        print(f"     Grounding   : {item.get('grounding')}")
        print()
except json.JSONDecodeError:
    print("WARNING: Could not parse rerank JSON")
    print(rerank_raw)

print("✓ Pipeline test complete.")
