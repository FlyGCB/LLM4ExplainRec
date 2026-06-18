"""
Batch evaluation script.
Metrics:
  - Recall@K, NDCG@K  (recall layer: SASRec / LightGCN / Fusion)
  - BERTScore F1       (explanation quality, sampled)

Index convention:
  - interactions.parquet / test.parquet: item_id is 0-indexed (0 ~ num_items-1)
  - SASRec predict(seq, arange(0, num_items)): scores[i] = score for item i (0-indexed)
  - LightGCN item_embs: 0-indexed
  - All comparisons use true_item directly (0-indexed)

Usage: python batch_evaluate.py
"""
import os
import json
import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
RECALL_K        = 50
NDCG_K          = 10
EXPLAIN_SAMPLES = 100
MODEL_NAME      = "deepseek-ai/DeepSeek-V3"
BASE_URL        = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY         = os.environ.get("SILICONFLOW_API_KEY", "")
BATCH_SIZE      = 512

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df       = pd.read_parquet("data/processed/interactions.parquet")
test_df  = pd.read_parquet("data/processed/test.parquet")
meta_raw = json.load(open("data/processed/meta.json"))
num_users, num_items = meta_raw["num_users"], meta_raw["num_items"]

int_to_asin   = {int(k): v for k, v in json.load(open("data/processed/int_to_asin.json")).items()}
meta_df       = pd.read_parquet("data/raw/meta.parquet")[["parent_asin","title","price"]].dropna(subset=["title"])
asin_to_title = dict(zip(meta_df["parent_asin"], meta_df["title"]))

def item_to_title(item_id: int) -> str:
    asin = int_to_asin.get(item_id)
    if not asin:
        return f"item_{item_id}"
    return asin_to_title.get(asin, f"item_{item_id}")

df_sorted = df.sort_values(["user_id", "timestamp"])
user_seqs = df_sorted.groupby("user_id")["item_id"].apply(list).to_dict()
test_data = list(test_df.itertuples(index=False, name=None))
print(f"  Test users: {len(test_data):,}")

# ── Load models ───────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

from models.sasrec.model import SASRec
from models.lightgcn.model import LightGCN
from models.lightgcn.train import build_edge_index

sasrec = SASRec(num_items=num_items, hidden_size=256, num_heads=4, num_layers=2, max_seq_len=50).to(device)
sasrec.load_state_dict(torch.load("checkpoints/sasrec_best.pt", map_location=device))
sasrec.eval()

lightgcn = LightGCN(num_users, num_items, embedding_dim=64, num_layers=3).to(device)
lightgcn.load_state_dict(torch.load("checkpoints/lightgcn_best.pt", map_location=device))
lightgcn.eval()

edge_index = build_edge_index(df, num_users, device)
print("  Models loaded.")

# ── Metrics ───────────────────────────────────────────────────────────────────
def ndcg_at_k(ranked_list, true_item, k):
    if true_item not in ranked_list[:k]:
        return 0.0
    rank = ranked_list[:k].index(true_item) + 1
    return 1.0 / np.log2(rank + 1)

# ── SASRec eval ───────────────────────────────────────────────────────────────
def eval_sasrec_batch(test_data, k=50):
    hits, ndcg_sum = 0, 0.0
    MAX_LEN = 50
    # Use arange(0, num_items) so scores[i] = score for item i (0-indexed)
    all_items = torch.arange(0, num_items, device=device)

    for start in tqdm(range(0, len(test_data), BATCH_SIZE), desc="SASRec eval"):
        batch = test_data[start:start+BATCH_SIZE]
        seqs, true_items = [], []
        for uid, true_item in batch:
            seq = user_seqs.get(uid, [])[-MAX_LEN:]
            pad = [0] * (MAX_LEN - len(seq))
            seqs.append(pad + seq)
            true_items.append(true_item)

        seq_tensor = torch.tensor(seqs, dtype=torch.long).to(device)

        with torch.no_grad():
            scores = sasrec.predict(seq_tensor, all_items)   # (B, num_items)
            topk   = scores.topk(k, dim=-1).indices          # 0-indexed

        for i, true_item in enumerate(true_items):
            ranked    = topk[i].tolist()
            hits     += int(true_item in ranked)
            ndcg_sum += ndcg_at_k(ranked, true_item, NDCG_K)

    n = len(test_data)
    return {"Recall@50": hits / n, "NDCG@10": ndcg_sum / n}

# ── LightGCN eval ─────────────────────────────────────────────────────────────
def eval_lightgcn_batch(test_data, k=50):
    hits, ndcg_sum = 0, 0.0

    with torch.no_grad():
        user_embs, item_embs = lightgcn(edge_index)

    for start in tqdm(range(0, len(test_data), BATCH_SIZE), desc="LightGCN eval"):
        batch = test_data[start:start+BATCH_SIZE]
        uids  = torch.tensor([u for u, _ in batch], dtype=torch.long, device=device)

        with torch.no_grad():
            scores = user_embs[uids] @ item_embs.T   # (B, num_items)
            topk   = scores.topk(k, dim=-1).indices   # 0-indexed

        for i, (_, true_item) in enumerate(batch):
            ranked    = topk[i].tolist()
            hits     += int(true_item in ranked)
            ndcg_sum += ndcg_at_k(ranked, true_item, NDCG_K)

    n = len(test_data)
    return {"Recall@50": hits / n, "NDCG@10": ndcg_sum / n}

# ── Fusion eval ───────────────────────────────────────────────────────────────
def eval_fusion_batch(test_data, k=50):
    """Both SASRec and LightGCN produce 0-indexed topk, merge directly."""
    hits, ndcg_sum = 0, 0.0
    MAX_LEN = 50
    all_items = torch.arange(0, num_items, device=device)

    with torch.no_grad():
        user_embs, item_embs = lightgcn(edge_index)

    for start in tqdm(range(0, len(test_data), BATCH_SIZE), desc="Fusion eval"):
        batch = test_data[start:start+BATCH_SIZE]
        seqs, true_items = [], []
        for uid, true_item in batch:
            seq = user_seqs.get(uid, [])[-MAX_LEN:]
            pad = [0] * (MAX_LEN - len(seq))
            seqs.append(pad + seq)
            true_items.append(true_item)

        seq_tensor = torch.tensor(seqs, dtype=torch.long).to(device)
        uids       = torch.tensor([u for u, _ in batch], dtype=torch.long, device=device)

        with torch.no_grad():
            s_scores = sasrec.predict(seq_tensor, all_items)
            s_topk   = s_scores.topk(k, dim=-1).indices.tolist()   # 0-indexed

            g_scores = user_embs[uids] @ item_embs.T
            g_topk   = g_scores.topk(k, dim=-1).indices.tolist()   # 0-indexed

        for i, true_item in enumerate(true_items):
            merged    = list(dict.fromkeys(s_topk[i] + g_topk[i]))[:k]
            hits     += int(true_item in merged)
            ndcg_sum += ndcg_at_k(merged, true_item, NDCG_K)

    n = len(test_data)
    return {"Recall@50": hits / n, "NDCG@10": ndcg_sum / n}

# ── Run recall evaluation ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("RECALL EVALUATION")
print("="*60)

results = {}

print("\n[1/3] SASRec")
results["SASRec"] = eval_sasrec_batch(test_data, k=RECALL_K)
print(f"  Recall@{RECALL_K}: {results['SASRec']['Recall@50']:.4f}  NDCG@{NDCG_K}: {results['SASRec']['NDCG@10']:.4f}")

print("\n[2/3] LightGCN")
results["LightGCN"] = eval_lightgcn_batch(test_data, k=RECALL_K)
print(f"  Recall@{RECALL_K}: {results['LightGCN']['Recall@50']:.4f}  NDCG@{NDCG_K}: {results['LightGCN']['NDCG@10']:.4f}")

print("\n[3/3] Fusion (SASRec + LightGCN)")
results["Fusion"] = eval_fusion_batch(test_data, k=RECALL_K)
print(f"  Recall@{RECALL_K}: {results['Fusion']['Recall@50']:.4f}  NDCG@{NDCG_K}: {results['Fusion']['NDCG@10']:.4f}")

# ── BERTScore evaluation ──────────────────────────────────────────────────────
if API_KEY:
    print("\n" + "="*60)
    print(f"EXPLANATION QUALITY (BERTScore, {EXPLAIN_SAMPLES} users)")
    print("="*60)

    client     = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    hyps, refs = [], []

    for uid, true_item in tqdm(test_data[:EXPLAIN_SAMPLES], desc="Generating explanations"):
        seq = user_seqs.get(uid, [])[-10:]
        if len(seq) < 3:
            continue

        history_str = "\n".join(f"{i+1}. {item_to_title(iid)}" for i, iid in enumerate(seq))

        cot_prompt = f"""You are an e-commerce analyst. Given the user's purchase sequence, output ONLY valid JSON:
{{"goal": "...", "motivation": "...", "constraint": "..."}}

User sequence:
{history_str}"""

        try:
            r1         = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": cot_prompt}],
                temperature=0.2, max_tokens=200,
            )
            intent_raw = re.sub(r"```(?:json)?|```", "", r1.choices[0].message.content).strip()
            intent     = json.loads(intent_raw)
            intent_str = json.dumps(intent, ensure_ascii=False)
        except Exception:
            intent_str = '{"goal": "outdoor activities", "motivation": "fitness", "constraint": "lightweight gear"}'

        MAX_LEN    = 50
        seq_tensor = torch.tensor(seq[-MAX_LEN:], dtype=torch.long).unsqueeze(0).to(device)
        pad_len    = MAX_LEN - seq_tensor.shape[1]
        if pad_len > 0:
            seq_tensor = torch.cat([torch.zeros(1, pad_len, dtype=torch.long, device=device), seq_tensor], dim=1)

        with torch.no_grad():
            all_items_llm = torch.arange(0, num_items, device=device)
            scores        = sasrec.predict(seq_tensor, all_items_llm).squeeze(0)
            topk_ids      = scores.topk(10).indices.tolist()  # 0-indexed

        candidates_str = "\n".join(f"ID:{iid} | {item_to_title(iid)}" for iid in topk_ids)

        rerank_prompt = f"""You are a faithful recommendation engine.
User intent: {intent_str}
User history: {history_str}
Candidates: {candidates_str}

Output ONLY a JSON array with 1 item:
[{{"rank":1,"item_id":0,"title":"...","explanation":"...","grounding":"based on your purchase of ..."}}]"""

        try:
            r2     = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": rerank_prompt}],
                temperature=0.2, max_tokens=300,
            )
            raw    = re.sub(r"```(?:json)?|```", "", r2.choices[0].message.content).strip()
            ranked = json.loads(raw)
            hyps.append(ranked[0].get("explanation", ""))
            refs.append(ranked[0].get("grounding", ""))
        except Exception:
            continue

    if hyps:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(hyps, refs, lang="en", verbose=False)
        bert_f1  = F1.mean().item()
        print(f"  BERTScore F1: {bert_f1:.4f}  (n={len(hyps)})")
        results["BERTScore_F1"] = bert_f1
    else:
        print("  No explanations generated.")
else:
    print("\nSkipping BERTScore (no API key found in .env)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Model':<20} {'Recall@50':>10} {'NDCG@10':>10}")
print("-"*42)
for model, m in results.items():
    if isinstance(m, dict):
        print(f"{model:<20} {m.get('Recall@50', 0):>10.4f} {m.get('NDCG@10', 0):>10.4f}")
if "BERTScore_F1" in results:
    print(f"\n{'BERTScore F1':<20} {results['BERTScore_F1']:>10.4f}")

Path("outputs").mkdir(exist_ok=True)
with open("outputs/eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to outputs/eval_results.json")
