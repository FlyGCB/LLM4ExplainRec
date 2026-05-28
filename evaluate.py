"""
Evaluation script: Recall@K, NDCG@K, BERTScore for explanations.
Usage: python evaluate.py --split test
"""
import json
import torch
import argparse
import numpy as np
import pandas as pd
from bert_score import score as bert_score
from pipeline import build_pipeline
from models.ranker_explainer import RankerExplainer
import os


def ndcg_at_k(ranked_list, true_item, k):
    if true_item not in ranked_list[:k]:
        return 0.0
    rank = ranked_list[:k].index(true_item) + 1
    return 1.0 / np.log2(rank + 1)


def evaluate_recall(pipeline, test_data, k=50):
    hits, ndcg_sum = 0, 0.0
    for uid, true_item in test_data:
        result = pipeline.recall(uid, topk=k)
        ranked_ids = [int(c["item_id"]) for c in result.candidates]
        hits += int(true_item in ranked_ids)
        ndcg_sum += ndcg_at_k(ranked_ids, true_item, k)
    n = len(test_data)
    return {"recall": hits / n, "ndcg": ndcg_sum / n}


def evaluate_explanations(llm, pipeline, test_data, n_samples=100):
    """Sample n users, generate explanations, compute BERTScore."""
    hyps, refs = [], []
    for uid, true_item in test_data[:n_samples]:
        result = pipeline.recall(uid)
        if not result.history:
            continue
        intent = llm.infer_intent(result.history)
        ranked = llm.rerank_and_explain(result.history, intent, result.candidates, topk=5)
        for item in ranked[:1]:
            hyps.append(item.get("explanation", ""))
            refs.append(item.get("grounding", ""))
    P, R, F1 = bert_score(hyps, refs, lang="en", verbose=False)
    return {"bertscore_f1": F1.mean().item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--explain_samples", type=int, default=100)
    args = parser.parse_args()

    pipe = build_pipeline()
    test_df = pd.read_parquet(f"data/processed/{args.split}.parquet")
    test_data = list(test_df.itertuples(index=False, name=None))

    print("=== Recall Metrics ===")
    rec_metrics = evaluate_recall(pipe, test_data, k=args.k)
    for k, v in rec_metrics.items():
        print(f"  {k}@{args.k}: {v:.4f}")

    if os.environ.get("OPENAI_API_KEY"):
        print("\n=== Explanation Quality (BERTScore) ===")
        llm = RankerExplainer(api_key=os.environ["OPENAI_API_KEY"])
        exp_metrics = evaluate_explanations(llm, pipe, test_data, args.explain_samples)
        for k, v in exp_metrics.items():
            print(f"  {k}: {v:.4f}")
