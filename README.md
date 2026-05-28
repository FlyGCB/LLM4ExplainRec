# LLM4ExplainRec

**Causal & Faithful LLM-Driven Explainable Recommendation**

A RecSys/WWW-level research project targeting industrial e-commerce recommendation —
directly aligned with Pinduoduo / Temu's large-model application engineering stack.

---

## Motivation

Most recommendation systems are black boxes: they return a list, but never explain *why*.
This project proposes an end-to-end explainable recommendation framework that addresses two core research questions:

1. **Can we infer *causal* user intent** (goal + motivation + constraint) from raw behavior sequences using LLM chain-of-thought reasoning?
2. **Can we generate *faithful* explanations** (grounded strictly in user history) that avoid hallucination?

---

## Framework

```
Behavior Sequence
      │
      ▼
┌─────────────────────────────────┐
│  Multi-Channel Recall           │
│  SASRec  ╋  LightGCN            │
│  (sequence)  (collaborative)    │
└──────────────┬──────────────────┘
               │ Top-50 candidates
               ▼
┌─────────────────────────────────┐
│  Causal CoT Intent Inference    │  ← Core Innovation
│  Goal → Motivation → Constraint │
└──────────────┬──────────────────┘
               │ Structured intent JSON
               ▼
┌─────────────────────────────────┐
│  Faithful Reranking + Explain   │
│  Faithfulness constraint prompt │
│  BERTScore quality evaluation   │
└──────────────┬──────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
  Offline batch     Online API
  pre-generation    FastAPI + Redis
  (nightly cron)    (<200ms latency)
       │
       ▼
  Conversational feedback loop
  (constraint / preference / explain)
```

---

## Key Innovations

| Innovation | Description |
|---|---|
| **Causal Explainability** | 3-layer CoT: Goal → Motivation → Constraint, not correlation |
| **Faithfulness Constraint** | Hard-grounding rule in prompt; measured via BERTScore |
| **Offline/Online Decoupling** | LLM runs offline nightly; Redis serves explanations in <200ms |
| **Conversational Feedback** | 3-type feedback parser (constraint / preference / explanation) |

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data & run full pipeline
bash scripts/run_pipeline.sh

# 3. Start API server
export OPENAI_API_KEY=sk-...
uvicorn service.main:app --reload

# 4. Try it
curl http://localhost:8000/rec/user_123
```

---

## Experiments

**Dataset**: Amazon Product Reviews — Beauty (5-core)
- ~198K interactions | 22K users | 12K items

**Baselines**: MF · LightGCN · SASRec · LLMRank · E4SRec

**Metrics**:
- Recommendation: Recall@50, NDCG@50
- Explanation quality: BERTScore-F1, Faithfulness Score (GPT-4 judge)

```bash
python evaluate.py --split test --k 50 --explain_samples 200
```

---

## Project Structure

```
LLM4ExplainRec/
├── data/
│   ├── preprocess.py        # 5-core filter, leave-one-out split
│   └── dataset.py           # PyTorch Dataset for SASRec / LightGCN
├── models/
│   ├── sasrec/              # Self-attentive sequential recall
│   ├── lightgcn/            # Graph-based collaborative recall
│   ├── distill/             # LoRA fine-tune for deployment (optional)
│   └── ranker_explainer.py  # LLM reranking + explanation generation
├── prompt/
│   ├── causal_cot.txt       # Goal-Motivation-Constraint CoT prompt
│   ├── faithfulness.txt     # Faithfulness-constrained rerank prompt
│   └── conversational.txt   # Feedback parsing prompt
├── service/
│   ├── main.py              # FastAPI endpoints
│   ├── cache.py             # Redis cache layer
│   └── schemas.py           # Pydantic request/response models
├── configs/                 # YAML hyperparameters
├── notebooks/               # EDA + evaluation analysis
├── pipeline.py              # End-to-end recall orchestration
├── evaluate.py              # Recall + BERTScore evaluation
└── scripts/
    ├── download_data.sh
    └── run_pipeline.sh
```

---

## References

- He et al., *LightGCN* (SIGIR 2020)
- Kang & McAuley, *SASRec* (ICDM 2018)
- Hou et al., *LLMRank* (RecSys 2023)
- Lin et al., *E4SRec* (2024)
- Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in LLMs* (NeurIPS 2022)
