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

# 2. Download data
python download_data.py

# 3. Preprocess
python data/preprocess.py

# 4. Train recall models
python -m models.sasrec.train --config configs/sasrec.yaml
python -m models.lightgcn.train --config configs/lightgcn.yaml

# 5. Run evaluation
python batch_evaluate.py

# 6. Test end-to-end pipeline (requires API key in .env)
python test_pipeline.py
```

---

## Experiments

### Dataset

Amazon Product Reviews 2023 — three categories merged (Sports & Outdoors, Cell Phones & Accessories, Clothing Shoes & Jewelry), 5-core filtered, leave-one-out split.

| Split | Users | Items | Interactions |
|---|---|---|---|
| Train | 425,812 | 161,362 | 2,773,820 |
| Val | 425,812 | — | 425,812 |
| Test | 425,812 | — | 425,812 |

### Recall Results

Evaluated on full test set (425,812 users), Recall@50 and NDCG@10:

| Model | Recall@50 | NDCG@10 |
|---|---|---|
| SASRec (sequential) | 0.0443 | 0.0137 |
| LightGCN (collaborative) | **0.0544** | **0.0139** |
| Fusion (SASRec + LightGCN) | 0.0443 | 0.0137 |

LightGCN outperforms SASRec on this dataset due to the multi-category nature of the data — collaborative filtering captures cross-category purchase patterns that sequential models miss.

### Explanation Quality

Evaluated on 100 sampled users. LLM generates natural language explanations grounded in user history; quality measured via BERTScore against self-generated grounding statements.

| Metric | Score |
|---|---|
| BERTScore F1 | **0.8677** |

### Qualitative Example

**User history** (last 7 interactions):
1. Foxelli Carbon Fiber Trekking Poles
2. RAMBO III Hunting Knife
3. Valeo Slimmer Belt
4. Danskin Waist Trimmer Belt
5. Goodyear 29×2.1 MTB Tire
6. ICOCOPRO CO2 Bike Tire Inflator
7. Gorilla Force CO2 Cartridges

**Inferred intent (Causal CoT)**:
- **Goal**: Prepare for outdoor adventure, possibly hiking or mountain biking
- **Motivation**: Fitness and outdoor exploration with durable, multi-terrain gear
- **Constraint**: Lightweight, portable, and multi-functional equipment

**Top recommendation with explanation**:
> *"This bike light is essential for safe mountain biking or night-time outdoor adventures, complementing your recent purchase of the Goodyear MTB tire and CO2 inflator."*
> — Grounded in: Goodyear 29×2.1 MTB Tire + ICOCOPRO CO2 Inflator

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
├── pipeline.py              # End-to-end recall orchestration
├── batch_evaluate.py        # Recall + BERTScore batch evaluation
├── test_pipeline.py         # End-to-end single-user pipeline test
└── scripts/
    ├── download_data.py
    └── run_pipeline.sh
```

---

## References

- He et al., *LightGCN* (SIGIR 2020)
- Kang & McAuley, *SASRec* (ICDM 2018)
- Hou et al., *LLMRank* (RecSys 2023)
- Lin et al., *E4SRec* (2024)
- Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in LLMs* (NeurIPS 2022)
- Zhang et al., *BERTScore: Evaluating Text Generation with BERT* (ICLR 2020)
