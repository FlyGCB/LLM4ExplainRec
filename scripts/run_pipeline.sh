#!/bin/bash
set -e
echo "Step 1/4: Preprocessing..."
python data/preprocess.py --category Beauty --kcore 5

echo "Step 2/4: Training SASRec..."
python -m models.sasrec.train --config configs/sasrec.yaml

echo "Step 3/4: Training LightGCN..."
python -m models.lightgcn.train --config configs/lightgcn.yaml

echo "Step 4/4: Evaluating..."
python evaluate.py --split test --k 50

echo "All done. Start server: uvicorn service.main:app --reload"
