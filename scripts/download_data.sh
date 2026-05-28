#!/bin/bash
# Download Amazon Beauty 5-core dataset
set -e
mkdir -p data/raw
echo "Downloading Amazon Reviews (Beauty, 5-core)..."
wget -q -O data/raw/reviews_Beauty_5.json.gz \
  "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz"
echo "Done. Run: python data/preprocess.py --category Beauty"
