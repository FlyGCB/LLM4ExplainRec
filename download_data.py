from datasets import load_dataset
import pandas as pd
from pathlib import Path

Path("data/raw").mkdir(parents=True, exist_ok=True)

print("download reviews...")
reviews = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Sports_and_Outdoors",
    split="full",
    trust_remote_code=True
)
pd.DataFrame(reviews).to_parquet("data/raw/reviews.parquet")
print(f"  finished, {len(reviews)} review records")

print("download product metadata...")
meta = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_meta_Sports_and_Outdoors",
    split="full",
    trust_remote_code=True
)
pd.DataFrame(meta).to_parquet("data/raw/meta.parquet")
print(f"  finished, {len(meta)} product records")

print("all downloads completed, files saved to data/raw/")