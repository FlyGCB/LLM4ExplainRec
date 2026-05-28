"""
Week 1 - Data preprocessing pipeline.
Downloads Amazon Reviews dataset, applies 5-core filtering,
sorts by timestamp, and performs train/val/test split.
"""
import json
import gzip
import pandas as pd
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")


def load_amazon(category: str = "Beauty") -> pd.DataFrame:
    path = RAW_DIR / "reviews.parquet"
    df = pd.read_parquet(path)
    return df[["user_id", "parent_asin", "timestamp", "rating"]].rename(
        columns={"parent_asin": "item_id"}
    )


def kcore_filter(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Iteratively remove users/items with fewer than k interactions."""
    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        before = len(df)
        df = df[df["user_id"].isin(user_counts[user_counts >= k].index)]
        df = df[df["item_id"].isin(item_counts[item_counts >= k].index)]
        if len(df) == before:
            break
    return df


def build_id_maps(df: pd.DataFrame):
    users = {u: i for i, u in enumerate(df["user_id"].unique())}
    items = {it: i for i, it in enumerate(df["item_id"].unique())}
    df["user_id"] = df["user_id"].map(users)
    df["item_id"] = df["item_id"].map(items)
    return df, users, items


def split_leave_one_out(df: pd.DataFrame):
    """
    Sort each user's history by timestamp.
    Test  = last interaction per user
    Val   = second-to-last
    Train = everything else
    """
    df = df.sort_values(["user_id", "timestamp"])
    grouped = df.groupby("user_id")

    train, val, test = [], [], []
    for uid, group in grouped:
        items = group["item_id"].tolist()
        if len(items) < 3:
            train.extend(items)
            continue
        train.extend(items[:-2])
        val.append((uid, items[-2]))
        test.append((uid, items[-1]))

    return train, val, test


def main(category: str = "Beauty", k: int = 5):
    print(f"Loading {category}...")
    df = load_amazon(category)
    print(f"  Raw: {len(df):,} interactions")

    df = kcore_filter(df, k=k)
    print(f"  After {k}-core: {len(df):,} interactions, "
          f"{df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")

    df, user_map, item_map = build_id_maps(df)
    train, val, test = split_leave_one_out(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "interactions.parquet", index=False)
    pd.DataFrame(val, columns=["user_id", "item_id"]).to_parquet(OUT_DIR / "val.parquet", index=False)
    pd.DataFrame(test, columns=["user_id", "item_id"]).to_parquet(OUT_DIR / "test.parquet", index=False)

    meta = {"num_users": len(user_map), "num_items": len(item_map)}
    pd.Series(meta).to_json(OUT_DIR / "meta.json")
    print(f"  Saved to {OUT_DIR}")
    print(f"  Train interactions: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Beauty")
    parser.add_argument("--kcore", type=int, default=5)
    args = parser.parse_args()
    main(args.category, args.kcore)
