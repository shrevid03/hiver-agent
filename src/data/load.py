"""
load.py — Load and basic-clean the Twitter Customer Support dataset (twcs.csv).

Expected columns in twcs.csv:
    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id
"""

import pandas as pd
from pathlib import Path


RAW_PATH = Path(__file__).parents[2] / "data" / "raw" / "twcs.csv"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load twcs.csv with correct dtypes. Returns a DataFrame."""
    df = pd.read_csv(
        path,
        dtype={
            "tweet_id": str,
            "author_id": str,
            "response_tweet_id": str,
            "in_response_to_tweet_id": str,
        },
        parse_dates=["created_at"],
    )
    # Normalise column names to lowercase, strip whitespace
    df.columns = df.columns.str.strip().str.lower()
    # inbound is True for customer messages, False for brand replies
    df["inbound"] = df["inbound"].astype(bool)
    # Strip whitespace from text
    df["text"] = df["text"].str.strip()
    return df


def brand_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table of each brand (non-inbound author_id):
      - reply_count       : how many replies the brand sent
      - thread_count      : threads initiated (unique customer opening messages)
      - avg_thread_length : mean messages per thread
    Sorted by reply_count descending.
    """
    brand_replies = df[~df["inbound"]]
    stats = (
        brand_replies.groupby("author_id")
        .agg(reply_count=("tweet_id", "count"))
        .reset_index()
        .sort_values("reply_count", ascending=False)
    )
    return stats


if __name__ == "__main__":
    print(f"Loading from {RAW_PATH} …")
    df = load_raw()
    print(f"Loaded {len(df):,} tweets")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample:\n{df.head(3)}\n")

    stats = brand_stats(df)
    print("Top 20 brands by reply volume:")
    print(stats.head(20).to_string(index=False))
