"""
01_eda.py — Exploratory analysis to pick a brand.

Run:
    python scripts/01_eda.py

Outputs:
    - Top 30 brands by reply volume (printed)
    - For top 5 brands: sample threads so you can read real conversations
    - A brand_stats.csv in data/processed/

Decision criteria we use to pick a brand:
  1. Enough data (>2000 replies) to have rich RAG context
  2. Diverse intents (not a single-product brand)
  3. Mix of short and long threads (so the escalation task is non-trivial)
  4. Decent reply quality (some brands just say "DM us" for everything — bad for RAG)
"""

import sys
from pathlib import Path
import pandas as pd

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.load import load_raw, brand_stats

PROCESSED = Path(__file__).parents[1] / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def dm_redirect_rate(df: pd.DataFrame, brand_id: str) -> float:
    """Fraction of brand replies that are just DM redirects."""
    brand_replies = df[(df["author_id"] == brand_id) & (~df["inbound"])]
    if len(brand_replies) == 0:
        return 0.0
    dm_keywords = ["dm us", "direct message", "send us a dm", "please dm", "message us directly"]
    is_dm = brand_replies["text"].str.lower().apply(
        lambda t: any(kw in t for kw in dm_keywords)
    )
    return is_dm.mean()


def avg_reply_length(df: pd.DataFrame, brand_id: str) -> float:
    brand_replies = df[(df["author_id"] == brand_id) & (~df["inbound"])]
    return brand_replies["text"].str.len().mean()


def thread_length_dist(df: pd.DataFrame, brand_id: str) -> dict:
    """Rough thread-length distribution (using response chains)."""
    brand_replies = df[(df["author_id"] == brand_id) & (~df["inbound"])]
    return {
        "total_replies": len(brand_replies),
        "unique_threads_approx": brand_replies["in_response_to_tweet_id"].nunique(),
    }


def sample_threads(df: pd.DataFrame, brand_id: str, n: int = 3) -> None:
    """Print n sample conversation snippets for a brand."""
    brand_reply_ids = set(
        df[(df["author_id"] == brand_id) & (~df["inbound"])]["in_response_to_tweet_id"]
        .dropna()
        .astype(str)
    )
    # Find some customer messages that got a response
    customer_msgs = df[
        df["tweet_id"].astype(str).isin(brand_reply_ids) & df["inbound"]
    ].head(n)

    for _, row in customer_msgs.iterrows():
        cid = str(row["tweet_id"])
        print(f"\n  [Customer] {row['text'][:200]}")
        # Find brand reply
        reply = df[
            (df["in_response_to_tweet_id"].astype(str) == cid) &
            (df["author_id"] == brand_id)
        ]
        if not reply.empty:
            print(f"  [{brand_id}] {reply.iloc[0]['text'][:200]}")
        print("  " + "-"*60)


def main():
    print("Loading dataset … (this may take ~30s for the full 3M file)")
    df = load_raw()
    print(f"Loaded {len(df):,} tweets\n")

    stats = brand_stats(df)
    print("=" * 55)
    print(f"{'Brand':30s} {'Replies':>10s}")
    print("=" * 55)
    for _, row in stats.head(30).iterrows():
        print(f"{row['author_id']:30s} {row['reply_count']:>10,}")

    # Deep-dive top 8 brands
    print("\n\n=== DEEP DIVE: top 8 brands ===\n")
    top_brands = stats.head(8)["author_id"].tolist()

    rows = []
    for brand_id in top_brands:
        dm_rate = dm_redirect_rate(df, brand_id)
        avg_len = avg_reply_length(df, brand_id)
        dist = thread_length_dist(df, brand_id)
        rows.append({
            "brand": brand_id,
            "total_replies": dist["total_replies"],
            "unique_threads_approx": dist["unique_threads_approx"],
            "dm_redirect_rate": round(dm_rate, 3),
            "avg_reply_chars": round(avg_len, 0),
        })

    detail = pd.DataFrame(rows)
    print(detail.to_string(index=False))
    detail.to_csv(PROCESSED / "brand_stats.csv", index=False)
    print(f"\nSaved → data/processed/brand_stats.csv")

    # Sample conversations from top 3 brands
    for brand_id in top_brands[:3]:
        print(f"\n{'='*55}")
        print(f"Sample conversations: {brand_id}")
        print('='*55)
        sample_threads(df, brand_id, n=3)

    print("\n\n=== RECOMMENDATION ===")
    # Score each brand: penalise high DM redirect rates, reward reply volume
    detail["score"] = (
        detail["total_replies"] / detail["total_replies"].max()
        - 2 * detail["dm_redirect_rate"]
    )
    best = detail.sort_values("score", ascending=False).iloc[0]
    print(f"Best pick based on data: {best['brand']}")
    print(f"  Replies: {best['total_replies']:,}")
    print(f"  DM redirect rate: {best['dm_redirect_rate']:.1%}  (lower is better)")
    print(f"  Avg reply length: {best['avg_reply_chars']:.0f} chars")
    print("\nFinal call is yours — see the table above and samples.")


if __name__ == "__main__":
    main()
