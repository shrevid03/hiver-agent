"""
reconstruct.py — Rebuild multi-turn conversation threads from twcs.csv.

Thread structure in the dataset:
  - Each tweet has an optional `in_response_to_tweet_id` (its parent).
  - `response_tweet_id` is the child tweet that replied to this one.
  - A thread root is a customer tweet with no parent, or whose parent
    isn't in the dataset.

Output per thread (dict):
  {
    "thread_id":       str,          # tweet_id of the root message
    "brand":           str,          # author_id of the brand (non-inbound author)
    "messages": [
        {"role": "customer"|"brand", "text": str, "tweet_id": str, "created_at": str}
    ],
    "resolved":        bool,         # True if brand sent ≥1 reply
    "turns":           int,          # total message count
  }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.data.load import load_raw


PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


def _build_index(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Returns:
      tweet_by_id : {tweet_id -> row dict}
      children    : {tweet_id -> list[tweet_id]}  (parent -> children)
    """
    tweet_by_id: dict[str, dict] = {}
    children: dict[str, list[str]] = {}

    for row in df.itertuples(index=False):
        tid = str(row.tweet_id)
        tweet_by_id[tid] = {
            "tweet_id": tid,
            "author_id": row.author_id,
            "inbound": row.inbound,
            "text": row.text,
            "created_at": str(row.created_at),
            "parent_id": str(row.in_response_to_tweet_id)
            if pd.notna(row.in_response_to_tweet_id)
            else None,
        }
        children.setdefault(tid, [])
        parent = str(row.in_response_to_tweet_id) if pd.notna(row.in_response_to_tweet_id) else None
        if parent:
            children.setdefault(parent, []).append(tid)

    return tweet_by_id, children


def _walk_thread(
    root_id: str,
    tweet_by_id: dict,
    children: dict,
) -> list[dict]:
    """BFS walk of a thread starting from root_id. Returns ordered messages."""
    messages = []
    queue = [root_id]
    visited = set()
    while queue:
        tid = queue.pop(0)
        if tid in visited or tid not in tweet_by_id:
            continue
        visited.add(tid)
        tw = tweet_by_id[tid]
        messages.append({
            "tweet_id": tw["tweet_id"],
            "role": "customer" if tw["inbound"] else "brand",
            "text": tw["text"],
            "created_at": tw["created_at"],
        })
        for child_id in children.get(tid, []):
            queue.append(child_id)
    return messages


def reconstruct_threads(
    df: pd.DataFrame,
    brand_id: str,
    max_threads: int | None = None,
) -> Iterator[dict]:
    """
    Reconstruct conversation threads for a given brand.

    Args:
        df:          Full raw DataFrame (all brands).
        brand_id:    The author_id of the brand's support handle.
        max_threads: Cap the number of threads (useful for sampling).

    Yields:
        Thread dicts (see module docstring for schema).
    """
    # Keep only tweets that are part of this brand's conversations:
    # either sent by the brand or sent TO the brand (inbound tweets
    # where the brand later replied).
    brand_tweet_ids = set(df[df["author_id"] == brand_id]["tweet_id"].astype(str))

    # Find customer tweets the brand replied to
    brand_rows = df[df["author_id"] == brand_id]
    parent_ids = set(
        brand_rows["in_response_to_tweet_id"]
        .dropna()
        .astype(str)
    )

    # Slice: only tweets that are either brand tweets or ancestors of brand tweets
    relevant_ids = brand_tweet_ids | parent_ids
    # Also include parents of parents (full thread chains)
    # We do two hops of ancestor expansion for efficiency
    for _ in range(5):
        extra_parents = set(
            df[df["tweet_id"].astype(str).isin(relevant_ids)]["in_response_to_tweet_id"]
            .dropna()
            .astype(str)
        )
        before = len(relevant_ids)
        relevant_ids |= extra_parents
        if len(relevant_ids) == before:
            break

    df_slice = df[df["tweet_id"].astype(str).isin(relevant_ids)].copy()
    tweet_by_id, children = _build_index(df_slice)

    # Thread roots: inbound tweets with no parent in our slice
    roots = [
        tid for tid, tw in tweet_by_id.items()
        if tw["inbound"] and (tw["parent_id"] is None or tw["parent_id"] not in tweet_by_id)
    ]

    count = 0
    for root_id in roots:
        if max_threads and count >= max_threads:
            break
        messages = _walk_thread(root_id, tweet_by_id, children)
        if not messages:
            continue
        brand_msgs = [m for m in messages if m["role"] == "brand"]
        if not brand_msgs:
            continue  # Skip threads where brand never replied

        yield {
            "thread_id": root_id,
            "brand": brand_id,
            "messages": messages,
            "resolved": len(brand_msgs) > 0,
            "turns": len(messages),
        }
        count += 1


def save_threads(threads: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for t in threads:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(threads):,} threads → {out_path}")


def load_threads(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    import sys

    brand_id = sys.argv[1] if len(sys.argv) > 1 else "AmazonHelp"
    max_t = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    print(f"Loading raw data …")
    df = load_raw()

    print(f"Reconstructing threads for brand '{brand_id}' (max {max_t}) …")
    threads = list(reconstruct_threads(df, brand_id, max_threads=max_t))

    out = PROCESSED_DIR / f"{brand_id}_threads.jsonl"
    save_threads(threads, out)

    # Quick stats
    turns = [t["turns"] for t in threads]
    print(f"\nThread stats:")
    print(f"  Total threads : {len(threads):,}")
    print(f"  Avg turns     : {sum(turns)/len(turns):.1f}")
    print(f"  Max turns     : {max(turns)}")
    print(f"  1-turn threads: {sum(1 for t in turns if t == 1):,}")
