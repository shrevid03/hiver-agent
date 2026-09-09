"""
02_build_dataset.py — Build the clean JSONL dataset for the chosen brand.

Run:
    python scripts/02_build_dataset.py --brand AmazonHelp --max-threads 8000

Outputs:
    data/processed/<brand>_threads.jsonl   (full dataset)
    data/processed/<brand>_sample.jsonl    (500 random threads, for quick iteration)
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.load import load_raw
from src.data.reconstruct import reconstruct_threads, save_threads

PROCESSED = Path(__file__).parents[1] / "data" / "processed"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--brand", default="AmazonHelp", help="Brand author_id from EDA")
    p.add_argument("--max-threads", type=int, default=8000)
    p.add_argument("--sample-size", type=int, default=500,
                   help="Size of quick-iteration sample JSONL")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def print_stats(threads: list[dict], brand: str) -> None:
    turns_list = [t["turns"] for t in threads]
    avg_turns = sum(turns_list) / len(turns_list)
    max_turns = max(turns_list)
    single = sum(1 for t in turns_list if t == 1)
    long_threads = sum(1 for t in turns_list if t >= 5)

    # Customer message length
    all_customer_texts = [
        m["text"]
        for t in threads
        for m in t["messages"]
        if m["role"] == "customer"
    ]
    avg_cust_len = sum(len(t) for t in all_customer_texts) / len(all_customer_texts)

    print(f"\n{'='*50}")
    print(f"Dataset stats for: {brand}")
    print(f"{'='*50}")
    print(f"  Threads total      : {len(threads):,}")
    print(f"  Avg turns/thread   : {avg_turns:.1f}")
    print(f"  Max turns          : {max_turns}")
    print(f"  Single-turn threads: {single:,}  ({single/len(threads):.1%})")
    print(f"  Long threads (≥5)  : {long_threads:,}  ({long_threads/len(threads):.1%})")
    print(f"  Avg customer msg   : {avg_cust_len:.0f} chars")

    # Show 3 sample threads
    print(f"\n--- 3 sample threads ---")
    for t in random.sample(threads, min(3, len(threads))):
        print(f"\n  Thread {t['thread_id']} ({t['turns']} turns):")
        for m in t["messages"][:4]:  # Show first 4 messages
            role = m["role"].upper()
            print(f"    [{role}] {m['text'][:120]}")
        if t["turns"] > 4:
            print(f"    ... ({t['turns'] - 4} more messages)")


def main():
    args = parse_args()
    random.seed(args.seed)

    print(f"Loading raw data …")
    df = load_raw()
    print(f"  {len(df):,} tweets loaded")

    print(f"\nReconstructing threads for '{args.brand}' (max {args.max_threads}) …")
    threads = list(reconstruct_threads(df, args.brand, max_threads=args.max_threads))

    if not threads:
        print(f"ERROR: No threads found for brand '{args.brand}'.")
        print("       Run scripts/01_eda.py first to see valid brand IDs.")
        sys.exit(1)

    print_stats(threads, args.brand)

    # Save full dataset
    out_full = PROCESSED / f"{args.brand}_threads.jsonl"
    save_threads(threads, out_full)

    # Save sample
    sample = random.sample(threads, min(args.sample_size, len(threads)))
    out_sample = PROCESSED / f"{args.brand}_sample.jsonl"
    save_threads(sample, out_sample)

    print(f"\nDone.")
    print(f"  Full dataset  → {out_full}")
    print(f"  Sample ({len(sample)}) → {out_sample}")
    print(f"\nNext step: run scripts/03_intent_discovery.py --brand {args.brand}")


if __name__ == "__main__":
    main()
