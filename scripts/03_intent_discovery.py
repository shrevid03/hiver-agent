"""
03_intent_discovery.py — Discover natural intent clusters from customer messages.

Uses sentence-transformers embeddings + k-means clustering.
NO LLM API needed — runs fully offline.

Run:
    python scripts/03_intent_discovery.py --brand AmazonHelp

Outputs:
    data/processed/AmazonHelp_clusters.json   (cluster assignments)
    data/processed/AmazonHelp_intents.json    (proposed intent taxonomy — edit this!)
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parents[1]))

PROCESSED = Path(__file__).parents[1] / "data" / "processed"

# ── Helpers ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove @mentions, URLs, HTML entities, extra whitespace."""
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"&amp;|&lt;|&gt;|&quot;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_english(text: str, threshold: float = 0.85) -> bool:
    """Rough English filter: fraction of ASCII printable chars."""
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) >= threshold


def load_customer_messages(brand: str, max_per_thread: int = 1) -> list[dict]:
    """
    Extract the opening customer message from each thread.
    Returns list of {thread_id, text, cleaned}.
    """
    path = PROCESSED / f"{brand}_threads.jsonl"
    messages = []
    with open(path) as f:
        for line in f:
            thread = json.loads(line)
            customer_msgs = [m for m in thread["messages"] if m["role"] == "customer"]
            if not customer_msgs:
                continue
            # Use the first customer message as the intent signal
            first = customer_msgs[0]
            cleaned = clean_text(first["text"])
            if len(cleaned) < 10:
                continue
            if not is_english(cleaned):
                continue
            messages.append({
                "thread_id": thread["thread_id"],
                "text": first["text"],
                "cleaned": cleaned,
                "turns": thread["turns"],
            })
    return messages


def embed_messages(texts: list[str]) -> np.ndarray:
    """Embed texts using a lightweight sentence-transformer model."""
    from sentence_transformers import SentenceTransformer
    print(f"  Loading sentence-transformer model …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  Embedding {len(texts):,} messages …")
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True)
    return normalize(embeddings)  # L2-normalise for better k-means


def find_best_k(embeddings: np.ndarray, k_range: range) -> int:
    """Find optimal k using silhouette score."""
    print("\n  Finding optimal k (silhouette scores) …")
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=2000, random_state=42)
        scores[k] = score
        print(f"    k={k}: silhouette={score:.4f}")
    best_k = max(scores, key=scores.get)
    print(f"  → Best k: {best_k} (score={scores[best_k]:.4f})")
    return best_k


def cluster_and_report(
    messages: list[dict],
    embeddings: np.ndarray,
    k: int,
    n_examples: int = 8,
) -> list[dict]:
    """Run k-means, print cluster summaries, return cluster assignments."""
    print(f"\n  Clustering into {k} groups …")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    clusters = []
    for cluster_id in range(k):
        indices = np.where(labels == cluster_id)[0]
        cluster_texts = [messages[i]["cleaned"] for i in indices]

        # Find examples closest to centroid
        centroid = km.cluster_centers_[cluster_id]
        cluster_embs = embeddings[indices]
        sims = cluster_embs @ centroid
        top_idx = np.argsort(-sims)[:n_examples]
        examples = [cluster_texts[i] for i in top_idx]

        # Top keywords (simple word freq, excluding stopwords)
        stopwords = {
            "the","a","an","is","it","i","my","to","for","of","and","in",
            "on","at","with","this","that","have","has","not","be","was",
            "are","but","or","your","you","we","me","they","do","did","get",
            "got","can","just","please","need","want","help","hi","hello",
            "hey","amazon","amazonhelp","order","would","could","should"
        }
        word_freq: Counter = Counter()
        for t in cluster_texts:
            for w in t.lower().split():
                w = re.sub(r"[^\w]", "", w)
                if w and w not in stopwords and len(w) > 2:
                    word_freq[w] += 1
        keywords = [w for w, _ in word_freq.most_common(10)]

        clusters.append({
            "cluster_id": cluster_id,
            "size": len(indices),
            "pct": round(len(indices) / len(messages) * 100, 1),
            "keywords": keywords,
            "examples": examples,
            "thread_ids": [messages[i]["thread_id"] for i in indices],
        })

    # Sort by size
    clusters.sort(key=lambda c: c["size"], reverse=True)
    return clusters, labels.tolist()


def print_clusters(clusters: list[dict]) -> None:
    print("\n" + "="*65)
    print("CLUSTER SUMMARY — use this to name your intents")
    print("="*65)
    for c in clusters:
        print(f"\nCluster {c['cluster_id']}  ({c['size']} msgs, {c['pct']}%)")
        print(f"  Keywords: {', '.join(c['keywords'][:7])}")
        print(f"  Examples:")
        for ex in c["examples"][:4]:
            print(f"    • {ex[:110]}")


def build_intent_taxonomy(clusters: list[dict]) -> list[dict]:
    """
    Propose intent names based on cluster keywords.
    You MUST review and edit data/processed/<brand>_intents.json after running.
    """
    # These are data-driven guesses — edit the output file to correct them
    intents = []
    for c in clusters:
        kws = set(c["keywords"][:5])

        # Heuristic name assignment — edit the JSON output to refine
        if kws & {"delivery", "delivered", "package", "shipped", "arrive", "shipping", "missing"}:
            name = "delivery_issue"
        elif kws & {"refund", "return", "returned", "returning", "money", "back"}:
            name = "return_refund"
        elif kws & {"charge", "charged", "billing", "payment", "card", "paid", "fee"}:
            name = "billing_dispute"
        elif kws & {"account", "login", "password", "access", "sign", "locked"}:
            name = "account_access"
        elif kws & {"cancel", "cancelled", "cancellation"}:
            name = "order_cancel"
        elif kws & {"prime", "membership", "subscription"}:
            name = "prime_subscription"
        elif kws & {"product", "item", "quality", "broken", "damaged", "wrong"}:
            name = "product_issue"
        elif kws & {"app", "website", "site", "error", "bug", "loading", "working"}:
            name = "tech_issue"
        else:
            name = f"other_{c['cluster_id']}"

        intents.append({
            "intent": name,
            "cluster_id": c["cluster_id"],
            "size": c["size"],
            "pct": c["pct"],
            "keywords": c["keywords"],
            "definition": f"EDIT THIS: customer messages about {name.replace('_', ' ')}",
            "examples": c["examples"][:3],
        })
    return intents


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--brand", default="AmazonHelp")
    p.add_argument("--k", type=int, default=None,
                   help="Number of clusters (default: auto-detect 8-14)")
    p.add_argument("--skip-k-search", action="store_true",
                   help="Skip silhouette search, use --k directly")
    args = p.parse_args()

    print(f"Loading customer messages for '{args.brand}' …")
    messages = load_customer_messages(args.brand)
    print(f"  {len(messages):,} English customer messages loaded")

    texts = [m["cleaned"] for m in messages]
    embeddings = embed_messages(texts)

    if args.k:
        k = args.k
    elif args.skip_k_search:
        k = 10
    else:
        k = find_best_k(embeddings, k_range=range(7, 14))

    clusters, labels = cluster_and_report(messages, embeddings, k)
    print_clusters(clusters)

    # Save cluster assignments
    out_clusters = PROCESSED / f"{args.brand}_clusters.json"
    with open(out_clusters, "w") as f:
        json.dump({
            "brand": args.brand,
            "k": k,
            "total_messages": len(messages),
            "clusters": clusters,
        }, f, indent=2)
    print(f"\nSaved clusters → {out_clusters}")

    # Save proposed intent taxonomy
    intents = build_intent_taxonomy(clusters)
    out_intents = PROCESSED / f"{args.brand}_intents.json"
    with open(out_intents, "w") as f:
        json.dump(intents, f, indent=2)
    print(f"Saved intent taxonomy → {out_intents}")
    print("\n⚠️  IMPORTANT: Open the intents JSON and edit the 'intent' and")
    print("   'definition' fields based on what you see in the cluster examples.")
    print("   This is your taxonomy — it should come from the data, not from guessing.")
    print(f"\nNext step: review {out_intents}, then run scripts/04_build_agent.py")


if __name__ == "__main__":
    main()