"""
rag_store.py — Build and query a RAG store of historical Amazon brand replies.
"""
import json, pickle
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

PROCESSED = Path(__file__).parents[2] / "data" / "processed"
STORE_PATH = PROCESSED / "AmazonHelp_rag_store.pkl"

class RAGStore:
    def __init__(self, store_path=STORE_PATH):
        self.store_path = store_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.records = []
        self.embeddings = None

    def build(self, threads_path, intent_labels=None):
        print("Building RAG store ...")
        records = []
        with open(threads_path) as f:
            for line in f:
                thread = json.loads(line)
                brand_msgs = [m for m in thread["messages"] if m["role"]=="brand"]
                customer_msgs = [m for m in thread["messages"] if m["role"]=="customer"]
                if not brand_msgs or not customer_msgs: continue
                records.append({
                    "thread_id": thread["thread_id"],
                    "customer_text": customer_msgs[0]["text"],
                    "brand_reply": " ".join(m["text"] for m in brand_msgs[:3]),
                    "turns": thread["turns"],
                    "intent": (intent_labels or {}).get(thread["thread_id"], "unknown"),
                })
        print(f"  {len(records):,} records loaded")
        print("  Embedding customer messages ...")
        embs = self.model.encode([r["customer_text"] for r in records], batch_size=256, show_progress_bar=True)
        embs = normalize(embs)
        self.records, self.embeddings = records, embs
        with open(self.store_path, "wb") as f:
            pickle.dump({"records":records,"embeddings":embs}, f)
        print(f"  RAG store saved -> {self.store_path}")

    def load(self):
        with open(self.store_path,"rb") as f: data = pickle.load(f)
        self.records, self.embeddings = data["records"], data["embeddings"]
        print(f"RAG store loaded: {len(self.records):,} records")

    def retrieve(self, query, intent=None, top_k=3):
        if self.embeddings is None: raise RuntimeError("Call build() or load() first.")
        q = normalize(self.model.encode([query]))[0]
        sims = self.embeddings @ q
        if intent and intent != "other":
            boost = np.array([1.4 if r["intent"]==intent else 1.0 for r in self.records])
            sims = sims * boost
        top_idx = np.argsort(-sims)[:top_k]
        return [{**self.records[i], "similarity":float(sims[i])} for i in top_idx]
