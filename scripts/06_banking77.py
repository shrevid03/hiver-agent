"""
06_banking77.py — Intent-generalization benchmark on Banking77 (77 intents).
Reuses the same LLM-classification approach as the Amazon agent, swapping in
Banking77's 77-intent label set, to show the architecture is domain-agnostic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, time, random
from collections import Counter, defaultdict
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from src.agent.llm import make_client, DEFAULT_MODEL, RateLimitError, APIError

SYSTEM_TMPL = """You are an intent classifier for a digital bank's customer support.
Classify the customer's message into exactly ONE of these {n} intents:
{intent_list}
Rules:
- Output ONLY JSON: {{"intent": "<exact intent name from the list>", "confidence": 0.0-1.0}}
- The intent MUST be copied EXACTLY from the list above.
- Do not explain. Just output JSON."""

class Banking77Classifier:
    def __init__(self, intent_names, model=None, max_retries=6):
        self.client = make_client()
        self.model = model or DEFAULT_MODEL
        self.max_retries = max_retries
        self.intent_names = intent_names
        self.intent_set = set(intent_names)
        self.system_prompt = SYSTEM_TMPL.format(
            n=len(intent_names),
            intent_list="\n".join(f"- {name}" for name in intent_names))

    def _wait(self, e, attempt):
        try:
            ra = e.response.headers.get("retry-after")
            if ra is not None:
                return float(ra) + random.uniform(0, 0.5)
        except Exception:
            pass
        return min(60, (2 ** attempt) + random.uniform(0, 1))

    def classify(self, message):
        last = None
        for attempt in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role":"system","content":self.system_prompt},
                              {"role":"user","content":f"Customer message: {message}"}],
                    temperature=0.0, max_tokens=512,
                    response_format={"type":"json_object"})
                intent = json.loads(r.choices[0].message.content).get("intent","")
                return intent if intent in self.intent_set else "UNKNOWN"
            except (RateLimitError, APIError) as e:
                last = e; time.sleep(self._wait(e, attempt))
            except Exception as e:
                last = e; break
        return "UNKNOWN"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-intent", type=int, default=2, help="stratified test examples per intent")
    ap.add_argument("--call-delay", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0, help="hard cap on total examples (0=none)")
    args = ap.parse_args()

    print("Loading Banking77 test split from Hugging Face …")
    ds = load_dataset("PolyAI/banking77", split="test", trust_remote_code=True)
    feat = ds.features["label"]
    names = feat.names if hasattr(feat, "names") else None
    rows = [(r["text"], names[r["label"]] if names else str(r["label"])) for r in ds]
    label_names = sorted({g for _, g in rows})
    print(f"  {len(rows)} test examples, {len(label_names)} intents.")

    by_label = defaultdict(list)
    for text, gold in rows:
        by_label[gold].append(text)
    rng = random.Random(42)
    sample = []
    for gold, texts in by_label.items():
        rng.shuffle(texts)
        for t in texts[:args.per_intent]:
            sample.append((t, gold))
    rng.shuffle(sample)
    if args.limit:
        sample = sample[:args.limit]
    print(f"  Evaluating on {len(sample)} stratified examples ({args.per_intent}/intent).\n")

    clf = Banking77Classifier(label_names)
    y_true, y_pred = [], []
    confusions = Counter()
    for i, (text, gold) in enumerate(sample, 1):
        pred = clf.classify(text)
        y_true.append(gold); y_pred.append(pred)
        if pred != gold:
            confusions[(gold, pred)] += 1
        if i % 20 == 0:
            acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
            print(f"  [{i:3d}/{len(sample)}]  acc={acc:.2%}")
        time.sleep(args.call_delay)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=label_names, average="macro", zero_division=0)
    unknown = sum(1 for p in y_pred if p == "UNKNOWN")

    print("\n" + "=" * 60)
    print("  BANKING77 GENERALIZATION RESULT")
    print("=" * 60)
    print(f"  Examples          : {len(sample)}  ({args.per_intent}/intent)")
    print(f"  Intents           : {len(label_names)}")
    print(f"  Accuracy          : {acc:.2%}")
    print(f"  Macro F1          : {macro_f1:.3f}")
    print(f"  Unmapped (UNKNOWN): {unknown}")
    print("\n  Top confusions (gold → predicted):")
    for (g, p), c in confusions.most_common(12):
        print(f"    {c}x  {g} → {p}")

    out = {"n": len(sample), "per_intent": args.per_intent,
           "n_intents": len(label_names), "accuracy": acc,
           "macro_f1": macro_f1, "unknown": unknown,
           "top_confusions": [{"gold": g, "pred": p, "count": c}
                              for (g, p), c in confusions.most_common(20)]}
    with open("data/processed/banking77_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n✅ Saved → data/processed/banking77_result.json")

if __name__ == "__main__":
    main()
