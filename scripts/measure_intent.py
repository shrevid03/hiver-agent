"""
measure_intent.py — Classify-only measurement against the golden set.
Runs ONLY the intent classifier (~200 calls) so we can iterate on classifier
accuracy cheaply. Prints accuracy, macro-F1, per-intent F1, top confusions.
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from src.agent.classifier import IntentClassifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="data/processed/AmazonHelp_golden_set.jsonl")
    ap.add_argument("--call-delay", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.golden) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    print(f"Loaded {len(rows)} golden examples.\n")

    clf = IntentClassifier()
    y_true, y_pred = [], []
    confusions = Counter()
    for i, r in enumerate(rows, 1):
        gold = r["golden_intent"]
        pred = clf.classify(r["message"])["intent"]
        y_true.append(gold); y_pred.append(pred)
        if pred != gold:
            confusions[(gold, pred)] += 1
        if i % 20 == 0:
            acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
            print(f"  [{i:3d}/{len(rows)}]  acc={acc:.2%}")
        time.sleep(args.call_delay)

    labels = sorted(set(y_true))
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    print("\n" + "=" * 55)
    print(f"  INTENT ACCURACY : {acc:.2%}")
    print(f"  MACRO F1        : {mf1:.3f}")
    print("=" * 55)
    print("\n  Per-intent F1:")
    p, r_, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    for lab, pf, rf, ff, sf in sorted(zip(labels, p, r_, f, s), key=lambda x: -x[3]):
        print(f"    {lab:32s} P={pf:.2f} R={rf:.2f} F1={ff:.2f} (n={sf})")
    print("\n  Top confusions (gold -> predicted):")
    for (g, p_), c in confusions.most_common(12):
        print(f"    {c}x  {g} -> {p_}")

if __name__ == "__main__":
    main()
