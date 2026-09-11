"""
measure_escalation.py — Measure ONLY the escalation decider against the golden set,
using the golden intent as input to isolate escalation quality from classifier errors.
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent.escalation import EscalationDecider

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

    dec = EscalationDecider()
    tp = fp = fn = tn = 0
    llm_calls = 0
    for i, r in enumerate(rows, 1):
        gold = bool(r["should_escalate"])
        out = dec.decide(r["message"], r["golden_intent"])
        pred = out["decision"] == "escalate"
        if out["method"] == "llm":
            llm_calls += 1
        if pred and gold: tp += 1
        elif pred and not gold: fp += 1
        elif not pred and gold: fn += 1
        else: tn += 1
        if i % 20 == 0:
            acc = (tp + tn) / (tp + fp + fn + tn)
            print(f"  [{i:3d}/{len(rows)}]  acc={acc:.2%}  (LLM calls: {llm_calls})")
        time.sleep(args.call_delay)

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn)
    print("\n" + "=" * 55)
    print("  ESCALATION (golden intent as input)")
    print("=" * 55)
    print(f"  Accuracy : {acc:.2%}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print(f"  F1       : {f1:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  LLM calls made: {llm_calls}")

if __name__ == "__main__":
    main()
