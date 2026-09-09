"""
04_build_agent.py — Build RAG store and demo the full agent pipeline.
Run: python scripts/04_build_agent.py --brand AmazonHelp
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from src.agent.classifier import IntentClassifier
from src.agent.rag_store import RAGStore
from src.agent.reply_drafter import ReplyDrafter
from src.agent.escalation import EscalationDecider

PROCESSED = Path(__file__).parents[1] / "data" / "processed"

def run_pipeline(message, clf, store, drafter, decider):
    cl = clf.classify(message)
    examples = store.retrieve(message, intent=cl["intent"], top_k=3)
    reply = drafter.draft(message, cl["intent"], examples)
    esc = decider.decide(message, cl["intent"], cl["confidence"])
    return {"message":message,"intent":cl["intent"],"intent_confidence":cl["confidence"],
            "reply":reply["reply"],"grounded_on":reply["grounded_on"],
            "escalation":esc["decision"],"escalation_reason":esc["reason"],"escalation_method":esc["method"]}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--brand", default="AmazonHelp")
    p.add_argument("--rebuild-rag", action="store_true")
    args = p.parse_args()
    store_path = PROCESSED / f"{args.brand}_rag_store.pkl"
    store = RAGStore(store_path=store_path)
    if not store_path.exists() or args.rebuild_rag:
        store.build(PROCESSED / f"{args.brand}_threads.jsonl")
    else:
        store.load()
    clf, drafter, decider = IntentClassifier(), ReplyDrafter(), EscalationDecider()
    print("\nAgent ready. Running demo...\n")
    demo = [
        "My order was supposed to arrive yesterday and still nothing. Where is it?",
        "I want to return the broken headphones I received.",
        "I have contacted Amazon 4 times about my refund and NOBODY has helped me!",
        "My account is locked. I cannot sign in at all.",
        "Alexa won't connect to Spotify since the last update.",
        "I was charged twice for the same order. Fix this NOW.",
        "Does Amazon deliver on Sundays?",
        "I pre-ordered a game and it hasn't shipped even though it came out 3 days ago.",
    ]
    results = []
    for msg in demo:
        r = run_pipeline(msg, clf, store, drafter, decider)
        results.append(r)
        print(f"{'='*65}")
        print(f"CUSTOMER : {r['message'][:100]}")
        print(f"  Intent    : {r['intent']} (conf={r['intent_confidence']:.2f})")
        print(f"  Escalation: {r['escalation'].upper()} [{r['escalation_method']}]")
        print(f"  Reason    : {r['escalation_reason']}")
        print(f"  Reply     : {r['reply']}")
    out = PROCESSED / f"{args.brand}_demo_results.json"
    with open(out,"w") as f: json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out}")
    print("Next: python scripts/05_evaluate.py --brand AmazonHelp")

if __name__ == "__main__": main()
