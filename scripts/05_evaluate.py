"""
05_evaluate.py — Evaluate the full agent pipeline against a golden labelled set.

Run:
    python scripts/05_evaluate.py --brand AmazonHelp

What this script does:
  1.  Loads 200 hand-labelled golden examples (AmazonHelp_golden_set.jsonl)
  2.  Runs the classifier, reply drafter, and escalation decider on each example
  3.  Computes intent classification metrics  (accuracy, per-intent precision/recall/F1)
  4.  Computes escalation metrics             (precision, recall, F1 vs golden labels)
  5.  Runs LLM-as-judge on a sample of replies (4 rubric dimensions, 1–3 scale)
  6.  Compares against two baselines:
        - Trivial: always reply "Please DM us with your order details so we can help."
        - TF-IDF: retrieve most-similar thread and echo its brand reply (no generation)
  7.  Saves a detailed JSON report + prints a summary table

Outputs:
  data/processed/AmazonHelp_eval_results.json   — full per-example results
  data/processed/AmazonHelp_eval_report.json    — aggregate metrics / report
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.agent.classifier import IntentClassifier
from src.agent.escalation import EscalationDecider
from src.agent.rag_store import RAGStore
from src.agent.reply_drafter import ReplyDrafter

# ── Groq client (for LLM judge) ──────────────────────────────────────────────
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

PROCESSED = Path(__file__).parents[1] / "data" / "processed"
GOLDEN_PATH = PROCESSED / "AmazonHelp_golden_set.jsonl"

# ── LLM-as-Judge prompt ───────────────────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating a customer support reply written by an AI agent for Amazon.

Customer message:
{message}

AI-generated reply:
{reply}

Reference examples of how Amazon has resolved similar issues (from historical data):
{examples}

Score the reply on FOUR dimensions. For each, output an integer 1, 2, or 3.

1. GROUNDEDNESS (1-3):
   1 = Makes up facts, timelines, or promises not supported by examples or context
   2 = Mostly grounded but includes one unsupported claim
   3 = Fully grounded — only states what it can actually know or commits to

2. TONE (1-3):
   1 = Robotic, dismissive, or tone-deaf to the customer's frustration
   2 = Acceptable but generic / could be warmer
   3 = Empathetic, warm, professional — matches the emotional weight of the message

3. COMPLETENESS (1-3):
   1 = Misses the core ask entirely or gives an off-topic response
   2 = Addresses the issue but omits a key step or next action
   3 = Fully addresses the issue with a clear next step

4. ACTIONABILITY (1-3):
   1 = No guidance — leaves the customer with nothing to do
   2 = Vague direction ("please contact us") without specifics
   3 = Concrete next step — what to do, how, or what to expect

Output ONLY a JSON object:
{{"groundedness": <1-3>, "tone": <1-3>, "completeness": <1-3>, "actionability": <1-3>, "overall_comment": "<one sentence>"}}"""


# ── TF-IDF Baseline ───────────────────────────────────────────────────────────

def build_tfidf_baseline(threads_path: Path):
    """Build a TF-IDF retriever from the thread JSONL for comparison."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts, replies = [], []
    with open(threads_path) as f:
        for line in f:
            t = json.loads(line)
            messages = t.get("messages", [])
            customer_msgs = [m["text"] for m in messages if m.get("role") == "customer"]
            brand_msgs = [m["text"] for m in messages if m.get("role") == "brand"]
            if customer_msgs and brand_msgs:
                texts.append(" ".join(customer_msgs))
                replies.append(brand_msgs[0])

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)

    def retrieve(query: str, top_k: int = 1) -> str:
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
        best_idx = sims.argsort()[::-1][:top_k]
        return replies[best_idx[0]] if len(replies) > 0 else "Please DM us so we can help."

    return retrieve


# ── LLM Judge ────────────────────────────────────────────────────────────────

class LLMJudge:
    def __init__(self, model: str = "qwen/qwen3-27b"):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def score(self, message: str, reply: str, examples: list[dict]) -> dict:
        import re as _re
        examples_text = ""
        for i, ex in enumerate(examples[:2], 1):
            examples_text += (
                f"\nExample {i}: Customer: {ex['customer_text'][:100]} "
                f"→ Amazon replied: {ex['brand_reply'][:150]}\n"
            )

        prompt = JUDGE_PROMPT.format(
            message=message,
            reply=reply,
            examples=examples_text or "No reference examples available.",
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
                # No response_format — qwen3 thinking tokens break json_object mode
            )
            raw = response.choices[0].message.content or ""
            # Strip <think>...</think> blocks (Qwen3 chain-of-thought)
            raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
            # Try direct parse; fall back to extracting first {...} block
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                m = _re.search(r"\{[^{}]+\}", raw, _re.DOTALL)
                if not m:
                    raise ValueError(f"No JSON in judge response: {raw[:300]!r}")
                result = json.loads(m.group())
            # Clamp scores to [1, 3]
            for dim in ["groundedness", "tone", "completeness", "actionability"]:
                val = result.get(dim, 2)
                result[dim] = max(1, min(3, int(float(val))))
            result["avg"] = round(
                (result["groundedness"] + result["tone"] +
                 result["completeness"] + result["actionability"]) / 4, 2
            )
            return result
        except Exception as e:
            print(f"    [judge error] {e}")
            return {
                "groundedness": 0, "tone": 0, "completeness": 0,
                "actionability": 0, "avg": 0.0, "error": str(e),
            }


# ── Intent metrics helper ─────────────────────────────────────────────────────

def compute_intent_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_intent = {}
    for i, label in enumerate(labels):
        per_intent[label] = {
            "precision": round(float(p[i]), 3),
            "recall": round(float(r[i]), 3),
            "f1": round(float(f[i]), 3),
        }

    # Macro averages
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_mac), 4),
        "macro_recall": round(float(r_mac), 4),
        "macro_f1": round(float(f_mac), 4),
        "per_intent": per_intent,
    }


def compute_escalation_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a and b)
    fp = sum(1 for a, b in zip(y_true, y_pred) if not a and b)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a and not b)
    tn = sum(1 for a, b in zip(y_true, y_pred) if not a and not b)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--brand", default="AmazonHelp")
    p.add_argument("--judge-sample", type=int, default=40,
                   help="Number of examples to send to LLM judge (default 40)")
    p.add_argument("--rebuild-rag", action="store_true")
    p.add_argument("--skip-judge", action="store_true",
                   help="Skip LLM-as-judge scoring (faster run, classification only)")
    p.add_argument("--judge-only", action="store_true",
                   help="Load saved eval_results.json and only re-run the LLM judge (saves API quota)")
    p.add_argument("--call-delay", type=float, default=1.5,
                   help="Seconds to sleep between classifier API calls (default 1.5)")
    args = p.parse_args()

    threads_path = PROCESSED / f"{args.brand}_threads.jsonl"
    store_path = PROCESSED / f"{args.brand}_rag_store.pkl"
    golden_path = PROCESSED / "AmazonHelp_golden_set.jsonl"

    # ── Load golden set ──
    print(f"Loading golden set from {golden_path} …")
    golden = []
    with open(golden_path) as f:
        for line in f:
            golden.append(json.loads(line.strip()))
    print(f"  {len(golden)} labelled examples loaded.\n")

    # ── Judge-only mode: reload saved results, skip classification ──
    if args.judge_only:
        results_path = PROCESSED / f"{args.brand}_eval_results.json"
        if not results_path.exists():
            print(f"[ERROR] No saved results at {results_path}. Run without --judge-only first.")
            sys.exit(1)
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} saved results from {results_path}\n")
        store = RAGStore(store_path=store_path)
        store.load()
        judge = LLMJudge()
        judge_candidates = list(range(0, len(results), max(1, len(results) // args.judge_sample)))
        judge_scores_agent, judge_scores_trivial, judge_scores_tfidf = [], [], []
        print(f"Running LLM judge on {len(judge_candidates)} examples …")
        for rank, idx in enumerate(judge_candidates):
            r = results[idx]
            examples_for_judge = store.retrieve(r["message"], intent=r["predicted_intent"], top_k=2)
            agent_score = judge.score(r["message"], r["agent_reply"], examples_for_judge)
            trivial_score = judge.score(r["message"], r["trivial_reply"], examples_for_judge)
            tfidf_score = judge.score(r["message"], r["tfidf_reply"], examples_for_judge)
            results[idx]["judge_scores_agent"] = agent_score
            results[idx]["judge_scores_trivial"] = trivial_score
            results[idx]["judge_scores_tfidf"] = tfidf_score
            judge_scores_agent.append(agent_score)
            judge_scores_trivial.append(trivial_score)
            judge_scores_tfidf.append(tfidf_score)
            if (rank + 1) % 10 == 0:
                print(f"  Judged {rank + 1}/{len(judge_candidates)} …")
            time.sleep(2.0)
        print("Judge pass complete.\n")
        # Recompute metrics from saved labels
        intent_true = [r["golden_intent"] for r in results]
        intent_pred = [r["predicted_intent"] for r in results]
        esc_true = [r["should_escalate"] for r in results]
        esc_pred = [r["predicted_escalation"] == "escalate" for r in results]
        intent_labels = sorted(list(set(intent_true)))
        intent_metrics = compute_intent_metrics(intent_true, intent_pred, intent_labels)
        esc_metrics = compute_escalation_metrics(esc_true, esc_pred)
        dims = ["groundedness", "tone", "completeness", "actionability", "avg"]
        def avg_judge_dim(scores, dim):
            vals = [s[dim] for s in scores if s and s.get(dim, 0) > 0]
            return round(sum(vals) / len(vals), 3) if vals else 0.0
        judge_summary = {
            "n_judged": len(judge_scores_agent),
            "agent": {d: avg_judge_dim(judge_scores_agent, d) for d in dims},
            "trivial_baseline": {d: avg_judge_dim(judge_scores_trivial, d) for d in dims},
            "tfidf_baseline": {d: avg_judge_dim(judge_scores_tfidf, d) for d in dims},
        }
        confusion = defaultdict(int)
        for t, p_i in zip(intent_true, intent_pred):
            if t != p_i:
                confusion[f"{t} → {p_i}"] += 1
        top_errors = sorted(confusion.items(), key=lambda x: -x[1])[:10]
        diff_breakdown = defaultdict(lambda: {"total": 0, "intent_correct": 0, "esc_correct": 0})
        for r in results:
            d = r["difficulty"]
            diff_breakdown[d]["total"] += 1
            diff_breakdown[d]["intent_correct"] += int(r["intent_correct"])
            diff_breakdown[d]["esc_correct"] += int(r["escalation_correct"])
        for d, v in diff_breakdown.items():
            n = v["total"]
            v["intent_accuracy"] = round(v["intent_correct"] / n, 4)
            v["escalation_accuracy"] = round(v["esc_correct"] / n, 4)
        report = {
            "brand": args.brand, "n_examples": len(results),
            "intent_classification": intent_metrics, "escalation": esc_metrics,
            "reply_quality_judge": judge_summary,
            "top_misclassifications": [{"pair": k, "count": v} for k, v in top_errors],
            "difficulty_breakdown": dict(diff_breakdown),
        }
        report_path = PROCESSED / f"{args.brand}_eval_report.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        _print_summary(report, judge_summary, top_errors, diff_breakdown, dims)
        print(f"\n✅ Full results → {results_path}")
        print(f"✅ Report       → {report_path}")
        return

    # ── Build / load RAG store ──
    store = RAGStore(store_path=store_path)
    if not store_path.exists() or args.rebuild_rag:
        if not threads_path.exists():
            print(f"[ERROR] Threads file not found: {threads_path}")
            print("  Run scripts/02_build_dataset.py first.")
            sys.exit(1)
        print("Building RAG store …")
        store.build(threads_path)
    else:
        print("Loading RAG store …")
        store.load()

    # ── Init components ──
    print("Initialising agent components …")
    clf = IntentClassifier()
    drafter = ReplyDrafter()
    decider = EscalationDecider()
    judge = LLMJudge() if not args.skip_judge else None
    print("Ready.\n")

    # ── Build TF-IDF baseline ──
    tfidf_retrieve = None
    if threads_path.exists():
        print("Building TF-IDF baseline …")
        tfidf_retrieve = build_tfidf_baseline(threads_path)
        print("  TF-IDF baseline ready.\n")

    # ── Per-example evaluation ──
    results = []
    intent_true, intent_pred = [], []
    esc_true, esc_pred = [], []
    esc_true_trivial, esc_pred_trivial = [], []

    TRIVIAL_REPLY = "Please DM us with your order details so we can help."

    judge_candidates = []  # indices to judge
    judge_interval = max(1, len(golden) // args.judge_sample)

    print(f"Running evaluation on {len(golden)} examples …")
    for i, example in enumerate(golden):
        msg = example["message"]
        true_intent = example["golden_intent"]
        should_esc = example["should_escalate"]

        # Agent pipeline
        classification = clf.classify(msg)
        examples = store.retrieve(msg, intent=classification["intent"], top_k=3)
        reply_result = drafter.draft(msg, classification["intent"], examples)
        escalation = decider.decide(msg, classification["intent"], classification["confidence"])

        predicted_intent = classification["intent"]
        predicted_esc = escalation["decision"] == "escalate"

        intent_true.append(true_intent)
        intent_pred.append(predicted_intent)
        esc_true.append(should_esc)
        esc_pred.append(predicted_esc)

        # Baselines
        trivial_reply = TRIVIAL_REPLY
        tfidf_reply = tfidf_retrieve(msg) if tfidf_retrieve else TRIVIAL_REPLY

        result = {
            "id": example["id"],
            "message": msg,
            "golden_intent": true_intent,
            "should_escalate": should_esc,
            "difficulty": example.get("difficulty", "medium"),
            "notes": example.get("notes", ""),
            # Agent outputs
            "predicted_intent": predicted_intent,
            "intent_correct": predicted_intent == true_intent,
            "intent_confidence": classification["confidence"],
            "predicted_escalation": escalation["decision"],
            "escalation_correct": predicted_esc == should_esc,
            "escalation_method": escalation["method"],
            "escalation_reason": escalation["reason"],
            "agent_reply": reply_result["reply"],
            "grounded_on": reply_result["grounded_on"],
            "trivial_reply": trivial_reply,
            "tfidf_reply": tfidf_reply,
            # Judge scores filled in later
            "judge_scores_agent": None,
            "judge_scores_trivial": None,
            "judge_scores_tfidf": None,
        }
        results.append(result)

        if i % judge_interval == 0:
            judge_candidates.append(i)

        time.sleep(args.call_delay)  # Rate-limit guard

        if (i + 1) % 20 == 0:
            correct = sum(1 for r in results if r["intent_correct"])
            print(f"  [{i+1:3d}/{len(golden)}]  intent_acc={correct/(i+1):.2%}")

    print(f"\nClassification pass complete.")

    # ── LLM Judge pass ──
    judge_scores_agent, judge_scores_trivial, judge_scores_tfidf = [], [], []

    if judge and judge_candidates:
        print(f"\nRunning LLM judge on {len(judge_candidates)} sampled examples …")
        for rank, idx in enumerate(judge_candidates):
            r = results[idx]
            examples_for_judge = store.retrieve(r["message"], intent=r["predicted_intent"], top_k=2)

            agent_score = judge.score(r["message"], r["agent_reply"], examples_for_judge)
            trivial_score = judge.score(r["message"], r["trivial_reply"], examples_for_judge)
            tfidf_score = judge.score(r["message"], r["tfidf_reply"], examples_for_judge)

            results[idx]["judge_scores_agent"] = agent_score
            results[idx]["judge_scores_trivial"] = trivial_score
            results[idx]["judge_scores_tfidf"] = tfidf_score

            judge_scores_agent.append(agent_score)
            judge_scores_trivial.append(trivial_score)
            judge_scores_tfidf.append(tfidf_score)

            if (rank + 1) % 10 == 0:
                print(f"  Judged {rank + 1}/{len(judge_candidates)} …")
            time.sleep(2.0)  # Groq free tier rate-limit guard
        print("Judge pass complete.\n")

    # ── Aggregate metrics ──
    intent_labels = sorted(list(set(intent_true)))
    intent_metrics = compute_intent_metrics(intent_true, intent_pred, intent_labels)

    esc_metrics = compute_escalation_metrics(esc_true, esc_pred)

    # Judge aggregate
    def avg_judge_dim(scores, dim):
        vals = [s[dim] for s in scores if s and s.get(dim, 0) > 0]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    dims = ["groundedness", "tone", "completeness", "actionability", "avg"]
    judge_summary = {}
    if judge_scores_agent:
        judge_summary = {
            "n_judged": len(judge_scores_agent),
            "agent": {d: avg_judge_dim(judge_scores_agent, d) for d in dims},
            "trivial_baseline": {d: avg_judge_dim(judge_scores_trivial, d) for d in dims},
            "tfidf_baseline": {d: avg_judge_dim(judge_scores_tfidf, d) for d in dims},
        }

    # Error analysis — most common misclassifications
    confusion = defaultdict(int)
    for t, p in zip(intent_true, intent_pred):
        if t != p:
            confusion[f"{t} → {p}"] += 1
    top_errors = sorted(confusion.items(), key=lambda x: -x[1])[:10]

    # Per-difficulty breakdown
    diff_breakdown = defaultdict(lambda: {"total": 0, "intent_correct": 0, "esc_correct": 0})
    for r in results:
        d = r["difficulty"]
        diff_breakdown[d]["total"] += 1
        diff_breakdown[d]["intent_correct"] += int(r["intent_correct"])
        diff_breakdown[d]["esc_correct"] += int(r["escalation_correct"])
    for d, v in diff_breakdown.items():
        n = v["total"]
        v["intent_accuracy"] = round(v["intent_correct"] / n, 4)
        v["escalation_accuracy"] = round(v["esc_correct"] / n, 4)

    report = {
        "brand": args.brand,
        "n_examples": len(golden),
        "intent_classification": intent_metrics,
        "escalation": esc_metrics,
        "reply_quality_judge": judge_summary,
        "top_misclassifications": [{"pair": k, "count": v} for k, v in top_errors],
        "difficulty_breakdown": dict(diff_breakdown),
    }

    # ── Save outputs ──
    results_path = PROCESSED / f"{args.brand}_eval_results.json"
    report_path = PROCESSED / f"{args.brand}_eval_report.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    _print_summary(report, judge_summary, top_errors, diff_breakdown, dims)
    print(f"\n✅ Full results → {results_path}")
    print(f"✅ Report       → {report_path}")
    print("\nNext step: write the report using data from these files.")


def _print_summary(report, judge_summary, top_errors, diff_breakdown, dims):
    intent_metrics = report["intent_classification"]
    esc_metrics = report["escalation"]

    print("=" * 70)
    print(f"  EVALUATION SUMMARY — {report['brand']}")
    print("=" * 70)

    print(f"\n📌 INTENT CLASSIFICATION  ({report['n_examples']} examples)")
    print(f"   Accuracy      : {intent_metrics['accuracy']:.2%}")
    print(f"   Macro F1      : {intent_metrics['macro_f1']:.3f}")
    print(f"   Macro Precision: {intent_metrics['macro_precision']:.3f}")
    print(f"   Macro Recall  : {intent_metrics['macro_recall']:.3f}")
    print(f"\n   Per-intent F1:")
    for intent, m in sorted(intent_metrics["per_intent"].items(), key=lambda x: -x[1]["f1"]):
        print(f"     {intent:<35s} P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}")

    print(f"\n📌 ESCALATION DECISION")
    print(f"   Accuracy   : {esc_metrics['accuracy']:.2%}")
    print(f"   Precision  : {esc_metrics['precision']:.3f}  (of flagged, how many should escalate)")
    print(f"   Recall     : {esc_metrics['recall']:.3f}  (of true escalations, how many caught)")
    print(f"   F1         : {esc_metrics['f1']:.3f}")
    print(f"   TP={esc_metrics['tp']}  FP={esc_metrics['fp']}  FN={esc_metrics['fn']}  TN={esc_metrics['tn']}")

    if judge_summary and judge_summary.get("n_judged", 0) > 0:
        print(f"\n📌 REPLY QUALITY (LLM Judge, n={judge_summary['n_judged']}, scale 1-3)")
        print(f"   {'Dimension':<18s}  {'Agent':>7s}  {'TF-IDF':>7s}  {'Trivial':>7s}")
        print(f"   {'-'*50}")
        for dim in dims:
            a = judge_summary["agent"].get(dim, 0)
            tf = judge_summary["tfidf_baseline"].get(dim, 0)
            tr = judge_summary["trivial_baseline"].get(dim, 0)
            print(f"   {dim:<18s}  {a:>7.2f}  {tf:>7.2f}  {tr:>7.2f}")
    else:
        print(f"\n📌 REPLY QUALITY: judge scores not available (run with --judge-only to score)")

    print(f"\n📌 DIFFICULTY BREAKDOWN")
    for d, v in sorted(diff_breakdown.items()):
        print(f"   {d:<8s}  n={v['total']:3d}  intent_acc={v['intent_accuracy']:.2%}  esc_acc={v['escalation_accuracy']:.2%}")

    print(f"\n📌 TOP MISCLASSIFICATIONS")
    for pair, count in list(top_errors)[:5]:
        count_val = count if isinstance(count, int) else count
        print(f"   {count_val:3d}x  {pair}")


if __name__ == "__main__":
    main()
