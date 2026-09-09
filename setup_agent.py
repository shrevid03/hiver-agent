import os, pathlib

base = pathlib.Path.home() / "Downloads" / "hiver-agent"
(base / "src" / "agent").mkdir(parents=True, exist_ok=True)

files = {}

files["src/agent/__init__.py"] = ""

files["src/agent/classifier.py"] = '''"""
classifier.py — Classify a customer message into one of the defined intents.
Uses few-shot prompting with Groq (LLaMA 3.1 70B).
"""
import json, os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

INTENTS_PATH = Path(__file__).parents[2] / "data" / "processed" / "AmazonHelp_intents.json"

SYSTEM_PROMPT = """You are an intent classifier for Amazon customer support messages.
Your job: read a customer\'s tweet and classify it into exactly ONE of the intents below.
{intent_definitions}
Rules:
- Output ONLY a JSON object with two fields: "intent" (the intent name) and "confidence" (0.0-1.0).
- If the message is not in English or doesn\'t fit any intent, use "other".
- Do not explain your reasoning. Just output JSON.
Example output:
{{"intent": "delivery_not_received", "confidence": 0.92}}"""

def load_intents(path=INTENTS_PATH):
    with open(path) as f: return json.load(f)

def build_intent_definitions(intents):
    return "\\n".join(f"- {i[\'intent\']}: {i[\'definition\']}" for i in intents)

class IntentClassifier:
    def __init__(self, model="llama-3.1-70b-versatile"):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model
        self.intents = load_intents()
        self.intent_names = [i["intent"] for i in self.intents]
        self.system_prompt = SYSTEM_PROMPT.format(
            intent_definitions=build_intent_definitions(self.intents))

    def classify(self, message):
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":self.system_prompt},
                          {"role":"user","content":f"Customer message: {message}"}],
                temperature=0.0, max_tokens=64,
                response_format={"type":"json_object"})
            result = json.loads(r.choices[0].message.content)
            intent = result.get("intent","other")
            if intent not in self.intent_names: intent = "other"
            return {"intent":intent,"confidence":float(result.get("confidence",0.5)),"raw_message":message}
        except Exception as e:
            return {"intent":"other","confidence":0.0,"raw_message":message,"error":str(e)}

    def classify_batch(self, messages):
        return [self.classify(m) for m in messages]
'''

files["src/agent/rag_store.py"] = '''"""
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
'''

files["src/agent/reply_drafter.py"] = '''"""
reply_drafter.py — Draft a reply grounded in historical brand responses (RAG).
"""
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = """You are a customer support agent for Amazon.
Write a helpful, empathetic reply to a customer\'s tweet.
- Keep the reply under 280 characters (Twitter limit).
- Be warm but concise. Don\'t be robotic.
- Ground your reply in how Amazon has resolved similar issues before.
- Do NOT make up order details, refund amounts, or specific timelines.
- If you need account/order details, ask politely via DM.
- End with a name initial like ^SJ.
- Never use ALL CAPS.
Write ONLY the reply. No preamble."""

USER_TEMPLATE = """Customer message: {message}
Detected intent: {intent}
Similar resolved cases:
{examples}
Write the reply now:"""

class ReplyDrafter:
    def __init__(self, model="llama-3.1-70b-versatile"):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def draft(self, message, intent, retrieved_examples):
        examples_text = ""
        for i, ex in enumerate(retrieved_examples[:3], 1):
            examples_text += f"\\nExample {i}:\\n  Customer: {ex[\'customer_text\'][:120]}\\n  Amazon replied: {ex[\'brand_reply\'][:200]}\\n"
        user_content = USER_TEMPLATE.format(
            message=message, intent=intent.replace("_"," "),
            examples=examples_text or "No similar examples found.")
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":SYSTEM_PROMPT},
                          {"role":"user","content":user_content}],
                temperature=0.4, max_tokens=150)
            return {"reply": r.choices[0].message.content.strip(), "grounded_on": len(retrieved_examples)}
        except Exception as e:
            return {"reply":"We\'re sorry you\'re experiencing this. Please DM us your order details so we can help. ^AM",
                    "grounded_on":0,"error":str(e)}
'''

files["src/agent/escalation.py"] = '''"""
escalation.py — Decide whether to auto-handle or escalate to a human.
Rule-based first, LLM for edge cases.
"""
import json, os, re
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

ESCALATION_KEYWORDS = [
    "lawyer","attorney","sue","lawsuit","legal action","court","bbb",
    "better business bureau","ftc","fraud","scam","stolen","identity theft",
    "injury","injured","dangerous","hazard","news channel","media",
    "this is unacceptable","never again","reported you",
]
SAFE_INTENTS = {"order_status","tech_issue"}
LLM_PROMPT = """Decide: should this Amazon support message be handled by a bot (auto) or escalated to a human?
Escalate if: legal threats, repeated contact attempts, safety risks, high financial distress, complex account issues.
Auto if: routine status check, solvable with a help link, first contact simple question.
Output ONLY JSON: {{"decision":"escalate" or "auto","reason":"one sentence","confidence":0.0-1.0}}
Customer message: {message}
Intent: {intent}
Rule flags: {flags}"""

class EscalationDecider:
    def __init__(self, model="llama-3.1-70b-versatile"):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def _check_rules(self, message, intent):
        msg_lower = message.lower()
        flags = []
        for kw in ESCALATION_KEYWORDS:
            if kw in msg_lower: flags.append(f"keyword:{kw}")
        if re.search(r"(called|contacted|emailed).{0,30}(times|again|already|twice)", msg_lower):
            flags.append("repeated_contact")
        words = message.split()
        if sum(1 for w in words if w.isupper() and len(w)>2)/max(len(words),1) > 0.4:
            flags.append("high_caps")
        if flags: return "escalate", flags
        if intent in SAFE_INTENTS: return "auto", ["safe_intent"]
        return None, flags

    def decide(self, message, intent, confidence=1.0):
        rule_decision, flags = self._check_rules(message, intent)
        if rule_decision:
            reason = f"Rule-based: {\', \'.join(flags[:2])}" if rule_decision=="escalate" else "Routine inquiry — auto-handle"
            return {"decision":rule_decision,"reason":reason,"method":"rules","flags":flags,"confidence":0.95}
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"user","content":LLM_PROMPT.format(message=message,intent=intent,flags=flags or "none")}],
                temperature=0.0, max_tokens=128, response_format={"type":"json_object"})
            result = json.loads(r.choices[0].message.content)
            decision = result.get("decision","escalate")
            if decision not in ("escalate","auto"): decision = "escalate"
            return {"decision":decision,"reason":result.get("reason","LLM decision"),"method":"llm","flags":flags,"confidence":float(result.get("confidence",0.7))}
        except Exception as e:
            return {"decision":"escalate","reason":"LLM unavailable — defaulting to human review","method":"fallback","flags":flags,"confidence":0.5}
'''

files["scripts/04_build_agent.py"] = '''"""
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
    print("\\nAgent ready. Running demo...\\n")
    demo = [
        "My order was supposed to arrive yesterday and still nothing. Where is it?",
        "I want to return the broken headphones I received.",
        "I have contacted Amazon 4 times about my refund and NOBODY has helped me!",
        "My account is locked. I cannot sign in at all.",
        "Alexa won\'t connect to Spotify since the last update.",
        "I was charged twice for the same order. Fix this NOW.",
        "Does Amazon deliver on Sundays?",
        "I pre-ordered a game and it hasn\'t shipped even though it came out 3 days ago.",
    ]
    results = []
    for msg in demo:
        r = run_pipeline(msg, clf, store, drafter, decider)
        results.append(r)
        print(f"{'='*65}")
        print(f"CUSTOMER : {r[\'message\'][:100]}")
        print(f"  Intent    : {r[\'intent\']} (conf={r[\'intent_confidence\']:.2f})")
        print(f"  Escalation: {r[\'escalation\'].upper()} [{r[\'escalation_method\']}]")
        print(f"  Reason    : {r[\'escalation_reason\']}")
        print(f"  Reply     : {r[\'reply\']}")
    out = PROCESSED / f"{args.brand}_demo_results.json"
    with open(out,"w") as f: json.dump(results, f, indent=2)
    print(f"\\nResults saved -> {out}")
    print("Next: python scripts/05_evaluate.py --brand AmazonHelp")

if __name__ == "__main__": main()
'''

for rel_path, content in files.items():
    full_path = base / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)
    print(f"Created: {rel_path}")

print("\nAll agent files created!")
