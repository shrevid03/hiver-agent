"""
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
Your job: read a customer's tweet and classify it into exactly ONE of the intents below.
{intent_definitions}
Rules:
- Output ONLY a JSON object with two fields: "intent" (the intent name) and "confidence" (0.0-1.0).
- If the message is not in English or doesn't fit any intent, use "other".
- Do not explain your reasoning. Just output JSON.
Example output:
{{"intent": "delivery_not_received", "confidence": 0.92}}"""

def load_intents(path=INTENTS_PATH):
    with open(path) as f: return json.load(f)

def build_intent_definitions(intents):
    return "\n".join(f"- {i['intent']}: {i['definition']}" for i in intents)

class IntentClassifier:
    def __init__(self, model="qwen/qwen3.8-27b"):
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
