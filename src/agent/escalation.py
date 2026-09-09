"""
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
    def __init__(self, model="qwen/qwen3.8-27b"):
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
            reason = f"Rule-based: {', '.join(flags[:2])}" if rule_decision=="escalate" else "Routine inquiry — auto-handle"
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
