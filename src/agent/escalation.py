"""
escalation.py — Decide whether to auto-handle or escalate to a human.
High-precision hard rules first; the LLM decides the rest.
Uses the shared provider-agnostic client with retry-and-backoff.
"""
import json, time, random, re
from src.agent.llm import make_client, DEFAULT_MODEL, RateLimitError, APIError

HARD_ESCALATE_KEYWORDS = [
    "lawyer","attorney","sue","lawsuit","legal action","court","ftc",
    "bbb","better business bureau","identity theft","injury","injured",
    "dangerous","hazard",
]
SOFT_SIGNAL_KEYWORDS = [
    "fraud","scam","stolen","this is unacceptable","never again",
    "reported you","news channel","media","ridiculous","worst",
]
SAFE_INTENTS = {"order_status","tech_issue"}

LLM_PROMPT = """You decide whether an Amazon customer-support tweet should be handled automatically by a bot (auto) or routed to a human (escalate).

Escalate ONLY when a human is genuinely needed:
- credible legal threats, safety/injury risk, or identity theft
- the customer has clearly contacted support repeatedly with no resolution
- account lockout or security issues a bot cannot safely resolve
- severe, specific financial harm

Auto-handle everything else, including:
- routine status/refund/return/delivery questions, even if the customer is frustrated or uses strong language
- first-contact complaints a help link, DM, or standard resolution can address
- general venting ("worst service ever", "this is ridiculous") with no concrete escalation trigger

Frustration, capital letters, or emphatic wording are NOT by themselves reasons to escalate.

Output ONLY JSON: {{"decision":"escalate" or "auto","reason":"one short sentence","confidence":0.0-1.0}}
Customer message: {message}
Intent: {intent}
Soft signals detected: {flags}"""

class EscalationDecider:
    def __init__(self, model=None, max_retries=6):
        self.client = make_client()
        self.model = model or DEFAULT_MODEL
        self.max_retries = max_retries

    def _check_rules(self, message, intent):
        msg_lower = message.lower()
        for kw in HARD_ESCALATE_KEYWORDS:
            if kw in msg_lower:
                return "escalate", [f"hard:{kw}"]
        if re.search(r"(called|contacted|emailed|messaged|dm'?d).{0,30}(times|again|already|twice|multiple)", msg_lower):
            return "escalate", ["repeated_contact"]
        if intent in SAFE_INTENTS:
            return "auto", ["safe_intent"]
        soft = [f"soft:{kw}" for kw in SOFT_SIGNAL_KEYWORDS if kw in msg_lower]
        return None, soft

    def _llm_decide(self, message, intent, flags):
        last = None
        for attempt in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role":"user","content":LLM_PROMPT.format(
                        message=message, intent=intent, flags=", ".join(flags) or "none")}],
                    temperature=0.0, max_tokens=512,
                    response_format={"type":"json_object"})
                return json.loads(r.choices[0].message.content)
            except (RateLimitError, APIError) as e:
                last = e
                try:
                    ra = e.response.headers.get("retry-after")
                    wait = float(ra) if ra is not None else min(60, (2**attempt)+random.uniform(0,1))
                except Exception:
                    wait = min(60, (2**attempt)+random.uniform(0,1))
                time.sleep(wait)
        raise RuntimeError(f"LLM escalation call failed after {self.max_retries} retries: {last}")

    def decide(self, message, intent, confidence=1.0):
        rule_decision, flags = self._check_rules(message, intent)
        if rule_decision:
            reason = (f"Rule-based: {', '.join(flags[:2])}" if rule_decision == "escalate"
                      else "Routine inquiry — auto-handle")
            return {"decision":rule_decision,"reason":reason,"method":"rules","flags":flags,"confidence":0.95}
        try:
            result = self._llm_decide(message, intent, flags)
            decision = result.get("decision","auto")
            if decision not in ("escalate","auto"): decision = "auto"
            return {"decision":decision,"reason":result.get("reason","LLM decision"),
                    "method":"llm","flags":flags,"confidence":float(result.get("confidence",0.7))}
        except Exception:
            fallback = "escalate" if flags else "auto"
            return {"decision":fallback,"reason":"LLM unavailable — heuristic fallback",
                    "method":"fallback","flags":flags,"confidence":0.4}
