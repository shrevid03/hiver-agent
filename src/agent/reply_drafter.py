"""
reply_drafter.py — Draft a reply grounded in historical brand responses (RAG).
Provider-agnostic LLM client with retry-and-backoff.
"""
import time, random
from src.agent.llm import make_client, DEFAULT_MODEL, RateLimitError, APIError

SYSTEM_PROMPT = """You are a customer support agent for Amazon.
Write a helpful, empathetic reply to a customer's tweet.
- Keep the reply under 280 characters (Twitter limit).
- Be warm but concise. Don't be robotic.
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
    def __init__(self, model=None, max_retries=6):
        self.client = make_client()
        self.model = model or DEFAULT_MODEL
        self.max_retries = max_retries

    def _call_api(self, system, user):
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role":"system","content":system},
                              {"role":"user","content":user}],
                    temperature=0.4, max_tokens=150)
                return r.choices[0].message.content.strip()
            except (RateLimitError, APIError) as e:
                last_err = e
                try:
                    ra = e.response.headers.get("retry-after")
                    wait = float(ra) if ra is not None else min(60, (2**attempt)+random.uniform(0,1))
                except Exception:
                    wait = min(60, (2**attempt)+random.uniform(0,1))
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    def draft(self, message, intent, retrieved_examples):
        examples_text = ""
        for i, ex in enumerate(retrieved_examples[:3], 1):
            examples_text += f"\nExample {i}:\n  Customer: {ex['customer_text'][:120]}\n  Amazon replied: {ex['brand_reply'][:200]}\n"
        user_content = USER_TEMPLATE.format(
            message=message, intent=intent.replace("_"," "),
            examples=examples_text or "No similar examples found.")
        try:
            reply = self._call_api(SYSTEM_PROMPT, user_content)
            return {"reply": reply, "grounded_on": len(retrieved_examples)}
        except Exception as e:
            return {"reply":"We're sorry you're experiencing this. Please DM us your order details so we can help. ^AM",
                    "grounded_on":0,"error":str(e)}
