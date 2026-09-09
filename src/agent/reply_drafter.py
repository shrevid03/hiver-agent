"""
reply_drafter.py — Draft a reply grounded in historical brand responses (RAG).
"""
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

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
    def __init__(self, model="qwen/qwen3.8-27b"):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def draft(self, message, intent, retrieved_examples):
        examples_text = ""
        for i, ex in enumerate(retrieved_examples[:3], 1):
            examples_text += f"\nExample {i}:\n  Customer: {ex['customer_text'][:120]}\n  Amazon replied: {ex['brand_reply'][:200]}\n"
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
            return {"reply":"We're sorry you're experiencing this. Please DM us your order details so we can help. ^AM",
                    "grounded_on":0,"error":str(e)}
