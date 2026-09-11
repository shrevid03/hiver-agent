"""
llm.py — Provider-agnostic LLM client (Groq / Gemini).
Both expose an OpenAI-compatible chat.completions interface, so callers use
.chat.completions.create(...) unchanged. Switch via env:
    LLM_PROVIDER=gemini|groq   LLM_MODEL=<model id>
"""
import os
from dotenv import load_dotenv
load_dotenv()

PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()

if PROVIDER == "gemini":
    from openai import OpenAI, RateLimitError, APIError
    DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")

    def make_client():
        return OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

elif PROVIDER == "groq":
    from groq import Groq, RateLimitError, APIError
    DEFAULT_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen3.8-27b")

    def make_client():
        return Groq(api_key=os.environ["GROQ_API_KEY"])

else:
    raise ValueError(f"Unknown LLM_PROVIDER '{PROVIDER}' (use 'gemini' or 'groq')")
