"""Local LLM client (Ollama) with JSON-schema structured output and full prompt/response logging.

Local explanation: wraps `ollama.Client.chat` (https://github.com/ollama/ollama-python) with `format=<JSON schema>`
(structured outputs, https://ollama.com/blog/structured-outputs), temperature 0, thinking disabled (`think=False`,
https://ollama.com/blog/thinking) and logs every call in full via `log_llm_call`.
Global context: used by auto-interpretation (labels, detection scoring) and the steering judge; swapping the local
model is a one-line `.env` change (LLM_MODEL).
"""

from __future__ import annotations

import json

import ollama

from .logging_utils import get_logger, log_llm_call

log = get_logger(__name__)


class LocalLLM:
    """Minimal JSON-returning chat client around one Ollama model."""

    def __init__(self, host: str, model: str, num_ctx: int = 8192):
        self.client = ollama.Client(host=host)
        self.model = model
        self.num_ctx = num_ctx  # context window passed per request (Ollama's default is smaller)

    def json(self, kind: str, system: str, user: str, schema: dict, retries: int = 2) -> dict:
        """Ask for an object matching `schema`; retries on unparsable output, returns {} if all attempts fail."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(retries + 1):
            resp = self.client.chat(model=self.model, messages=messages, format=schema, think=False,
                                    options={"temperature": 0, "num_ctx": self.num_ctx, "seed": attempt})
            text = resp.message.content
            log_llm_call(kind, f"[SYSTEM]\n{system}\n[USER]\n{user}", text, {"model": self.model, "attempt": attempt})
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                log.warning("LLM[%s] returned invalid JSON (attempt %d)", kind, attempt)
        return {}

    def unload(self) -> None:
        """Free the GPU: a request with keep_alive=0 unloads the model (https://github.com/ollama/ollama/blob/main/docs/faq.md)."""
        self.client.generate(model=self.model, prompt="", keep_alive=0)
