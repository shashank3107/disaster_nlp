"""
Gemini-API generator — the cloud counterpart to src.rag.generator.LocalLLM.

Reuses the *exact same* grounded prompt and system instruction as the local LLM
path (so retrieval/answer behaviour is identical), but sends the request to the
Gemini API instead of running a model on the GPU. This removes any GPU/SLURM
dependency: the whole app becomes CPU-only + one network call per question.

Key (in priority order): explicit `api_key` arg -> GEMINI_API_KEY -> GOOGLE_API_KEY.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from src.rag.generator import build_prompt, _SYSTEM
from src.rag.retriever import Retrieved

logger = logging.getLogger(__name__)


class GeminiGenerator:
    """Grounded answer generation via the Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        max_output_tokens: int = 600,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self.model_name = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._model = None

    def available(self) -> bool:
        return bool(self.api_key)

    def _ensure_model(self):
        if self._model is not None:
            return
        if not self.api_key:
            raise RuntimeError(
                "No Gemini API key. Set GEMINI_API_KEY or pass it in the sidebar."
            )
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        # The grounding rules live in the system instruction (same text as local LLM).
        self._model = genai.GenerativeModel(
            self.model_name, system_instruction=_SYSTEM
        )
        logger.info("Gemini model ready: %s", self.model_name)

    def generate(self, question: str, contexts: List[Retrieved]) -> str:
        """Return a grounded, citation-bearing answer for `question`."""
        self._ensure_model()
        prompt = build_prompt(question, contexts)
        try:
            resp = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                },
            )
            return (resp.text or "").strip()
        except Exception as e:  # surface a clean message to the UI
            logger.exception("Gemini generation failed")
            return f"[Gemini error: {e}]"
