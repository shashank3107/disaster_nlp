"""
Generation: a local HuggingFace instruct model answers, grounded in retrieved tweets.

Loads an open instruct LLM (default Qwen2.5-7B-Instruct — ungated, ~15 GB fp16,
fits a 24 GB H100 MIG) and produces an answer constrained to the retrieved context,
with inline citations to tweet_ids. The system prompt forbids using outside
knowledge, which keeps the answer faithful to the KG and is the key requirement
flagged by the disaster-LLM literature (LLMs otherwise hallucinate plausible facts).

GPU is expected; run via the SLURM script. Falls back to CPU only for tiny tests.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .retriever import Retrieved

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a disaster-response analyst. Answer the user's question using ONLY the "
    "numbered tweets provided as context. Do not use any outside knowledge. If the "
    "context does not contain the answer, say so explicitly. Cite the tweets you use "
    "with their tweet_id in square brackets, e.g. [905952332923338752]. Be concise "
    "and factual."
)


def build_prompt(question: str, contexts: List[Retrieved]) -> str:
    """Linearise retrieved tweets into a numbered, provenance-rich context block."""
    if not contexts:
        return f"CONTEXT:\n(no relevant tweets found)\n\nQUESTION: {question}\n\nANSWER:"
    blocks = []
    for i, r in enumerate(contexts, 1):
        blocks.append(f"({i}) {r.doc.context_block()}")
    ctx = "\n\n".join(blocks)
    return f"CONTEXT:\n{ctx}\n\nQUESTION: {question}\n\nANSWER:"


class LocalLLM:
    """Thin wrapper around a HF causal-LM with chat templating."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: Optional[str] = None,
        dtype: str = "float16",
        max_new_tokens: int = 400,
        load_4bit: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._tok = None
        self._model = None
        self._device = device
        self._dtype = dtype
        self._load_4bit = load_4bit

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading LLM '%s' (4bit=%s)", self.model_name, self._load_4bit)
        self._tok = AutoTokenizer.from_pretrained(self.model_name)

        kwargs = {"device_map": "auto"}
        if self._load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = getattr(torch, self._dtype)

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self._model.eval()
        logger.info("LLM loaded on %s", next(self._model.parameters()).device)

    def generate(self, question: str, contexts: List[Retrieved]) -> str:
        import torch
        self._ensure_loaded()
        prompt = build_prompt(question, contexts)
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        inputs = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,                # deterministic → reproducible
                temperature=None, top_p=None,
                pad_token_id=self._tok.eos_token_id,
            )
        text = self._tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        return text.strip()
