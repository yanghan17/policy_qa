"""
llm.py
------
Pluggable LLM backends. The QA engine calls llm.generate(prompt) → str.

Backends
--------
1. MockLLM               — default; zero dependencies, no model needed.
                           Use for development, CI, and smoke tests.

2. OllamaLLM             — local LLM via Ollama (recommended for real answers).
                           Install: https://ollama.com
                           Pull a model: ollama pull mistral
                           No API key. Fully offline. Real LLM quality.

3. HuggingFaceLocalLLM   — any HuggingFace text-generation model, runs locally.
                           Install: pip install transformers torch

4. OpenAICompatibleLLM   — any OpenAI-API-compatible server:
                           LM Studio, llama.cpp server, vLLM, Groq, etc.

All backends share the same build_prompt() function which formats
retrieved chunks with metadata headers for citation.
"""

import json
import textwrap
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import List

from ingestor import Chunk

OLLAMA_ERROR_PREFIX = "[OllamaLLM ERROR]"
# Cap context per chunk — full ~600-token chunks × 8 blows past CPU timeouts.
MAX_CHUNK_CHARS     = 2000


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a precise insurance policy assistant.

    STRICT RULES — follow these exactly:
    1. Answer ONLY using information from the CONTEXT provided below.
       Do NOT use general insurance knowledge or facts outside the context.
    2. Every factual claim MUST end with a citation in this exact format:
         [Doc: <doc_name>, §<section/clause>, p.<page>]
    3. Use "I cannot find a definitive answer in the provided policy wording."
       ONLY when the context has no clause that addresses the question topic.
       If relevant clauses exist (even if they conflict), answer from them —
       do NOT use this phrase.
    4. If two clauses in the SAME document conflict, explain BOTH, then state
       clearly which rule applies to the specific facts in the QUESTION.
    5. Use bullet points for lists. Keep answers concise and structured.
    6. For questions asking whether something is COVERED: read EXCLUSIONS
       sections first. If an exclusion applies, answer NO clearly up front,
       then note any narrow exceptions from the same clause.
    7. When multiple documents appear in context, prefer clauses from the
       home insurance policy wording (sections numbered Section 1, Section 2,
       Section 5, etc.) for home-contents and main-policy questions unless
       the question explicitly refers to another product.
    8. When the context lists multiple compensation amounts, age bands, or
       limits, include EVERY band in your answer (do not stop after the first).
    9. If the context contains two clauses that appear to conflict (e.g. cover
       in one place, exclusion in another), explain BOTH and state which
       applies to the specific situation in the question. Start with the
       situation-specific conclusion (e.g. "NO for garden contents" / "YES
       with limits") — never open with "I cannot find" when rule 9 applies.
    10. If asked about a waiting period and none is stated in the context, say:
        "The policy does not specify a waiting period for [topic]." Then cite
        what the policy does say (e.g. notify insurer promptly). Do not use
        rule 3 in that case.
    11. Cite the section named in the question when possible (e.g. Section 1
        Home Contents for home-contents questions; Section 4 for portable
        valuables; Section 5 for personal accident).
""")


def build_prompt(
    question: str,
    chunks: List[Chunk],
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> str:
    """
    Construct the full prompt from the question and retrieved chunks.

    Each chunk gets a metadata header so the LLM can form exact citations
    like [Doc: qbe_home_policy.pdf, §Section 1 > Exclusions > 1(c), p.9].
    """
    context_parts = []
    for i, c in enumerate(chunks, start=1):
        header = (
            f"[CHUNK {i}]  "
            f"Doc: {c.doc_name}  |  "
            f"Section: {c.heading_path}  |  "
            f"Clause: {c.clause_ref or 'N/A'}  |  "
            f"Page: {c.page}"
        )
        body = c.text
        if len(body) > max_chunk_chars:
            body = body[:max_chunk_chars] + "\n[... clause text truncated for length ...]"
        context_parts.append(f"{header}\n{body}")

    context_block = ("\n\n" + "-" * 60 + "\n\n").join(context_parts)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT (retrieved policy clauses):\n\n"
        f"{context_block}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER (cite every claim):"
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to the model, return response text."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# 1. MockLLM
# ---------------------------------------------------------------------------

class MockLLM(BaseLLM):
    """
    Returns a structured placeholder that shows the expected output format.
    Replace with OllamaLLM() for real answers.
    """

    def generate(self, prompt: str) -> str:
        question = ""
        for line in reversed(prompt.splitlines()):
            if line.startswith("QUESTION:"):
                question = line.replace("QUESTION:", "").strip()
                break

        num_chunks = prompt.count("[CHUNK ")

        return (
            f"[MockLLM — replace with OllamaLLM() for real answers]\n\n"
            f"Question: \"{question}\"\n"
            f"Context chunks received: {num_chunks}\n\n"
            f"Example of expected answer format:\n"
            f"• Wear and tear is NOT covered. "
            f"[Doc: qbe_home_policy.pdf, §Section 1 > Exclusions > 1(c), p.9]\n"
            f"• However, if wear and tear causes a fire, the resulting fire "
            f"damage IS covered. "
            f"[Doc: qbe_home_policy.pdf, §Section 1 > Exclusions > 1(c), p.9]"
        )


# ---------------------------------------------------------------------------
# 2. OllamaLLM
# ---------------------------------------------------------------------------

class OllamaLLM(BaseLLM):
    """
    Local LLM via Ollama. No API key. Fully offline.

    Setup
    -----
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:
         ollama pull mistral       # 4 GB, good quality
         ollama pull llama3        # 4.7 GB, best quality
         ollama pull phi3:mini     # 2.3 GB, fast on CPU
    3. Ollama runs as a background service automatically after install.

    Parameters
    ----------
    model    : Ollama model name (default: "mistral")
    base_url : Ollama server URL (default: http://localhost:11434)
    """

    def __init__(
        self,
        model:       str = "mistral",
        base_url:    str = "http://localhost:11434",
        timeout:     int = 600,
        num_predict: int = 512,
    ):
        self.model       = model
        self.base_url    = base_url.rstrip("/")
        self.timeout     = timeout
        self.num_predict = num_predict
        self._verify()

    def _verify(self):
        try:
            urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3)
            print(f"[OllamaLLM] Connected — model: {self.model}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}.\n"
                f"  Start with: ollama serve\n"
                f"  Pull model: ollama pull {self.model}\n"
                f"  Error: {e}"
            )

    def generate(self, prompt: str) -> str:
        payload = json.dumps({
            "model":      self.model,
            "prompt":     prompt,
            "stream":     False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.1,
                "num_predict": self.num_predict,
                "num_ctx":     4096,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data    = payload,
            method  = "POST",
            headers = {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())["response"]
        except Exception as e:
            return (
                f"{OLLAMA_ERROR_PREFIX} {e}\n"
                f"(timeout={self.timeout}s — use --ollama-timeout or a smaller "
                f"model e.g. phi3:mini if this persists on CPU)"
            )

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"


# ---------------------------------------------------------------------------
# 3. HuggingFaceLocalLLM
# ---------------------------------------------------------------------------

class HuggingFaceLocalLLM(BaseLLM):
    """
    Any HuggingFace text-generation model, runs entirely locally.

    Install: pip install transformers torch accelerate

    Good free models
    ----------------
    "microsoft/phi-2"                        2.7B, strong reasoning, fast
    "mistralai/Mistral-7B-Instruct-v0.2"     7B, best open-source quality
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"    1.1B, fits in 4 GB RAM

    Parameters
    ----------
    model_name : HuggingFace model identifier
    max_tokens : max new tokens to generate
    device     : "cpu", "cuda", or "mps" (Apple Silicon)
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        max_tokens: int = 512,
        device:     str = "cpu",
    ):
        from transformers import pipeline
        print(f"[HuggingFaceLocalLLM] Loading {model_name} on {device}...")
        self._pipe       = pipeline("text-generation", model=model_name, device=device)
        self._model_name = model_name
        self.max_tokens  = max_tokens
        print("[HuggingFaceLocalLLM] Ready.")

    def generate(self, prompt: str) -> str:
        out = self._pipe(
            prompt,
            max_new_tokens   = self.max_tokens,
            do_sample        = False,
            temperature      = 0.1,
            return_full_text = False,
        )
        return out[0]["generated_text"].strip()

    @property
    def name(self) -> str:
        return f"hf/{self._model_name}"


# ---------------------------------------------------------------------------
# 4. OpenAICompatibleLLM
# ---------------------------------------------------------------------------

class OpenAICompatibleLLM(BaseLLM):
    """
    Any OpenAI-API-compatible endpoint.

    Works with LM Studio, llama.cpp server, vLLM, Groq, Together.ai, etc.
    For local servers (LM Studio, llama.cpp) no API key is needed.

    Parameters
    ----------
    base_url : server URL (LM Studio default: http://localhost:1234/v1)
    model    : model name as the server expects it
    api_key  : API key ("not-needed" for local servers)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model:    str = "local-model",
        api_key:  str = "not-needed",
    ):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.api_key  = api_key

    def generate(self, prompt: str) -> str:
        payload = json.dumps({
            "model":       self.model,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  1024,
            "temperature": 0.1,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data    = payload,
            method  = "POST",
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            return f"[OpenAICompatibleLLM ERROR] HTTP {e.code}: {e.read().decode()}"
        except Exception as e:
            return f"[OpenAICompatibleLLM ERROR] {e}"

    @property
    def name(self) -> str:
        return f"openai-compat/{self.model}"
