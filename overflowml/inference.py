"""Inference client — connect to LLM backends or load models directly.

Supports: Ollama, llama.cpp server, any OpenAI-compatible API, direct HF loading.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Iterator, Optional

logger = logging.getLogger("overflowml")


@dataclass
class ServerConfig:
    """Connection config for an LLM server."""
    name: str = ""
    url: str = "http://localhost:11434"  # Ollama default
    backend: str = "ollama"              # ollama, llamacpp, openai
    model: str = ""
    api_key: str = ""

    @property
    def chat_url(self) -> str:
        if self.backend == "ollama":
            return f"{self.url}/api/chat"
        elif self.backend == "llamacpp":
            return f"{self.url}/v1/chat/completions"
        else:  # openai-compatible
            return f"{self.url}/v1/chat/completions"


@dataclass
class ChatMessage:
    role: str = "user"
    content: str = ""


@dataclass
class ChatResponse:
    content: str = ""
    model: str = ""
    tokens_used: int = 0
    error: str = ""


# ============================================================
# Server Discovery
# ============================================================

COMMON_SERVERS = [
    ServerConfig(name="Ollama", url="http://localhost:11434", backend="ollama"),
    ServerConfig(name="llama.cpp (8080)", url="http://localhost:8080", backend="llamacpp"),
    ServerConfig(name="llama.cpp (8081)", url="http://localhost:8081", backend="llamacpp"),
    ServerConfig(name="llama.cpp (8082)", url="http://localhost:8082", backend="llamacpp"),
    ServerConfig(name="llama.cpp (8083)", url="http://localhost:8083", backend="llamacpp"),
    ServerConfig(name="vLLM", url="http://localhost:8000", backend="openai"),
    ServerConfig(name="TGI", url="http://localhost:8080", backend="openai"),
]


def discover_servers(timeout: float = 2.0) -> list[ServerConfig]:
    """Scan for running LLM servers on common ports."""
    found = []
    for server in COMMON_SERVERS:
        try:
            if server.backend == "ollama":
                url = f"{server.url}/api/tags"
            else:
                url = f"{server.url}/v1/models"

            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=timeout)
            data = json.loads(resp.read())

            # Extract available models
            models = []
            if server.backend == "ollama" and "models" in data:
                models = [m.get("name", "") for m in data["models"]]
            elif "data" in data:
                models = [m.get("id", "") for m in data["data"]]

            s = ServerConfig(
                name=server.name,
                url=server.url,
                backend=server.backend,
                model=models[0] if models else "",
            )
            s._models = models
            found.append(s)
            logger.info("Found %s at %s with %d models", server.name, server.url, len(models))
        except (urllib.error.URLError, OSError, json.JSONDecodeError, Exception):
            pass
    return found


def list_models(server: ServerConfig, timeout: float = 5.0) -> list[str]:
    """List models available on a server."""
    try:
        if server.backend == "ollama":
            url = f"{server.url}/api/tags"
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=timeout)
            data = json.loads(resp.read())
            return [m.get("name", "") for m in data.get("models", [])]
        else:
            url = f"{server.url}/v1/models"
            headers = {}
            if server.api_key:
                headers["Authorization"] = f"Bearer {server.api_key}"
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=timeout)
            data = json.loads(resp.read())
            return [m.get("id", "") for m in data.get("data", [])]
    except Exception as e:
        logger.debug("Failed to list models from %s: %s", server.url, e)
        return []


# ============================================================
# Chat Client
# ============================================================

def chat(
    server: ServerConfig,
    messages: list[ChatMessage],
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> ChatResponse:
    """Send a chat request to an LLM server."""
    response = ChatResponse()

    try:
        if server.backend == "ollama":
            return _chat_ollama(server, messages, temperature)
        else:
            return _chat_openai(server, messages, temperature, max_tokens)
    except urllib.error.URLError as e:
        response.error = f"Connection failed: {e}"
    except Exception as e:
        response.error = f"Error: {e}"
    return response


def _chat_ollama(server: ServerConfig, messages: list[ChatMessage], temperature: float) -> ChatResponse:
    """Chat via Ollama API."""
    payload = json.dumps({
        "model": server.model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": False,
        "options": {"temperature": temperature},
    }).encode()

    req = urllib.request.Request(
        server.chat_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())

    return ChatResponse(
        content=data.get("message", {}).get("content", ""),
        model=server.model,
        tokens_used=data.get("eval_count", 0),
    )


def _chat_openai(server: ServerConfig, messages: list[ChatMessage], temperature: float, max_tokens: int) -> ChatResponse:
    """Chat via OpenAI-compatible API (llama.cpp, vLLM, TGI)."""
    payload = json.dumps({
        "model": server.model or "default",
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()

    headers = {"Content-Type": "application/json"}
    if server.api_key:
        headers["Authorization"] = f"Bearer {server.api_key}"

    req = urllib.request.Request(server.chat_url, data=payload, headers=headers)
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())

    choice = data.get("choices", [{}])[0]
    usage = data.get("usage", {})

    return ChatResponse(
        content=choice.get("message", {}).get("content", ""),
        model=data.get("model", server.model),
        tokens_used=usage.get("total_tokens", 0),
    )


# ============================================================
# Direct Model Loading (HuggingFace)
# ============================================================

class LocalModel:
    """Wrapper for a locally loaded HuggingFace model."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = ""
        self.device = "cpu"
        self.loaded = False

    def load(self, model_name: str, trust_remote_code: bool = False) -> str:
        """Load a model with OverflowML's optimal strategy. Returns status message."""
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                from overflowml import load_model
            self.model, self.tokenizer = load_model(
                model_name, trust_remote_code=trust_remote_code,
            )
            self.model_name = model_name
            self.loaded = True
            if hasattr(self.model, "device"):
                self.device = str(self.model.device)
            return f"Loaded {model_name} on {self.device}"
        except Exception as e:
            return f"Failed to load {model_name}: {e}"

    def chat(self, user_message: str, max_new_tokens: int = 256) -> ChatResponse:
        """Generate a response from the loaded model."""
        if not self.loaded:
            return ChatResponse(error="No model loaded")
        try:
            import torch
            inputs = self.tokenizer(user_message, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=0.7,
                )
            text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            return ChatResponse(content=text, model=self.model_name)
        except Exception as e:
            return ChatResponse(error=f"Generation failed: {e}")

    def unload(self):
        """Free the model from memory."""
        if self.model:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.loaded = False
            try:
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
