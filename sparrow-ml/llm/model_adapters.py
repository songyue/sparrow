# New file: model_adapters.py
# Lightweight adapter layer to allow using different LLM backends (OpenAI, HF Inference API, local transformers, llama.cpp).
# Designed to be called from pipelines (e.g. sparrow_parse) with a config dict or option string.
import os
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

# Use requests to keep dependencies minimal for remote APIs
try:
    import requests
except Exception:
    requests = None

class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a dict with at least a 'text' key."""
        raise NotImplementedError()


class OpenAIAdapter(ModelAdapter):
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if requests is None:
            raise RuntimeError("requests is required for OpenAIAdapter. Install requests.")
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        self.base_url = base_url or "https://api.openai.com/v1/chat/completions"

    def generate(self, prompt: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0
        }
        r = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        j = r.json()
        # Try to extract text in common structures
        if "choices" in j and len(j["choices"]) > 0:
            content = j["choices"][0].get("message", {}).get("content") or j["choices"][0].get("text")
        else:
            content = j.get("text") or json.dumps(j)
        return {"text": content, "raw": j}


class HFInferenceAPIAdapter(ModelAdapter):
    def __init__(self, model: str, hf_token: Optional[str] = None):
        if requests is None:
            raise RuntimeError("requests is required for HF Inference API adapter. Install requests.")
        self.model = model
        self.hf_token = hf_token or os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise RuntimeError("HF_API_KEY or HF_TOKEN not set in environment")
        self.endpoint = f"https://api-inference.huggingface.co/models/{model}"

    def generate(self, prompt: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {"inputs": prompt}
        r = requests.post(self.endpoint, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        j = r.json()
        # HF inference might return plain text or structured output
        if isinstance(j, dict) and "error" in j:
            raise RuntimeError(f"Hugging Face inference error: {j['error']}")
        # If returned list of tokens or dict, try to extract text
        if isinstance(j, list) and len(j) > 0 and isinstance(j[0], dict) and "generated_text" in j[0]:
            text = j[0]["generated_text"]
        elif isinstance(j, str):
            text = j
        else:
            text = json.dumps(j)
        return {"text": text, "raw": j}


class HFLocalAdapter(ModelAdapter):
    def __init__(self, model: str, device: str = "cpu"):
        # Lazy imports
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            raise RuntimeError("transformers is required for HFLocalAdapter. Install transformers.") from e
        # Create text-generation pipeline (simple)
        self.model = model
        self.device = -1 if device == "cpu" else 0
        self.generator = pipeline("text-generation", model=model, device=self.device)

    def generate(self, prompt: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = self.generator(prompt, max_new_tokens=512, do_sample=False)
        text = out[0].get("generated_text") if isinstance(out, list) else str(out)
        return {"text": text, "raw": out}


class LlamaCppAdapter(ModelAdapter):
    def __init__(self, model_path: str, n_ctx: int = 2048):
        try:
            from llama_cpp import Llama
        except Exception as e:
            raise RuntimeError("llama-cpp-python is required for LlamaCppAdapter. Install llama-cpp-python.") from e
        self.client = Llama(model_path=model_path, n_ctx=n_ctx)

    def generate(self, prompt: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        r = self.client.create(prompt=prompt, max_tokens=512, stop=None)
        # r typically has 'choices' list with 'text' key
        text = ""
        if isinstance(r, dict) and "choices" in r and len(r["choices"]) > 0:
            text = r["choices"][0].get("text", "")
        else:
            text = str(r)
        return {"text": text, "raw": r}


def get_adapter(option_or_config: Any) -> ModelAdapter:
    """
    Accepts either:
      - option_or_config: str like "openai:gpt-4o" or "hfapi:username/model"
      - or config dict like {"method": "openai", "model_name": "gpt-4o", ...}
    Returns a ModelAdapter instance.
    """
    # If dict, normalize to method/model_name pattern
    if isinstance(option_or_config, dict):
        cfg = option_or_config
        method = cfg.get("method")
        model = cfg.get("model_name") or cfg.get("hf_space") or cfg.get("model")
        if method is None or model is None:
            raise ValueError("Config dict must contain 'method' and 'model_name' or equivalent.")
        if method.lower() == "openai":
            return OpenAIAdapter(model=model, api_key=cfg.get("openai_key"), base_url=cfg.get("base_url"))
        if method.lower() == "hfapi":
            return HFInferenceAPIAdapter(model=model, hf_token=cfg.get("hf_token"))
        if method.lower() == "hf_local":
            device = cfg.get("device", "cpu")
            return HFLocalAdapter(model=model, device=device)
        if method.lower() == "llamacpp":
            return LlamaCppAdapter(model_path=model, n_ctx=cfg.get("n_ctx", 2048))
        # Fallback: try to interpret 'method' as huggingface/local
        raise ValueError(f"Unknown adapter method: {method}")

    if isinstance(option_or_config, str):
        if ":" in option_or_config:
            prefix, val = option_or_config.split(":", 1)
            prefix = prefix.lower()
            if prefix == "openai":
                return OpenAIAdapter(model=val)
            if prefix in ("hfapi", "huggingface-api"):
                return HFInferenceAPIAdapter(model=val)
            if prefix in ("hf", "hf_local", "huggingface"):
                return HFLocalAdapter(model=val)
            if prefix in ("llamacpp", "llama"):
                return LlamaCppAdapter(model_path=val)
        raise ValueError("Unknown option string format. Use openai:<model>, hfapi:<model>, hf:<model>, llamacpp:<path>.")
