# New file: model_adapters.py
# Lightweight adapter layer to allow using different LLM backends (OpenAI, HF Inference API, local transformers, llama.cpp, Qwen, DeepSeek).
# Designed to be called from pipelines (e.g. sparrow_parse) with a config dict or option string.
"""
Model Adapters for Sparrow LLM Engine

This module provides a unified interface for different LLM backends, allowing easy integration
with various AI model providers.

Supported Adapters:
    - OpenAI: OpenAI API (GPT-4, GPT-3.5, etc.)
    - QwenAdapter: Alibaba Qwen (阿里千问) via DashScope API
    - DeepSeekAdapter: DeepSeek API (deepseek-chat, deepseek-coder)
    - HFInferenceAPIAdapter: Hugging Face Inference API
    - HFLocalAdapter: Local Hugging Face transformers
    - LlamaCppAdapter: Local llama.cpp models

Usage:
    # Using get_adapter with dict config
    config = {"method": "qwen", "model_name": "qwen-plus", "qwen_api_key": "your-key"}
    adapter = get_adapter(config)
    result = adapter.generate("Your prompt here")
    
    # Using get_adapter with string format
    adapter = get_adapter("qwen:qwen-plus")  # Requires DASHSCOPE_API_KEY in environment
    result = adapter.generate("Your prompt here")
"""
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
        # Validate prompt
        if prompt is None or (isinstance(prompt, str) and not prompt.strip()):
            raise ValueError(
                "OpenAI adapter requires a non-empty prompt. "
                "Please provide a text query instead of using '*' (query all data)."
            )
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0
        }
        
        try:
            r = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Try to extract error message from response
            error_detail = ""
            try:
                error_json = r.json()
                if "error" in error_json:
                    if isinstance(error_json["error"], dict) and "message" in error_json["error"]:
                        error_detail = f" - {error_json['error']['message']}"
                    else:
                        error_detail = f" - {error_json['error']}"
                else:
                    error_detail = f" - {json.dumps(error_json)}"
            except:
                error_detail = f" - {r.text[:200]}"
            raise RuntimeError(f"OpenAI API request failed: {e}{error_detail}") from e
        
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
        # Validate prompt
        if prompt is None or (isinstance(prompt, str) and not prompt.strip()):
            raise ValueError(
                "Hugging Face Inference API adapter requires a non-empty prompt. "
                "Please provide a text query instead of using '*' (query all data)."
            )
        
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {"inputs": prompt}
        
        try:
            r = requests.post(self.endpoint, json=payload, headers=headers, timeout=120)
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Try to extract error message from response
            error_detail = ""
            try:
                error_json = r.json()
                if "error" in error_json:
                    error_detail = f" - {error_json['error']}"
                else:
                    error_detail = f" - {json.dumps(error_json)}"
            except:
                error_detail = f" - {r.text[:200]}"
            raise RuntimeError(f"Hugging Face API request failed: {e}{error_detail}") from e
        
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


class QwenAdapter(ModelAdapter):
    """
    Adapter for Alibaba Qwen (DashScope API).
    Requires DASHSCOPE_API_KEY environment variable to be set.
    """
    def __init__(self, model: str, api_key: Optional[str] = None):
        if requests is None:
            raise RuntimeError("requests is required for QwenAdapter. Install requests.")
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY or QWEN_API_KEY not set in environment")
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    def generate(self, prompt: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Validate prompt
        if prompt is None or (isinstance(prompt, str) and not prompt.strip()):
            raise ValueError(
                "Qwen adapter requires a non-empty prompt. "
                "The adapter does not support document/image inputs directly. "
                "Please provide a text query instead of using '*' (query all data)."
            )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "result_format": "message"
            }
        }
        
        try:
            r = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Try to extract error message from response
            error_detail = ""
            try:
                error_json = r.json()
                if "message" in error_json:
                    error_detail = f" - {error_json['message']}"
                elif "error" in error_json:
                    error_detail = f" - {error_json['error']}"
                else:
                    error_detail = f" - {json.dumps(error_json)}"
            except:
                error_detail = f" - {r.text[:200]}"
            raise RuntimeError(f"Qwen API request failed: {e}{error_detail}") from e
        
        j = r.json()
        
        # DashScope API response format
        if "output" in j and "choices" in j["output"] and len(j["output"]["choices"]) > 0:
            content = j["output"]["choices"][0].get("message", {}).get("content", "")
        elif "output" in j and "text" in j["output"]:
            content = j["output"]["text"]
        else:
            content = json.dumps(j)
        
        return {"text": content, "raw": j}


class DeepSeekAdapter(ModelAdapter):
    """
    Adapter for DeepSeek API.
    DeepSeek uses an OpenAI-compatible API format.
    Requires DEEPSEEK_API_KEY environment variable to be set.
    """
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if requests is None:
            raise RuntimeError("requests is required for DeepSeekAdapter. Install requests.")
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set in environment")
        self.base_url = base_url or "https://api.deepseek.com/v1/chat/completions"

    def generate(self, prompt: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Validate prompt
        if prompt is None or (isinstance(prompt, str) and not prompt.strip()):
            raise ValueError(
                "DeepSeek adapter requires a non-empty prompt. "
                "Please provide a text query instead of using '*' (query all data)."
            )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0,
            "max_tokens": 1024
        }
        
        try:
            r = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Try to extract error message from response
            error_detail = ""
            try:
                error_json = r.json()
                if "error" in error_json:
                    if isinstance(error_json["error"], dict) and "message" in error_json["error"]:
                        error_detail = f" - {error_json['error']['message']}"
                    else:
                        error_detail = f" - {error_json['error']}"
                else:
                    error_detail = f" - {json.dumps(error_json)}"
            except:
                error_detail = f" - {r.text[:200]}"
            raise RuntimeError(f"DeepSeek API request failed: {e}{error_detail}") from e
        
        j = r.json()
        
        # OpenAI-compatible response format
        if "choices" in j and len(j["choices"]) > 0:
            content = j["choices"][0].get("message", {}).get("content") or j["choices"][0].get("text", "")
        else:
            content = json.dumps(j)
        
        return {"text": content, "raw": j}


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
        if method.lower() == "qwen":
            return QwenAdapter(model=model, api_key=cfg.get("qwen_api_key") or cfg.get("dashscope_api_key"))
        if method.lower() == "deepseek":
            return DeepSeekAdapter(model=model, api_key=cfg.get("deepseek_api_key"), base_url=cfg.get("base_url"))
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
            if prefix == "qwen":
                return QwenAdapter(model=val)
            if prefix == "deepseek":
                return DeepSeekAdapter(model=val)
        raise ValueError("Unknown option string format. Use openai:<model>, hfapi:<model>, hf:<model>, llamacpp:<path>, qwen:<model>, deepseek:<model>.")
