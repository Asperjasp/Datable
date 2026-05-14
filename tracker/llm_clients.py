import asyncio
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class LLMResponse:
    raw_text: str
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseLLMClient(ABC):
    provider: str
    model_id: str

    @abstractmethod
    async def query(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, system_prompt: Optional[str] = None) -> LLMResponse:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Cloud clients
# ──────────────────────────────────────────────────────────────────────────────

class AnthropicClient(BaseLLMClient):
    provider = "anthropic"

    def __init__(self, model: str = "claude-opus-4-7"):
        import anthropic
        self.model_id = model
        self._client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    async def query(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, system_prompt: Optional[str] = None) -> LLMResponse:
        t0 = time.monotonic()
        try:
            kwargs: dict = dict(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if system_prompt:
                kwargs["system"] = system_prompt
            msg = await self._client.messages.create(**kwargs)
            ms = (time.monotonic() - t0) * 1000
            return LLMResponse(
                raw_text=msg.content[0].text,
                model_id=self.model_id,
                provider=self.provider,
                input_tokens=msg.usage.input_tokens,
                output_tokens=msg.usage.output_tokens,
                latency_ms=ms,
            )
        except Exception as exc:
            return LLMResponse("", self.model_id, self.provider, 0, 0, (time.monotonic() - t0) * 1000, error=str(exc))


class OpenAIClient(BaseLLMClient):
    provider = "openai"

    def __init__(self, model: str = "gpt-4o"):
        import openai
        self.model_id = model
        self._client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def query(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, system_prompt: Optional[str] = None) -> LLMResponse:
        t0 = time.monotonic()
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            resp = await self._client.chat.completions.create(
                model=self.model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
            )
            ms = (time.monotonic() - t0) * 1000
            return LLMResponse(
                raw_text=resp.choices[0].message.content or "",
                model_id=self.model_id,
                provider=self.provider,
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                latency_ms=ms,
            )
        except Exception as exc:
            return LLMResponse("", self.model_id, self.provider, 0, 0, (time.monotonic() - t0) * 1000, error=str(exc))


class GeminiClient(BaseLLMClient):
    provider = "google"

    def __init__(self, model: str = "gemini-1.5-pro"):
        import google.generativeai as genai
        self.model_id = model
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self._model = genai.GenerativeModel(model)
        self._genai = genai

    async def query(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, system_prompt: Optional[str] = None) -> LLMResponse:
        t0 = time.monotonic()
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            config = self._genai.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
            response = await self._model.generate_content_async(full_prompt, generation_config=config)
            ms = (time.monotonic() - t0) * 1000
            usage = response.usage_metadata
            return LLMResponse(
                raw_text=response.text if response.text else "",
                model_id=self.model_id,
                provider=self.provider,
                input_tokens=usage.prompt_token_count if usage else 0,
                output_tokens=usage.candidates_token_count if usage else 0,
                latency_ms=ms,
            )
        except Exception as exc:
            return LLMResponse("", self.model_id, self.provider, 0, 0, (time.monotonic() - t0) * 1000, error=str(exc))


class OpenRouterClient(BaseLLMClient):
    """
    OpenAI-compatible wrapper for OpenRouter.
    Uses: Qwen, DeepSeek, Ling (Ant Group), Baichuan, etc.
    Cross-Provider: Run same model via both OpenRouter AND native for consistency checks.
    """
    provider = "openrouter"

    def __init__(self, model: str):
        import openai
        self.model_id = model
        self._client = openai.AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/SIMG-UN/debat-zero",
                "X-Title": "debat-zero LLM Bias Tracker",
            },
        )

    async def query(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, system_prompt: Optional[str] = None) -> LLMResponse:
        t0 = time.monotonic()
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            resp = await self._client.chat.completions.create(
                model=self.model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
            )
            ms = (time.monotonic() - t0) * 1000
            usage = resp.usage
            return LLMResponse(
                raw_text=resp.choices[0].message.content or "",
                model_id=self.model_id,
                provider=self.provider,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency_ms=ms,
            )
        except Exception as exc:
            return LLMResponse("", self.model_id, self.provider, 0, 0, (time.monotonic() - t0) * 1000, error=str(exc))


class MistralClient(BaseLLMClient):
    """
    Mistral AI API client.
    Provider: Mistral AI (France) — European model for regional comparison.
    """
    provider = "mistral"

    def __init__(self, model: str = "mistral-large-latest"):
        import httpx
        self.model_id = model
        self._api_key = os.environ["MISTRAL_API_KEY"]
        self._base_url = "https://api.mistral.ai/v1"
        self._client_lib = httpx

    async def query(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, system_prompt: Optional[str] = None) -> LLMResponse:
        t0 = time.monotonic()
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            async with self._client_lib.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model_id,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                data = resp.json()
                ms = (time.monotonic() - t0) * 1000
                
                if "choices" in data:
                    usage = data.get("usage", {})
                    return LLMResponse(
                        raw_text=data["choices"][0]["message"].get("content", "") or "",
                        model_id=self.model_id,
                        provider=self.provider,
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        latency_ms=ms,
                    )
                else:
                    return LLMResponse(
                        "", self.model_id, self.provider, 0, 0, ms,
                        error=f"Unexpected response: {data}"
                    )
        except Exception as exc:
            return LLMResponse("", self.model_id, self.provider, 0, 0, (time.monotonic() - t0) * 1000, error=str(exc))


class QwenClient(BaseLLMClient):
    """
    Alibaba Dashscope (Qwen) native API client.
    Provider: Alibaba Cloud (China) — Native Chinese model.
    
    Cross-Provider Consistency Note:
    Run the same Qwen model via both Dashscope (native) AND OpenRouter (proxy)
    to measure provider-level effects on response consistency.
    See paper/methodology.md for details.
    """
    provider = "dashscope"

    def __init__(self, model: str = "qwen-plus"):
        import httpx
        self.model_id = model
        self._api_key = os.environ["DASHSCOPE_API_KEY"]
        self._base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self._client_lib = httpx

    async def query(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, system_prompt: Optional[str] = None) -> LLMResponse:
        t0 = time.monotonic()
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            async with self._client_lib.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    self._base_url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model_id,
                        "input": {
                            "messages": messages
                        },
                        "parameters": {
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "result_format": "message",
                        },
                    },
                )
                data = resp.json()
                ms = (time.monotonic() - t0) * 1000
                
                if "output" in data and "choices" in data["output"]:
                    choices = data["output"]["choices"]
                    usage = data.get("usage", {})
                    return LLMResponse(
                        raw_text=choices[0]["message"].get("content", "") or "",
                        model_id=self.model_id,
                        provider=self.provider,
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        latency_ms=ms,
                    )
                else:
                    return LLMResponse(
                        "", self.model_id, self.provider, 0, 0, ms,
                        error=f"Unexpected response: {data}"
                    )
        except Exception as exc:
            return LLMResponse("", self.model_id, self.provider, 0, 0, (time.monotonic() - t0) * 1000, error=str(exc))


class HuggingFaceClient(BaseLLMClient):
    """
    HuggingFace Inference API client.
    Provider: HuggingFace — for open-source models like Llama 3.1, LATAM models.
    
    Used for:
    - Llama 3.1 8B (de facto regional model for Latin America)
    - Future LATAM GPT models when available
    - Any other open-source model on HF Hub
    """
    provider = "huggingface"

    def __init__(self, model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        import httpx
        self.model_id = model
        self._api_key = os.environ["HUGGINGFACE_API_KEY"]
        self._base_url = f"https://api-inference.huggingface.co/models/{model}"
        self._client_lib = httpx

    async def query(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, system_prompt: Optional[str] = None) -> LLMResponse:
        t0 = time.monotonic()
        try:
            full_input = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>" if system_prompt else prompt
            async with self._client_lib.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    self._base_url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "inputs": full_input,
                        "parameters": {
                            "max_new_tokens": max_tokens,
                            "temperature": temperature,
                            "return_full_text": False,
                        },
                    },
                )
                
                if resp.status_code == 503:
                    return LLMResponse(
                        "", self.model_id, self.provider, 0, 0, (time.monotonic() - t0) * 1000,
                        error="Model is cold starting. Please retry in 30-60 seconds."
                    )
                
                resp.raise_for_status()
                data = resp.json()
                ms = (time.monotonic() - t0) * 1000
                
                if isinstance(data, list) and len(data) > 0:
                    text = data[0].get("generated_text", "") or ""
                    return LLMResponse(
                        raw_text=text,
                        model_id=self.model_id,
                        provider=self.provider,
                        input_tokens=0,
                        output_tokens=len(text) // 4,
                        latency_ms=ms,
                    )
                elif isinstance(data, dict):
                    text = data.get("generated_text", "") or ""
                    return LLMResponse(
                        raw_text=text,
                        model_id=self.model_id,
                        provider=self.provider,
                        input_tokens=0,
                        output_tokens=len(text) // 4,
                        latency_ms=ms,
                    )
                else:
                    return LLMResponse(
                        "", self.model_id, self.provider, 0, 0, ms,
                        error=f"Unexpected response format: {type(data)}"
                    )
        except Exception as exc:
            return LLMResponse("", self.model_id, self.provider, 0, 0, (time.monotonic() - t0) * 1000, error=str(exc))


class OllamaClient(BaseLLMClient):
    """
    Local inference via Ollama — Gemma 2 9B or Gemma 4 local.
    Provider: Local (sovereignty comparison).
    """
    provider = "ollama"

    def __init__(self, model: str = "gemma2:9b", host: str = "http://localhost:11434"):
        import ollama
        self.model_id = model
        self._client = ollama.AsyncClient(host=host)

    async def query(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, system_prompt: Optional[str] = None) -> LLMResponse:
        t0 = time.monotonic()
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = await self._client.chat(
                model=self.model_id,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            ms = (time.monotonic() - t0) * 1000
            return LLMResponse(
                raw_text=response.message.content or "",
                model_id=self.model_id,
                provider=self.provider,
                input_tokens=response.prompt_eval_count or 0,
                output_tokens=response.eval_count or 0,
                latency_ms=ms,
            )
        except Exception as exc:
            return LLMResponse("", self.model_id, self.provider, 0, 0, (time.monotonic() - t0) * 1000, error=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Model registry — single source of truth
#
# Regional Classification:
# - US: Anthropic, OpenAI, Google (commercial)
# - China: Qwen (via OpenRouter + Dashscope for cross-provider check), DeepSeek, Ling, Baichuan
# - Europe: Mistral (France)
# - Latin America: Llama 3.1 (via HF), LATAM GPT (future)
# - Local: Ollama (sovereignty)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    key: str
    display_name: str
    provider: str
    group: str
    region: str
    env_key: Optional[str]
    factory: Callable[[], BaseLLMClient]


AVAILABLE_MODELS: dict[str, ModelConfig] = {
    # ── US Cloud Models ──────────────────────────────────────────────────────
    "claude": ModelConfig(
        key="claude",
        display_name="Claude Opus 4.7",
        provider="anthropic",
        group="Anthropic",
        region="us",
        env_key="ANTHROPIC_API_KEY",
        factory=AnthropicClient,
    ),
    "gpt4o": ModelConfig(
        key="gpt4o",
        display_name="GPT-4o",
        provider="openai",
        group="OpenAI",
        region="us",
        env_key="OPENAI_API_KEY",
        factory=OpenAIClient,
    ),
    "gemini": ModelConfig(
        key="gemini",
        display_name="Gemini 1.5 Pro",
        provider="google",
        group="Google",
        region="us",
        env_key="GOOGLE_API_KEY",
        factory=GeminiClient,
    ),
    
    # ── China Models (via OpenRouter) ─────────────────────────────────────────
    "qwen_or": ModelConfig(
        key="qwen_or",
        display_name="Qwen 2.5 72B (OpenRouter)",
        provider="openrouter",
        group="China (via OpenRouter)",
        region="china_proxy",
        env_key="OPENROUTER_API_KEY",
        factory=lambda: OpenRouterClient("qwen/qwen-2.5-72b-instruct"),
    ),
    "deepseek": ModelConfig(
        key="deepseek",
        display_name="DeepSeek V3 (OpenRouter)",
        provider="openrouter",
        group="China (via OpenRouter)",
        region="china_proxy",
        env_key="OPENROUTER_API_KEY",
        factory=lambda: OpenRouterClient("deepseek/deepseek-chat"),
    ),
    "ling": ModelConfig(
        key="ling",
        display_name="Ling Mini (Ant Group, OpenRouter)",
        provider="openrouter",
        group="China (via OpenRouter)",
        region="china_proxy",
        env_key="OPENROUTER_API_KEY",
        factory=lambda: OpenRouterClient("antgroup/ling-mini"),
    ),
    "baichuan": ModelConfig(
        key="baichuan",
        display_name="Baichuan 13B (OpenRouter)",
        provider="openrouter",
        group="China (via OpenRouter)",
        region="china_proxy",
        env_key="OPENROUTER_API_KEY",
        factory=lambda: OpenRouterClient("baichuan-inc/baichuan2-13b-chat"),
    ),
    
    # ── China Models (Direct/Native) — Cross-Provider Consistency Check ──────
    # [CROSS-PROVIDER-CONSISTENCY]
    # Run the same model via both native API AND OpenRouter to measure
    # provider-level effects. See paper/methodology.md for methodology.
    "qwen_direct": ModelConfig(
        key="qwen_direct",
        display_name="Qwen 2.5 (Dashscope Native)",
        provider="dashscope",
        group="China (Direct)",
        region="china_direct",
        env_key="DASHSCOPE_API_KEY",
        factory=lambda: QwenClient("qwen-plus"),
    ),
    
    # ── European Models ───────────────────────────────────────────────────────
    "mistral_large": ModelConfig(
        key="mistral_large",
        display_name="Mistral Large (France)",
        provider="mistral",
        group="Europe (Mistral)",
        region="europe",
        env_key="MISTRAL_API_KEY",
        factory=lambda: MistralClient("mistral-large-latest"),
    ),
    "mistral_small": ModelConfig(
        key="mistral_small",
        display_name="Mistral Small (France)",
        provider="mistral",
        group="Europe (Mistral)",
        region="europe",
        env_key="MISTRAL_API_KEY",
        factory=lambda: MistralClient("mistral-small-latest"),
    ),
    
    # ── Latin America / Open Source (HuggingFace) ────────────────────────────
    # LATAM GPT note: latam-gpt org on HF only has Wayra-Perplexity-Estimator-55M
    # which is a classification model, not a chat LLM. Use Llama 3.1 as the
    # de facto regional model until LATAM GPT releases their foundation model.
    "llama_8b": ModelConfig(
        key="llama_8b",
        display_name="Llama 3.1 8B (LATAM Proxy)",
        provider="huggingface",
        group="Latin America (HF)",
        region="latam",
        env_key="HUGGINGFACE_API_KEY",
        factory=lambda: HuggingFaceClient("meta-llama/Llama-3.1-8B-Instruct"),
    ),
    
    # ── Local Models (Sovereignty Comparison) ─────────────────────────────────
    "gemma2_local": ModelConfig(
        key="gemma2_local",
        display_name="Gemma 2 9B (local)",
        provider="ollama",
        group="Local (Ollama)",
        region="local",
        env_key=None,
        factory=lambda: OllamaClient(
            model=os.environ.get("OLLAMA_MODEL", "gemma2:9b"),
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        ),
    ),
    "gemma4_local": ModelConfig(
        key="gemma4_local",
        display_name="Gemma 4 12B (local)",
        provider="ollama",
        group="Local (Ollama)",
        region="local",
        env_key=None,
        factory=lambda: OllamaClient(
            model=os.environ.get("OLLAMA_MODEL_GEMMA4", "gemma4:12b"),
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        ),
    ),
}


# ── Recommended daily model set ───────────────────────────────────────────────
#
# Core set for WEIRD bias tracking:
# 1. Claude (US)
# 2. GPT-4o (US)
# 3. Gemini (US)
# 4. Mistral Large (Europe)
# 5. Qwen Direct (China)
# 6. Llama 3.1 8B (LATAM proxy)
# 7. Gemma 2 Local (sovereignty)
#
# Cross-provider comparison models (run alongside for consistency checks):
# - qwen_or (same model as qwen_direct, but via OpenRouter)

DAILY_MODELS = [
    "qwen_direct",
    "qwen_or",
    "mistral_large",
    "llama_8b",
    "deepseek",
    "gemma2_local",
]


def build_selected_clients(keys: list[str]) -> dict[str, BaseLLMClient]:
    import logging
    log = logging.getLogger(__name__)
    clients: dict[str, BaseLLMClient] = {}
    for key in keys:
        if key not in AVAILABLE_MODELS:
            log.warning(f"Unknown model key: {key}")
            continue
        cfg = AVAILABLE_MODELS[key]
        if cfg.env_key and not os.environ.get(cfg.env_key):
            log.warning(f"[SKIP] {key}: {cfg.env_key} not set")
            continue
        try:
            clients[key] = cfg.factory()
            log.info(f"[OK]   {key} ({clients[key].model_id})")
        except Exception as exc:
            log.error(f"[FAIL] {key}: {exc}")
    return clients


def build_all_clients() -> dict[str, BaseLLMClient]:
    return build_selected_clients(list(AVAILABLE_MODELS.keys()))
