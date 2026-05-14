from tracker.llm_clients import (
    LLMResponse,
    BaseLLMClient,
    AnthropicClient,
    OpenAIClient,
    GeminiClient,
    OpenRouterClient,
    MistralClient,
    QwenClient,
    HuggingFaceClient,
    OllamaClient,
    ModelConfig,
    AVAILABLE_MODELS,
    DAILY_MODELS,
    build_selected_clients,
    build_all_clients,
)
from tracker.debate import (
    Debate,
    DebatePhase,
    DebateTurn,
    GovernmentPlanTopic,
    GovernmentPlanSection,
    GovernmentPlanUpload,
    PresidentialElection,
    build_presidential_election,
    save_debate,
)
from tracker.candidates import CANDIDATES, CANDIDATE_BY_KEY, Candidate
from tracker.prompts import Prompt, ALL_PROMPTS, DAILY_PROMPTS
from tracker.storage import TrackerRecord, build_record, save_records, append_to_timeseries

__version__ = "0.2.0"

# ── Public API ─────────────────────────────────────────────────────────────

__all__ = [
    # LLM Clients
    "LLMResponse",
    "BaseLLMClient",
    "AnthropicClient",
    "OpenAIClient",
    "GeminiClient",
    "OpenRouterClient",
    "MistralClient",
    "QwenClient",
    "HuggingFaceClient",
    "OllamaClient",
    "ModelConfig",
    "AVAILABLE_MODELS",
    "DAILY_MODELS",
    "build_selected_clients",
    "build_all_clients",
    
    # Debate
    "Debate",
    "DebatePhase",
    "DebateTurn",
    "GovernmentPlanTopic",
    "GovernmentPlanSection",
    "GovernmentPlanUpload",
    "PresidentialElection",
    "build_presidential_election",
    "save_debate",
    
    # Candidates
    "CANDIDATES",
    "CANDIDATE_BY_KEY",
    "Candidate",
    
    # Prompts
    "Prompt",
    "ALL_PROMPTS",
    "DAILY_PROMPTS",
    
    # Storage
    "TrackerRecord",
    "build_record",
    "save_records",
    "append_to_timeseries",
]

# ── Regional Classification ────────────────────────────────────────────────
#
# Models grouped by region for analysis:
#
# REGION_US = ["claude", "gpt4o", "gemini"]
# REGION_CHINA = ["qwen_direct", "qwen_or", "deepseek", "ling", "baichuan"]
# REGION_EUROPE = ["mistral_large", "mistral_small"]
# REGION_LATAM = ["llama_8b"]
# REGION_LOCAL = ["gemma2_local", "gemma4_local"]
#
# Cross-provider comparison pairs:
# - ("qwen_direct", "qwen_or") — same model, different API path
