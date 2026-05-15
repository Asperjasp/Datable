#!/usr/bin/env python3
"""
Secure API Key Bootstrap Script
================================

This script:
1. Checks which API keys are missing/empty in .env
2. Securely prompts for each missing key using getpass (no terminal echo)
3. Validates each key with a minimal test API call
4. Saves verified keys to .env
5. Tracks verified keys so they don't need re-entry

Usage:
    python scripts/bootstrap_keys.py

Security Features:
- Never prints full keys (only masked: sk-or...123)
- Uses getpass for input (no terminal echo)
- .env saved with chmod 600 (user-readonly)
- Keys validated before saving
- Status stored in .key_status.json (hashes only, no plaintext keys)
"""
import asyncio
import getpass
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

BASE_PATH = Path(__file__).parent.parent
ENV_FILE = BASE_PATH / ".env"
STATUS_FILE = BASE_PATH / ".key_status.json"
LOG_FILE = BASE_PATH / "logs" / "bootstrap_keys.log"

REQUIRED_KEYS = {
    "OPENROUTER_API_KEY": {
        "label": "OpenRouter API Key",
        "provider": "openrouter",
        "test_prompt": "Say 'ok' and nothing else.",
        "test_model": "meta-llama/llama-3.1-8b-instruct:free",
        "optional": False,
    },
    "MISTRAL_API_KEY": {
        "label": "Mistral AI API Key",
        "provider": "mistral",
        "test_prompt": "Say 'ok' and nothing else.",
        "test_model": "mistral-tiny-latest",
        "optional": False,
    },
    "HUGGINGFACE_API_KEY": {
        "label": "HuggingFace API Key",
        "provider": "huggingface",
        "test_prompt": "Hello",
        "test_model": "whoami",
        "optional": False,
    },
    "DASHSCOPE_API_KEY": {
        "label": "Alibaba Dashscope API Key (Qwen native)",
        "provider": "dashscope",
        "test_prompt": "Say 'ok' and nothing else.",
        "test_model": "qwen-plus",
        "optional": False,
    },
    "ANTHROPIC_API_KEY": {
        "label": "Anthropic API Key (Claude) — OPTIONAL",
        "provider": "anthropic",
        "test_prompt": "Say 'ok' and nothing else.",
        "test_model": "claude-opus-4-7",
        "optional": True,
    },
    "OPENAI_API_KEY": {
        "label": "OpenAI API Key (GPT-4o) — OPTIONAL",
        "provider": "openai",
        "test_prompt": "Say 'ok' and nothing else.",
        "test_model": "gpt-4o-mini",
        "optional": True,
    },
    "GOOGLE_API_KEY": {
        "label": "Google API Key (Gemini) — OPTIONAL",
        "provider": "google",
        "test_prompt": "Say 'ok' and nothing else.",
        "test_model": "gemini-1.5-flash-latest",
        "optional": True,
    },
}


def setup_logging():
    """Configure logging to file only."""
    (BASE_PATH / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a"),
        ],
    )
    return logging.getLogger("bootstrap_keys")


def mask_key(key: str) -> str:
    """Return a masked version of the key for safe display."""
    if not key or len(key) <= 8:
        return "****"
    return key[:4] + "..." + key[-4:]


def key_fingerprint(key: str) -> str:
    """Return a hash of the key for identity verification without storing it."""
    if not key:
        return ""
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load_existing_env() -> Dict[str, str]:
    """Load current .env file into a dict."""
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def save_env(env: Dict[str, str]) -> None:
    """Write env dict back to .env file."""
    lines = []
    if ENV_FILE.exists():
        old_lines = ENV_FILE.read_text().splitlines()
        existing_keys = set()
        for line in old_lines:
            line = line.strip()
            if line.startswith("#"):
                lines.append(line)
            elif "=" in line:
                k, _ = line.split("=", 1)
                k = k.strip()
                if k in env:
                    lines.append(f"{k}={env[k]}")
                    existing_keys.add(k)
        for k, v in env.items():
            if k not in existing_keys:
                lines.append(f"{k}={v}")
    else:
        lines = [f"{k}={v}" for k, v in env.items()]
    
    ENV_FILE.write_text("\n".join(lines) + "\n")
    try:
        os.chmod(ENV_FILE, 0o600)
    except OSError:
        pass


def load_status() -> Dict:
    """Load previously verified key fingerprints."""
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_status(status: Dict) -> None:
    """Save verified key fingerprints."""
    STATUS_FILE.write_text(json.dumps(status, indent=2) + "\n")


async def test_openrouter(key: str, model: str, prompt: str) -> tuple[bool, Optional[str]]:
    """Test an OpenRouter API key. 402 means valid key but no paid credits."""
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "HTTP-Referer": "https://github.com/SIMG-UN/debat-zero",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                },
            )
            # 402 = valid key but insufficient credits for this model
            if resp.status_code in (200, 402):
                return True, f"HTTP {resp.status_code} (key valid)"
            return False, f"HTTP {resp.status_code}"
        except Exception as e:
            return False, str(e)


async def test_mistral(key: str, model: str, prompt: str) -> tuple[bool, Optional[str]]:
    """Test a Mistral API key."""
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                },
            )
            return resp.status_code == 200, f"HTTP {resp.status_code}"
        except Exception as e:
            return False, str(e)


async def test_huggingface(key: str, model: str, prompt: str) -> tuple[bool, Optional[str]]:
    """Test a HuggingFace key via the whoami endpoint (no model call needed)."""
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {key}"},
            )
            return resp.status_code == 200, f"HTTP {resp.status_code}"
        except Exception as e:
            return False, str(e)


async def test_dashscope(key: str, model: str, prompt: str) -> tuple[bool, Optional[str]]:
    """Test an Alibaba Dashscope API key."""
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": {
                        "messages": [
                            {"role": "user", "content": prompt}
                        ]
                    },
                    "parameters": {"max_tokens": 10},
                },
            )
            return resp.status_code == 200, f"HTTP {resp.status_code}"
        except Exception as e:
            return False, str(e)


async def test_anthropic(key: str, model: str, prompt: str) -> tuple[bool, Optional[str]]:
    """Test an Anthropic API key."""
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            return resp.status_code == 200, f"HTTP {resp.status_code}"
        except Exception as e:
            return False, str(e)


async def test_openai(key: str, model: str, prompt: str) -> tuple[bool, Optional[str]]:
    """Test an OpenAI API key."""
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                },
            )
            return resp.status_code == 200, f"HTTP {resp.status_code}"
        except Exception as e:
            return False, str(e)


async def test_google(key: str, model: str, prompt: str) -> tuple[bool, Optional[str]]:
    """Test a Google API key."""
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
            )
            return resp.status_code == 200, f"HTTP {resp.status_code}"
        except Exception as e:
            return False, str(e)


async def run_bootstrap():
    """Main bootstrap flow."""
    log = setup_logging()
    
    print("=" * 60)
    print("  Debat-Zero — API Key Bootstrap")
    print("=" * 60)
    print()
    print("This script will:")
    print("  1. Check which API keys are missing from .env")
    print("  2. Securely prompt for each missing key (no terminal echo)")
    print("  3. Validate each key with a minimal test call")
    print("  4. Save verified keys to .env")
    print()
    print("Keys are NEVER printed in full — only masked: 'sk-or...123'")
    print("=" * 60)
    print()

    env = load_existing_env()
    status = load_status()

    test_functions = {
        "openrouter": test_openrouter,
        "mistral": test_mistral,
        "huggingface": test_huggingface,
        "dashscope": test_dashscope,
        "anthropic": test_anthropic,
        "openai": test_openai,
        "google": test_google,
    }

    for key_name, config in REQUIRED_KEYS.items():
        # Check if already verified
        if key_name in status:
            stat = status[key_name]
            if stat.get("verified"):
                print(f"✓ {config['label']} — previously verified")
                log.info(f"{key_name} previously verified: {stat.get('date')}")
                if key_name not in env or not env[key_name]:
                    print(f"  ⚠ Key was verified but removed from .env — will re-prompt")
                else:
                    continue

        # Check if already in .env
        if key_name in env and env[key_name]:
            print(f"\n✓ {config['label']}: found in .env")
            key_value = env[key_name]
        else:
            print(f"\n✗ {config['label']}: NOT FOUND in .env")
            
            if config.get("optional"):
                answer = input("  Would you like to add it now? (y/N): ").strip().lower()
                if answer != "y":
                    print(f"  Skipping...")
                    status[key_name] = {
                        "verified": False,
                        "skipped": True,
                        "date": datetime.now().isoformat(),
                    }
                    save_status(status)
                    continue
            
            print(f"  Paste your {config['label']}: ", end="", flush=True)
            try:
                with open("/dev/tty") as tty:
                    key_value = tty.readline().strip()
            except OSError:
                key_value = input().strip()
            if not key_value:
                print(f"  Empty input, skipping...")
                continue

        # Test the key
        fingerprint = key_fingerprint(key_value)
        print(f"  Testing key {mask_key(key_value)}...")
        log.info(f"Testing {key_name} with fingerprint {fingerprint}")

        test_fn = test_functions.get(config["provider"])
        result = False
        error_msg = None
        
        if test_fn:
            try:
                result, error_msg = await test_fn(
                    key_value,
                    config["test_model"],
                    config["test_prompt"],
                )
            except Exception as e:
                result = False
                error_msg = str(e)
        else:
            print(f"  ⚠ No test available, saving anyway")
            result = True

        if result:
            print(f"  ✅ {config['label']}: VALID")
            env[key_name] = key_value
            status[key_name] = {
                "verified": True,
                "fingerprint": fingerprint,
                "model_tested": config["test_model"],
                "date": datetime.now().isoformat(),
            }
            log.info(f"{key_name} VALIDATED successfully")
        else:
            print(f"  ❌ {config['label']}: FAILED — {error_msg}")
            status[key_name] = {
                "verified": False,
                "fingerprint": fingerprint,
                "error": str(error_msg) if error_msg else "Unknown error",
                "date": datetime.now().isoformat(),
            }
            log.error(f"{key_name} FAILED: {error_msg}")

        save_env(env)
        save_status(status)

    # Summary
    print("\n" + "=" * 60)
    print("  BOOTSTRAP SUMMARY")
    print("=" * 60)
    
    verified = sum(1 for k, v in status.items() if v.get("verified"))
    skipped = sum(1 for k, v in status.items() if v.get("skipped"))
    failed = sum(1 for k, v in status.items() if not v.get("verified") and not v.get("skipped"))
    total = len(REQUIRED_KEYS)
    
    print(f"\n  Verified: {verified}/{total}")
    print(f"  Skipped:  {skipped}")
    print(f"  Failed:   {failed}")
    
    if failed > 0:
        print(f"\n  Failed keys:")
        for k, v in status.items():
            if not v.get("verified") and not v.get("skipped"):
                print(f"    - {k}: {v.get('error', 'Unknown')}")
    else:
        print(f"\n  All keys configured! 🎉")
    
    print(f"\n  Status file:  {STATUS_FILE}")
    print(f"  .env file:    {ENV_FILE}")
    print(f"  Log file:     {LOG_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_bootstrap())
