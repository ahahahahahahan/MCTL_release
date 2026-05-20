"""
API configuration.

Set the following environment variables before running (see .env.example):
  - MCTL_API_KEY   : Your OpenAI-compatible API key
  - MCTL_API_URL   : API endpoint URL (default: https://api.openai.com/v1/chat/completions)
  - MCTL_API_MODEL : Model name (default: gpt-4o)
"""
import os

API_KEY = os.environ.get("MCTL_API_KEY", "")
API_URL = os.environ.get("MCTL_API_URL", "https://api.openai.com/v1/chat/completions")
API_MODEL = os.environ.get("MCTL_API_MODEL", "gpt-4o")
API_TIMEOUT = int(os.environ.get("MCTL_API_TIMEOUT", "60"))

if not API_KEY:
    raise EnvironmentError(
        "MCTL_API_KEY is not set. "
        "Please copy .env.example to .env and fill in your API key, "
        "then run: export $(cat .env | xargs)"
    )
