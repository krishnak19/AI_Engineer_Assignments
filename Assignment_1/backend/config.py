"""Small, environment-based configuration for MediBot services."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load secrets from medibot/.env without overriding values set by the shell.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def get_groq_api_key() -> str:
    """Return the Groq secret without ever storing it in tracked source code."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to the local medibot/.env file."
        )
    return api_key


MEDIBOT_JWT_SECRET = os.getenv(
    'MEDIBOT_JWT_SECRET', 'development-only-secret-change-me-1234'
)
MEDIBOT_DEMO_PASSWORD = os.getenv('MEDIBOT_DEMO_PASSWORD', 'medibot-demo')

# Browser origin allowed to call the API during local frontend development.
MEDIBOT_FRONTEND_ORIGIN = os.getenv('MEDIBOT_FRONTEND_ORIGIN', 'http://localhost:3000')

