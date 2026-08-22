"""Environment-aware configuration shared by backend services."""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

LOCAL = "LOCAL"
PROD = "PROD"


def get_app_env() -> str:
    """Return the supported application environment or fail fast."""
    app_env = os.getenv("APP_ENV", LOCAL).strip().upper()
    if app_env not in {LOCAL, PROD}:
        raise RuntimeError("APP_ENV must be either 'LOCAL' or 'PROD'.")
    return app_env


APP_ENV = get_app_env()
IS_PRODUCTION = APP_ENV == PROD
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_UPLOAD_DIR = PROJECT_ROOT / "uploads"


def get_service_setting(name: str, default: str | None = None) -> str | None:
    """Read an environment-specific setting before its legacy shared setting.

    For example, in PROD this checks ``QDRANT_URL_PROD`` and then
    ``QDRANT_URL``. This keeps the current deployment configuration working
    while allowing LOCAL and PROD values to be configured independently.
    """
    return os.getenv(f"{name}_{APP_ENV}") or os.getenv(name, default)
