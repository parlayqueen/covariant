from pathlib import Path
from dotenv import load_dotenv

def load_env():
    """
    Deterministic .env loader.

    Works in:
    - normal python execution
    - REPL
    - heredoc (python - <<'PY')
    - cron / background jobs
    """

    root = Path(__file__).resolve().parent
    env_path = root / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        raise FileNotFoundError(f".env not found at {env_path}")
