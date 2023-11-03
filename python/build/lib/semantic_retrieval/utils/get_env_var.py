import os
from typing import Optional


def get_env_var(key: str) -> Optional[str]:
    if key not in os.environ:
        return None
    return os.environ[key]
