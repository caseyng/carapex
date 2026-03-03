"""
backends/base.py
----------------
Abstract base and minimal shared config for LLM backends.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO ADD A NEW BACKEND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Create a file in backends/  e.g. backends/my_backend.py
2. Subclass LLMBackend
3. Set class attribute `name` — this is the registry key
4. Implement chat()
5. Done — registry autodiscovers it, config picks it up

MINIMAL BACKEND (no extra config fields):

    class MyBackend(LLMBackend):
        name = "my_backend"

        def chat(self, system_prompt, user_prompt):
            ...  # call your LLM, return str or None

    from_config() and default_config() are inherited.
    Your backend appears in config.json automatically.

BACKEND WITH EXTRA CONFIG FIELDS:

    @dataclass
    class MyConfig(BackendConfig):
        my_field: str = "default_value"

    class MyBackend(LLMBackend):
        name = "my_backend"

        @classmethod
        def from_config(cls, raw: dict) -> "MyBackend":
            import dataclasses
            known    = {f.name for f in dataclasses.fields(MyConfig)}
            # "type" is a routing key — filter it out before passing to the dataclass.
            filtered = {k: v for k, v in raw.items() if k in known and v is not None}
            return cls(MyConfig(**filtered))

        @classmethod
        def default_config(cls) -> dict:
            base = super().default_config()  # includes "type": cls.name automatically
            base["my_field"] = "default_value"
            return base

        def chat(self, system_prompt, user_prompt):
            ...

RULES:
    - chat() must always be implemented — no universal default exists
    - Return None for operational failures (connection error, timeout,
      HTTP error, unparseable response) — log before returning
    - Allow programming errors to propagate — do not swallow bugs
    - from_config() — override only if you have a custom config type;
      always filter out "type" (routing key) before passing to your dataclass
    - default_config() — call super(); it includes "type": cls.name automatically
    - Never edit existing files to register a new backend
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class BackendConfig:
    """
    Minimal shared config for all backends.
    Backend-specific fields belong in a subclass in the backend's own file.
    """
    base_url:          str           = "http://localhost:8080"
    api_key:           Optional[str] = None
    temperature:       Optional[float] = None   # None = use model default
    max_tokens:        int           = 1500
    request_timeout_s: int           = 120

    def __post_init__(self):
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be 0.0-2.0, got {self.temperature}"
            )


class LLMBackend(ABC):
    """
    Abstract base for all LLM backends.
    See module docstring for full extension guide.
    """

    name: str  # registry key — must be set on every subclass

    @classmethod
    def from_config(cls, raw: dict) -> "LLMBackend":
        """
        Instantiate from raw config dict.
        Default: constructs with BackendConfig, ignoring unknown keys.
        "type" is a routing key consumed by provide_backend() — filtered here
        so it does not reach BackendConfig which has no such field.
        Override if your backend uses a custom config dataclass.
        """
        known    = {f.name for f in dataclasses.fields(BackendConfig)}
        filtered = {k: v for k, v in raw.items() if k in known and v is not None}
        return cls(BackendConfig(**filtered))

    @classmethod
    def default_config(cls) -> dict:
        """
        Return config keys with defaults for config.json generation.
        Includes "type" set to cls.name — the routing key required by provide_backend().
        Override and call super() to add backend-specific fields.
        """
        result = {"type": cls.name}
        for f in dataclasses.fields(BackendConfig):
            if f.default is not dataclasses.MISSING:
                result[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:
                result[f.name] = f.default_factory()
        return result

    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        user_prompt:   str,
        temperature:   Optional[float] = None,
    ) -> Optional[str]:
        """
        Send prompts to the LLM, return response string or None on failure.
        Must be implemented — no universal default exists.

        Returns None for expected operational failures (connection error,
        timeout, HTTP error, unparseable response). Allows programming
        errors and unexpected exceptions to propagate — they are signals,
        not noise.

        Args:
            temperature : per-call override. None uses backend configured default.
                          Pass 0.1 for classification tasks (guard model).
                          Zero overhead — temperature is a per-request parameter.
        """
        ...

    def health_check(self) -> bool:
        """Override to verify backend is reachable."""
        return True

    def last_usage(self) -> dict:
        """
        Override to return token usage from last chat() call.
        Expected keys: prompt_tokens, completion_tokens, total_tokens.
        """
        return {}

    def close(self) -> None:
        """Override to release resources at shutdown."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
