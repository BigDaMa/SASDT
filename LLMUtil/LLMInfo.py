from __future__ import annotations

from pathlib import Path
from gdt.models import resolve_openai_model


class LLMInfo:
    _project_root = Path(__file__).resolve().parent.parent
    api_key = str(_project_root / "openaiapi.txt")
    novita_api_key = str(_project_root / "novitaapi.txt")

    DEFAULT_MODEL = "gpt-5.1"
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 262000
    DEFAULT_SEED = 42
    ENABLE_SEED = True
    DEFAULT_PROMPT = "label+string"

    @classmethod
    def set_api_key(cls, key_file: str | Path | None = None) -> None:
        cls.api_key = str(key_file or cls._project_root / "openaiapi.txt")

    @classmethod
    def set_novita_api_key(cls, key_file: str | Path | None = None) -> None:
        cls.novita_api_key = str(key_file or cls._project_root / "novitaapi.txt")

    @classmethod
    def set_default_model(cls, model: str = "gpt-5.1") -> None:
        cls.DEFAULT_MODEL = resolve_openai_model(model)

    @classmethod
    def set_default_temperature(cls, temperature: float = 0.0) -> None:
        cls.DEFAULT_TEMPERATURE = temperature

    @classmethod
    def set_default_max_tokens(cls, max_tokens: int = 262000) -> None:
        cls.DEFAULT_MAX_TOKENS = max_tokens

    @classmethod
    def set_default_seed(cls, seed: int = 42) -> None:
        cls.DEFAULT_SEED = seed

    @classmethod
    def set_default_prompt(cls, prompt: str = "label+string") -> None:
        if prompt not in {"label", "label+string"}:
            raise ValueError(f"unknown prompt format: {prompt}")
        cls.DEFAULT_PROMPT = prompt
