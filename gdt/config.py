from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

from gdt.constants import RESERVED_EXAMPLE_COUNT
from gdt.models import resolve_openai_model


@dataclass(slots=True)
class ExperimentConfig:
    data_directory: Path = Path("data_shuffled")
    perfect_directory: Path = Path("data_perfect")
    cache_directory: Path = Path("cache")
    output_directory: Path = Path("results")
    use_cache: bool = False
    save_cache: bool = False
    perfect_cluster: bool = False
    perfect_split: bool = False
    perfect_operator: bool = False
    verbose: bool = False
    money_saver: bool = False
    language_model_batch: bool = False
    slm_split: bool = False
    num_retry: int | None = None
    query_individual_factor: float | None = None
    num_examples: int = 5
    timeout_seconds: int = 600
    model: str = "gpt-5.1"
    llm_model: str = "gpt-5.1"
    mode: str = "gen"
    cluster_config: str = "collapse+prompt_label_gpt"

    def __post_init__(self) -> None:
        self.model = resolve_openai_model(self.model)
        self.llm_model = resolve_openai_model(self.llm_model)
        if self.perfect_operator:
            self.perfect_cluster = True
            self.perfect_split = True
        if self.num_retry is None and self.query_individual_factor is None:
            self.query_individual_factor = 1.5

    def validate(self) -> None:
        if self.num_retry is not None and self.num_retry < 0:
            raise ValueError("num_retry must be non-negative")
        if self.num_retry is not None and self.query_individual_factor is not None:
            raise ValueError("num_retry and query_individual_factor are mutually exclusive")
        if self.query_individual_factor is not None and (
            not math.isfinite(self.query_individual_factor) or self.query_individual_factor < 1
        ):
            raise ValueError("query_individual_factor must be finite and at least 1")
        if self.num_examples < 0:
            raise ValueError("num_examples must be non-negative")
        if self.num_examples > RESERVED_EXAMPLE_COUNT:
            raise ValueError(f"num_examples must not exceed {RESERVED_EXAMPLE_COUNT}")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.use_cache and (
            self.perfect_cluster or self.perfect_split or self.perfect_operator
        ):
            raise ValueError("cached and perfect-data overrides cannot be combined")
        if self.perfect_operator:
            self.perfect_cluster = True
            self.perfect_split = True

    def as_display_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("slm_split")
        values["slm"] = values.pop("model")
        values["llm"] = values.pop("llm_model")
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in values.items()
        }


class Config:
    """Compatibility view of the active :class:`ExperimentConfig`.

    Old algorithms read class attributes such as ``Config.verbose``.  Keeping that
    surface avoids threading configuration through every mathematical helper while
    the CLI and orchestration layers use a proper immutable-at-the-boundary object.
    """

    use_cache = False
    save_cache = False
    cache_dir = "cache"
    perfect_cluster = False
    perfect_split = False
    perfect_operator = False
    perfect_directory = "data_perfect"
    verbose = False
    money_saver = False
    language_model_batch = False
    slm_split = False
    num_retry = 1
    query_individual_factor = 1.5
    data_directory = "data_shuffled"
    num_examples = 5
    model = "gpt-5.1"
    llm_model = "gpt-5.1"
    mode = "gen"

    @classmethod
    def apply(cls, config: ExperimentConfig) -> None:
        config.validate()
        cls.use_cache = config.use_cache
        cls.save_cache = config.save_cache
        cls.cache_dir = str(config.cache_directory)
        cls.perfect_cluster = config.perfect_cluster
        cls.perfect_split = config.perfect_split
        cls.perfect_operator = config.perfect_operator
        cls.perfect_directory = str(config.perfect_directory)
        cls.verbose = config.verbose
        cls.money_saver = config.money_saver
        cls.language_model_batch = config.language_model_batch
        cls.slm_split = config.slm_split
        cls.num_retry = config.num_retry if config.num_retry is not None else 0
        cls.query_individual_factor = config.query_individual_factor
        cls.data_directory = str(config.data_directory)
        cls.num_examples = config.num_examples
        cls.model = resolve_openai_model(config.model)
        cls.llm_model = resolve_openai_model(config.llm_model)
        cls.mode = config.mode

    @classmethod
    def extraction_model(cls) -> str:
        return cls.model if cls.slm_split else cls.llm_model

    @classmethod
    def set_use_cache(cls, value: bool) -> None:
        cls.use_cache = value

    @classmethod
    def set_save_cache(cls, value: bool) -> None:
        cls.save_cache = value

    @classmethod
    def set_cache_dir(cls, value: str) -> None:
        cls.cache_dir = value

    @classmethod
    def set_perfect_cluster(cls, value: bool) -> None:
        cls.perfect_cluster = value

    @classmethod
    def set_perfect_split(cls, value: bool) -> None:
        cls.perfect_split = value

    @classmethod
    def set_perfect_operator(cls, value: bool) -> None:
        cls.perfect_operator = value
        if value:
            cls.perfect_cluster = True
            cls.perfect_split = True

    @classmethod
    def set_perfect_directory(cls, value: str) -> None:
        cls.perfect_directory = value

    @classmethod
    def set_verbose(cls, value: bool) -> None:
        cls.verbose = value

    @classmethod
    def set_money_saver(cls, value: bool) -> None:
        cls.money_saver = value
