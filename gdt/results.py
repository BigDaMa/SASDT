from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EvaluationResult:
    accuracy: float
    correct: int
    incorrect: int
    incorrect_items: list[dict[str, Any]] = field(default_factory=list)
    output_items: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "EvaluationResult":
        return cls(
            accuracy=float(value.get("accuracy", 0.0)),
            correct=int(value.get("correct", 0)),
            incorrect=int(value.get("incorrect", 0)),
            incorrect_items=list(value.get("incorrect_items", [])),
            output_items=list(value.get("output_items", [])),
        )


@dataclass(slots=True)
class RunResult:
    dataset: str
    model: str
    mode: str
    num_examples: int
    duration_seconds: float
    repetition: int = 1
    evaluation: EvaluationResult | None = None
    error: str | None = None
    llm_model: str = "gpt-5.1"
    has_lm_failures: bool = False
    lm_failed_responses: int = 0
    lm_empty_results: int = 0
    lm_tolerated_splits: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    slm_input_tokens: int = 0
    slm_output_tokens: int = 0
    # None means unmeasured or missing pricing, never a free model request.
    cf_llm: float | None = None
    cf_slm: float | None = None
    cf_total: float | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.evaluation is not None

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        evaluation = record.pop("evaluation")
        if evaluation:
            record.update(evaluation)
        else:
            record.update({"accuracy": 0.0, "correct": 0, "incorrect": 0})
        record.pop("incorrect_items", None)
        record.pop("output_items", None)
        return record


def append_jsonl(path: str | Path, result: RunResult) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result), ensure_ascii=False, default=str) + "\n")
