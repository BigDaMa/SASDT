from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from gdt.results import RunResult


DATASET_SEPARATOR = "#" * 96


@contextmanager
def dataset_output(dataset: str) -> Iterator[None]:
    """Make one dataset's complete output unmistakable in long console logs."""

    print()
    print(DATASET_SEPARATOR)
    print(f"### DATASET START: {dataset} ###")
    print(DATASET_SEPARATOR)
    try:
        yield
    finally:
        print(DATASET_SEPARATOR)
        print(f"### DATASET END: {dataset} ###")
        print(DATASET_SEPARATOR)
        print()


def print_run_result(result: RunResult) -> None:
    prefix = (
        f"{result.dataset} | model={result.model} | mode={result.mode} | "
        f"examples={result.num_examples} | runtime: {result.duration_seconds:.2f}s"
    )
    if result.has_lm_failures:
        prefix += (f" | LM FAILURES: {result.lm_failed_responses} rejected responses, "
                   f"{result.lm_empty_results} empty results, "
                   f"{result.lm_tolerated_splits} partial splits tolerated")
    if result.succeeded:
        assert result.evaluation is not None
        print(
            f"{prefix} | correct: {result.evaluation.correct} | "
            f"error: {result.evaluation.incorrect} | "
            f"accuracy: {result.evaluation.accuracy:.4f}"
        )
    else:
        print(f"{prefix} | ERROR: {result.error}")
    print()
