from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from gdt.experiments import write_benchmark_tables, write_summary
from gdt.results import EvaluationResult, RunResult


def load_results(path: Path) -> list[RunResult]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle)) if path.suffix == ".csv" else [
            json.loads(line) for line in handle if line.strip()
        ]
    results = []
    for row in rows:
        error = row.get("error") or None
        evaluation = row.get("evaluation")
        if evaluation is None and row.get("accuracy") not in (None, ""):
            evaluation = {key: row[key] for key in ("accuracy", "correct", "incorrect")}
        results.append(RunResult(
            dataset=row["dataset"], model=row["model"], mode=row["mode"],
            num_examples=int(row["num_examples"]), duration_seconds=float(row["duration_seconds"]),
            repetition=int(row.get("repetition", 1)), error=error,
            llm_model=row.get("llm_model") or "gpt-5.1",
            has_lm_failures=str(row.get("has_lm_failures", False)).lower() in {"true", "1"},
            lm_failed_responses=int(row.get("lm_failed_responses") or 0),
            lm_empty_results=int(row.get("lm_empty_results") or 0),
            lm_tolerated_splits=int(row.get("lm_tolerated_splits") or 0),
            **{key: int(row.get(key) or 0) for key in (
                "llm_input_tokens", "llm_output_tokens", "slm_input_tokens", "slm_output_tokens",
            )},
            **{key: float(row[key]) if row.get(key) not in (None, "", "N/A") else None
               for key in ("cf_llm", "cf_slm", "cf_total")},
            evaluation=EvaluationResult.from_mapping(evaluation) if evaluation is not None else None,
        ))
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--example-counts", help="comma-separated counts; defaults to all saved counts")
    args = parser.parse_args(argv)
    counts = ([int(value) for value in args.example_counts.split(",")]
              if args.example_counts is not None else None)
    results = load_results(args.source)
    write_benchmark_tables(results, args.output_dir, counts)
    write_summary(results, args.output_dir)


if __name__ == "__main__":
    main()
