from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import replace
from pathlib import Path
import re
from statistics import mean
from typing import Iterable, Sequence

import pandas as pd

from gdt.diagnostics import clear_errors
from gdt.config import ExperimentConfig
from gdt.data import discover_csv_files
from gdt.output import dataset_output, print_run_result
from gdt.pipeline import PipelineRunner
from gdt.progress import ProgressLog
from gdt.results import RunResult, append_jsonl
from gdt.models import resolve_openai_model
from gdt.pricing import write_pricing


def _model_label(model: str) -> str:
    """Keep model versions and variants, but omit parameter-count tokens."""
    model = resolve_openai_model(model.strip())
    if model.casefold() == "gpt":
        model = "gpt-5.1"
    model = re.sub(r"[-_: ]\d+(?:\.\d+)?[bm](?=$|[-_: ])", "", model, flags=re.I)
    for family in ("GPT", "Qwen", "LLaMA", "Phi", "Granite", "Ministral"):
        if model.casefold().startswith(family.casefold()):
            return family + model[len(family):]
    return model


def model_column(model: str, llm_model: str = "gpt-5.1") -> str:
    llm, slm = _model_label(llm_model), _model_label(model)
    return llm if llm.casefold() == slm.casefold() else f"{llm}+{slm}"


def run_pipeline_experiment(
    config: ExperimentConfig,
    datasets: Sequence[str] | None = None,
) -> list[RunResult]:
    clear_errors(config.output_directory)
    paths = discover_csv_files(config.data_directory, datasets)
    progress = ProgressLog(config.output_directory, "pipeline", len(paths))
    runner = PipelineRunner(config)
    results = []
    for path in paths:
        with dataset_output(path.name), progress.run(path.name, config) as current:
            result = runner.run(path)
            current.finish(result)
            print_run_result(result)
        results.append(result)
    return results


def run_benchmark(
    config: ExperimentConfig,
    *,
    datasets: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    modes: Sequence[str] | None = None,
    example_counts: Iterable[int] | None = None,
    repetitions: int = 1,
    fail_fast: bool = False,
) -> list[RunResult]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    clear_errors(config.output_directory)
    paths = discover_csv_files(config.data_directory, datasets)
    selected_models = [resolve_openai_model(model) for model in (models or [config.model])]
    selected_modes = list(modes or [config.mode])
    selected_counts = list(example_counts or [config.num_examples])
    if config.money_saver:
        from Utils.BenchmarkCache import BenchmarkCache

        BenchmarkCache.clear()
        # The largest real run primes both GPT phases without an unmeasured
        # warm-up run or an additional paid request.
        selected_counts = sorted(set(selected_counts), reverse=True)
        selected_models.sort(
            key=lambda model: not (model == "gpt" or model.lower().startswith("gpt-5.1"))
        )
    results: list[RunResult] = []
    jsonl_path = config.output_directory / "runs.jsonl"
    total = len(paths) * len(selected_models) * len(selected_modes) * len(selected_counts) * repetitions
    progress = ProgressLog(config.output_directory, "benchmark", total)

    for repetition in range(1, repetitions + 1):
        for count in selected_counts:
            for model in selected_models:
                for mode in selected_modes:
                    run_config = replace(
                        config,
                        num_examples=count,
                        model=model,
                        mode=mode,
                    )
                    runner = PipelineRunner(run_config)
                    for path in paths:
                        with dataset_output(path.name), progress.run(
                            path.name, run_config, repetition
                        ) as current:
                            result = runner.run(path)
                            current.finish(result)
                            result.repetition = repetition
                            results.append(result)
                            append_jsonl(jsonl_path, result)
                            print_run_result(result)
                        if fail_fast and not result.succeeded:
                            write_summary(results, config.output_directory)
                            write_benchmark_tables(
                                results, config.output_directory, selected_counts,
                            )
                            return results

    write_summary(results, config.output_directory)
    write_benchmark_tables(
        results, config.output_directory, selected_counts,
    )
    return results


def write_summary(results: Sequence[RunResult], output_directory: str | Path) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    records = [result.to_record() for result in results]
    pd.DataFrame(records).to_csv(output / "summary.csv", index=False)
    write_pricing(results, output)


def write_benchmark_tables(
    results: Sequence[RunResult], output_directory: str | Path,
    example_counts: Iterable[int] | None = None, datasets: Sequence[str] | None = None,
) -> None:
    """Export measured accuracy and elapsed seconds, with equal-weight dataset means.

    Columns identify LLM/SLM pairs across the entire run, without parameter counts.
    Rows include only datasets with recorded results for that example count.
    The optional datasets argument suggests row order; it cannot add or omit results.
    Repetitions/modes are averaged within a dataset first, including failed runs.
    Accuracy and runtime use the same values as summary.csv; absent evaluations
    have zero accuracy. Only combinations without any recorded result stay blank.
    """
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    counts = sorted(set(example_counts if example_counts is not None else
                        (result.num_examples for result in results)))
    columns = list(dict.fromkeys(model_column(result.model, result.llm_model)
                                 for result in results))
    for count in counts:
        count_results = [result for result in results if result.num_examples == count]
        recorded_names = dict.fromkeys(result.dataset for result in count_results)
        names = list(dict.fromkeys(
            [name for name in (datasets or ()) if name in recorded_names] + list(recorded_names)
        ))
        metrics = {"Accuracy": defaultdict(list), "Runtime": defaultdict(list)}
        for result in count_results:
            column = model_column(result.model, result.llm_model)
            key = (result.dataset, column)
            metrics["Runtime"][key].append(result.duration_seconds)
            accuracy = result.evaluation.accuracy if result.evaluation is not None else 0.0
            metrics["Accuracy"][key].append(accuracy)
        for metric, values in metrics.items():
            cells = {key: mean(samples) for key, samples in values.items()}
            rows = [[name, *(cells.get((name, column), "") for column in columns)]
                    for name in names]
            means = []
            for column in columns:
                available = [cells[(name, column)] for name in names if (name, column) in cells]
                means.append(mean(available) if available else "")
            with (output / f"{metric}-{count}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Dataset", *columns])
                writer.writerows(rows)
                writer.writerow(["mean", *means])
