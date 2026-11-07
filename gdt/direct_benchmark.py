from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

from gdt.diagnostics import clear_errors, dataset_diagnostics, split_phase, write_dataset_error
from gdt.progress import ProgressLog, progress_phase
from gdt.timing import PipelineTimer, stop_pipeline_timer
from gdt.timeout import dataset_time_limit
from gdt.config import ExperimentConfig
from gdt.data import PairDataset, discover_csv_files
from gdt.direct_models import UnsupportedModelError, check_models, is_unsupported_model
from gdt.experiments import write_summary
from gdt.output import dataset_output, print_run_result
from gdt.results import EvaluationResult, RunResult, append_jsonl
from gdt.pricing import CostTracker, cost_session, track_request
from LLMUtil.LLMPrompt import LLMPrompt


def run_direct_benchmark(
    config: ExperimentConfig,
    *,
    provider: str = "openai",
    datasets: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    example_counts: Sequence[int] | None = None,
    repetitions: int = 1,
    fail_fast: bool = False,
) -> list[RunResult]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    selected_models = list(models or [config.llm_model])
    check_models(selected_models, provider)
    clear_errors(config.output_directory)
    paths = discover_csv_files(config.data_directory, datasets)
    selected_counts = list(example_counts or [config.num_examples])
    results: list[RunResult] = []
    total = len(paths) * len(selected_models) * len(selected_counts) * repetitions
    progress = ProgressLog(config.output_directory, "direct-llm", total)

    for repetition in range(1, repetitions + 1):
        for count in selected_counts:
            for model in selected_models:
                run_config = replace(config, num_examples=count, model=model, llm_model=model)
                for path in paths:
                    with dataset_output(path.name), progress.run(
                        path.name, run_config, repetition, provider=provider
                    ) as current:
                        try:
                            result = _run_one(run_config, path, provider)
                        except UnsupportedModelError as error:
                            if error.result is not None:
                                error.result.repetition = repetition
                                results.append(error.result)
                                current.finish(error.result)
                                append_jsonl(config.output_directory / "direct_llm_runs.jsonl",
                                             error.result)
                            if results:
                                write_summary(results, config.output_directory / "direct_llm")
                            raise
                        current.finish(result)
                        result.repetition = repetition
                        results.append(result)
                        append_jsonl(
                            config.output_directory / "direct_llm_runs.jsonl", result
                        )
                        print_run_result(result)
                    if fail_fast and not result.succeeded:
                        write_summary(results, config.output_directory / "direct_llm")
                        return results

    write_summary(results, config.output_directory / "direct_llm")
    return results


def _run_one(config: ExperimentConfig, path: Path, provider: str) -> RunResult:
    timer = PipelineTimer()
    costs = CostTracker()
    # The direct baseline has one prediction model, specified by -llm.
    config = replace(config, model=config.llm_model)
    try:
        with (
            dataset_diagnostics(config.output_directory, path.name, config) as diagnostics,
            cost_session(costs),
            timer,
            dataset_time_limit(config.timeout_seconds),
        ):
            dataset = PairDataset.from_csv(path)
            dataset.validate_example_count(config.num_examples)
            with split_phase("direct LLM prediction"):
                predictions = _predict(dataset, config, provider)
            stop_pipeline_timer()
            progress_phase("evaluation")
            ground_truth = dataset.ground_truth(config.num_examples)
            incorrect_items = []
            correct = 0
            for offset, (value_id, expected) in enumerate(ground_truth.items()):
                prediction = predictions[offset].strip() if offset < len(predictions) else ""
                if prediction == expected.strip():
                    correct += 1
                else:
                    incorrect_items.append(
                        {
                            "index": value_id,
                            "prediction": prediction,
                            "ground_truth": expected,
                        }
                    )
            total = len(ground_truth)
            evaluation = EvaluationResult(
                accuracy=correct / total if total else 0.0,
                correct=correct,
                incorrect=total - correct,
                incorrect_items=incorrect_items,
            )
            return RunResult(
                dataset=path.name,
                model=config.model,
                llm_model=config.llm_model,
                mode=provider,
                num_examples=config.num_examples,
                duration_seconds=timer.elapsed,
                evaluation=evaluation,
                **diagnostics.result_fields(),
                **costs.result_fields(),
            )
    except Exception as error:
        result = RunResult(
            dataset=path.name,
            model=config.llm_model,
            llm_model=config.llm_model,
            mode=provider,
            num_examples=config.num_examples,
            duration_seconds=timer.elapsed,
            error=f"{type(error).__name__}: {error}",
            **diagnostics.result_fields(),
            **costs.result_fields(),
        )
        if is_unsupported_model(error):
            unsupported = UnsupportedModelError(config.llm_model, provider, result=result)
            result.error = f"UnsupportedModelError: {unsupported}"
            raise unsupported from error
        write_dataset_error(config.output_directory, path.name, config, error)
        return result


def _predict(
    dataset: PairDataset,
    config: ExperimentConfig,
    provider: str,
) -> list[str]:
    from LLMUtil.LLMUtil import NovitaUtil, OpenAIUtil

    examples = dataset.examples(config.num_examples)
    questions = list(dataset.test_sources(config.num_examples).values())
    if not questions:
        return []
    if provider == "openai":
        prompt = (
            LLMPrompt.direct_transform_examples(examples)
            + LLMPrompt.direct_transform_questions(questions)
        )
        response = OpenAIUtil.create_response(
            prompt,
            instructions=LLMPrompt.GPT_DIRECT_TRANSFORM,
            model=config.llm_model,
            num_retry=1,
        )
        return [_clean_prediction(line) for line in response.splitlines() if line.strip()]

    if provider == "novita":
        client = NovitaUtil.get_client().with_options(max_retries=0)
        model = NovitaUtil.get_novita_model(config.llm_model)
        predictions = []
        for question in questions:
            prompt = LLMPrompt.direct_transform_question(
                examples=examples,
                question=question,
            )
            with track_request(model, prompt) as usage:
                response = client.completions.create(
                    model=model,
                    prompt=prompt,
                    temperature=0.1,
                    stop=[LLMPrompt.DIRECT_TRANSFORM_STOP],
                )
                if usage is not None:
                    usage.record_output(response.choices[0].text)
            predictions.append(_clean_prediction(response.choices[0].text))
        return predictions

    raise ValueError(f"unknown provider: {provider}")


def _clean_prediction(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("A:"):
        cleaned = cleaned[2:].strip()
    return cleaned
