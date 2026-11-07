from __future__ import annotations

from pathlib import Path
from typing import Sequence

from gdt.cli import main as cli_main
from gdt.config import ExperimentConfig
from gdt.dataset_tools import expand_dataset, split_dataset
from gdt.direct_benchmark import run_direct_benchmark


def benchmark_gpt(file: str, temperature: float = 0.1, numexamples: int = 5) -> float:
    del temperature  # The Responses adapter chooses model-supported parameters.
    path = Path(file)
    config = ExperimentConfig(
        data_directory=path.parent,
        num_examples=numexamples,
        model="gpt-5.1",
    )
    result = run_direct_benchmark(
        config,
        provider="openai",
        datasets=[path.name],
    )[0]
    if not result.succeeded:
        raise RuntimeError(result.error)
    assert result.evaluation is not None
    return result.evaluation.accuracy


def benchmark_novita(
    file: str,
    modelname: str,
    temperature: float = 0.1,
    numexamples: int = 5,
) -> float:
    del temperature
    path = Path(file)
    config = ExperimentConfig(
        data_directory=path.parent,
        num_examples=numexamples,
        model=modelname,
        llm_model=modelname,
    )
    result = run_direct_benchmark(
        config,
        provider="novita",
        datasets=[path.name],
    )[0]
    if not result.succeeded:
        raise RuntimeError(result.error)
    assert result.evaluation is not None
    return result.evaluation.accuracy


def generate_large_dataset(mutate: bool = False):
    return expand_dataset(
        "data_shuffled/Address_0.csv",
        "data_large/Address_0_large.csv",
        mutate=mutate,
    )


def split_large_dataset(maxnum: int = 1000):
    return split_dataset(
        "data_large/Address_0_large.csv",
        "data_large",
        chunk_size=maxnum,
    )


def benchmark_large(file: str):
    import time

    started = time.perf_counter()
    accuracy = benchmark_gpt(file, numexamples=5)
    return accuracy, time.perf_counter() - started


def benchmark(argv: Sequence[str] | None = None) -> int:
    return cli_main(argv, default_experiment="direct-llm")


if __name__ == "__main__":
    raise SystemExit(benchmark())
