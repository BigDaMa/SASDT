from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from gdt.cli import main as _main, parse_arguments
from gdt.output import dataset_output, print_run_result
from gdt.pipeline import PipelineRunner


def run_single(file: str, model: str | None = None, numexamples: int = 5):
    """Run one dataset using the new pipeline and return the legacy result shape."""

    args = parse_arguments([], default_experiment="pipeline")
    config = args.config
    from DataAccess.dataaccess import DataAccess

    config.data_directory = Path(DataAccess.DATA_FOLDER)
    config.num_examples = numexamples
    if model:
        config.model = model
    with dataset_output(file):
        result = PipelineRunner(config).run(config.data_directory / file)
        print_run_result(result)
    if not result.succeeded:
        raise RuntimeError(result.error)
    assert result.evaluation is not None
    return asdict(result.evaluation)


def main(argv: Sequence[str] | None = None) -> int:
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
