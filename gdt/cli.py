"""Command-line interface for all supported experiment variants."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Sequence

from gdt.config import Config, ExperimentConfig


EXPERIMENTS = ("benchmark", "pipeline", "cluster", "direct-llm")


def build_parser(*, default_experiment: str = "benchmark") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GDT string-transformation experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--experiment",
        choices=EXPERIMENTS,
        default=default_experiment,
        help="experiment entry point",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_shuffled"),
        help="directory containing two-column CSV datasets",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="CSV filename to run; repeat to select multiple files",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("-c", "--use-cache", action="store_true")
    parser.add_argument("-s", "--save-cache", action="store_true")
    parser.add_argument("-d", "--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("-pg", "--perfect-cluster", action="store_true")
    parser.add_argument("-ps", "--perfect-split", action="store_true")
    parser.add_argument("-pp", "--perfect-operator", action="store_true",
                        help="use ground-truth operators; also enables clusters and splits")
    parser.add_argument("-pd", "--perfect-directory", type=Path, default=Path("data_perfect"))
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-ms", "--money-saver", action="store_true")
    parser.add_argument("-sp", "--slm-split", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "-lb", "--language-model-batch",
        action="store_true",
        help="batch SLM composition and extraction queries; LLM queries are always batched",
    )
    retry_group = parser.add_mutually_exclusive_group()
    retry_group.add_argument("--num-retry", type=int, help="use legacy whole-batch retries")
    retry_group.add_argument(
        "--query-individual-factor", "--query_individual_factor", "--qif", "-qif",
        type=float,
        help="per-phase string-query budget factor (at least 1; default: 1.5)",
    )
    parser.add_argument("-n", "--num-examples", type=int, default=5,
                        help="active examples from the first five reserved pairs (maximum: 5)")
    parser.add_argument(
        "--example-counts",
        type=_comma_separated_ints,
        help="comma-separated example counts from 0 to 5; testing always starts at row 6",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--timeout", type=int, default=600, dest="timeout_seconds",
        help="per-dataset wall-clock limit in seconds, including LM waits (Linux; default: 600)",
    )
    parser.add_argument(
        "-llm",
        default="gpt-5.1",
        help="OpenAI model for exact extraction, example compositions, and remaining labels "
             "when no SLM is used; prediction model for direct-llm",
    )
    parser.add_argument(
        "-slm",
        action="append",
        dest="models",
        help="model for remaining-input composition prediction "
             "(default: gpt-5.1); repeat to benchmark models",
    )
    parser.add_argument(
        "--mode",
        action="append",
        dest="modes",
        default=argparse.SUPPRESS,
        choices=("gen", "generate", "chat"),
        help="local model API mode (default: gen); repeat to benchmark multiple modes",
    )
    parser.add_argument(
        "--cluster-config",
        choices=(
            "collapse+emb",
            "emb+collapse",
            "emb",
            "collapse",
            "collapse+embsplit",
            "collapse+prompt_label",
            "collapse+prompt_label_gpt",
        ),
        default="collapse+prompt_label_gpt",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--provider",
        choices=("openai", "novita"),
        default="openai",
        help="provider for the direct-LLM baseline",
    )
    return parser


def parse_cli_arguments(
    argv: Sequence[str] | None = None,
    *,
    default_experiment: str = "benchmark",
    command: str | None = None,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    """Restrict the direct baseline before parsing any pipeline-only settings."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--experiment", default=default_experiment)
    selected, _ = selector.parse_known_args(arguments)
    parser = parser or build_parser(default_experiment=default_experiment)
    if selected.experiment != "direct-llm":
        return parser.parse_args(arguments)

    command = command or ("python benchmarkgpt.py" if default_experiment == "direct-llm"
                          else "python main.py --experiment direct-llm")
    full_command = f"{command} --provider Provider_Name -llm LLM_Name -n Number_of_Examples"
    direct = argparse.ArgumentParser(description="Direct LLM experiment", allow_abbrev=False)
    direct.add_argument("--experiment", choices=("direct-llm",), default="direct-llm",
                        help=argparse.SUPPRESS)
    direct.add_argument("--provider", choices=("openai", "novita"), default="openai")
    direct.add_argument("-llm", metavar="LLM_Name")
    direct.add_argument("-n", dest="example_counts", type=_comma_separated_ints, default=[5],
                        metavar="Number_of_Examples",
                        help="one example count or comma-separated counts from 0 to 5")
    # argparse accepts abbreviated multi-letter short options even with
    # allow_abbrev=False. Check the exact option surface before it parses values.
    index = 0
    while index < len(arguments):
        token = arguments[index]
        option, equals, _ = token.partition("=")
        if option in {"--experiment", "--provider", "-llm", "-n"}:
            index += 1
            if not equals and index < len(arguments):
                value = arguments[index]
                if not value.startswith("-") or (option == "-n" and re.fullmatch(r"-\d+", value)):
                    index += 1
        elif token in {"-h", "--help"} or re.fullmatch(r"-n-?\d+(?:,-?\d+)*", token):
            index += 1
        else:
            direct.exit(2, "Some of the provided arguments are not supported for the direct LLM "
                        f"experiment.\nThe valid, full command is: {full_command}\n")
    direct_args = direct.parse_args(arguments)
    direct_args.num_examples = direct_args.example_counts[0]
    # Populate the common namespace with defaults only; unsupported options can
    # never mutate global configuration or reach the experiment.
    args = parser.parse_args([])
    vars(args).update(vars(direct_args))
    args.llm = args.llm or ("gpt-5.1" if args.provider == "openai" else "deepseek")
    return args


def parse_arguments(
    argv: Sequence[str] | None = None,
    *,
    default_experiment: str = "benchmark",
) -> argparse.Namespace:
    args = parse_cli_arguments(argv, default_experiment=default_experiment)
    args.modes = getattr(args, "modes", None) or ["gen"]
    try:
        args.config = config_from_args(args)
    except ValueError as error:
        build_parser(default_experiment=default_experiment).error(str(error))
    return args


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    models = ([args.llm] if args.experiment == "direct-llm" else
              args.models or ["gpt-5.1"])
    modes = getattr(args, "modes", None) or ["gen"]
    config = ExperimentConfig(
        data_directory=args.data_dir,
        perfect_directory=args.perfect_directory,
        cache_directory=args.cache_dir,
        output_directory=args.output_dir,
        use_cache=args.use_cache,
        save_cache=args.save_cache,
        perfect_cluster=args.perfect_cluster,
        perfect_split=args.perfect_split,
        perfect_operator=args.perfect_operator,
        verbose=args.verbose,
        money_saver=args.money_saver,
        language_model_batch=args.language_model_batch,
        slm_split=args.slm_split,
        num_retry=args.num_retry,
        query_individual_factor=args.query_individual_factor,
        num_examples=args.num_examples,
        timeout_seconds=args.timeout_seconds,
        model=models[0],
        llm_model=args.llm,
        mode=modes[0],
        cluster_config=args.cluster_config,
    )
    config.validate()
    Config.apply(config)
    return config


def main(argv: Sequence[str] | None = None, *, default_experiment: str = "benchmark") -> int:
    args = parse_arguments(argv, default_experiment=default_experiment)
    config: ExperimentConfig = args.config
    if args.experiment != "direct-llm":
        _print_config(config, args.experiment)

    if args.experiment == "direct-llm":
        from gdt.direct_benchmark import run_direct_benchmark
        from gdt.direct_models import DirectModelError

        try:
            results = run_direct_benchmark(config, provider=args.provider,
                                           example_counts=args.example_counts)
        except DirectModelError as error:
            print(str(error), file=sys.stderr)
            return 1
    elif args.experiment == "cluster":
        from gdt.cluster_experiment import run_cluster_experiment
        from gdt.experiments import write_summary

        results = run_cluster_experiment(config, datasets=args.datasets)
        write_summary(results, config.output_directory / "cluster")
    elif args.experiment == "pipeline":
        from gdt.experiments import run_pipeline_experiment, write_summary

        results = run_pipeline_experiment(config, args.datasets)
        write_summary(results, config.output_directory)
    else:
        from gdt.experiments import run_benchmark

        results = run_benchmark(
            config,
            datasets=args.datasets,
            models=args.models,
            modes=args.modes,
            example_counts=args.example_counts,
            repetitions=args.repetitions,
            fail_fast=args.fail_fast,
        )

    return 0 if all(result.succeeded for result in results) else 1


def _comma_separated_ints(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(not 0 <= item <= 5 for item in values):
        raise argparse.ArgumentTypeError("example counts must be between 0 and 5")
    return values


def _print_config(config: ExperimentConfig, experiment: str) -> None:
    print(f"Experiment: {experiment}")
    for key, value in config.as_display_dict().items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
