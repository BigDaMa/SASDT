from __future__ import annotations

import argparse
from pathlib import Path

from gdt.dataset_tools import expand_dataset, shuffle_datasets, split_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare GDT experiment datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    shuffle = subparsers.add_parser("shuffle")
    shuffle.add_argument("input_directory", type=Path)
    shuffle.add_argument("output_directory", type=Path)
    shuffle.add_argument("--repetitions", type=int, default=3)
    shuffle.add_argument("--seed", type=int, default=42)
    shuffle.add_argument("--dataset", action="append", dest="datasets")

    expand = subparsers.add_parser("expand")
    expand.add_argument("input_path", type=Path)
    expand.add_argument("output_path", type=Path)
    expand.add_argument("--copies", type=int, default=30)
    expand.add_argument("--mutate", action="store_true")
    expand.add_argument("--seed", type=int, default=42)

    split = subparsers.add_parser("split")
    split.add_argument("input_path", type=Path)
    split.add_argument("output_directory", type=Path)
    split.add_argument("--chunk-size", type=int, default=1000)

    args = parser.parse_args()
    if args.command == "shuffle":
        written = shuffle_datasets(
            args.input_directory,
            args.output_directory,
            repetitions=args.repetitions,
            datasets=args.datasets,
            seed=args.seed,
        )
    elif args.command == "expand":
        written = [
            expand_dataset(
                args.input_path,
                args.output_path,
                copies=args.copies,
                mutate=args.mutate,
                seed=args.seed,
            )
        ]
    else:
        written = split_dataset(
            args.input_path,
            args.output_directory,
            chunk_size=args.chunk_size,
        )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
