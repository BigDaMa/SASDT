from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import pandas as pd

from gdt.constants import RESERVED_EXAMPLE_COUNT


@dataclass(frozen=True, slots=True)
class PairRecord:
    row_id: int
    source: str
    target: str


class PairDataset:
    """A validated two-column source/target dataset.

    IDs remain compatible with the original experiments: source row ``i`` has ID
    ``i`` and its target has ID ``row_count + i``.  The conversion is centralized
    here, which removes the repeated offset arithmetic from the pipeline.
    """

    def __init__(self, frame: pd.DataFrame, *, path: Path | None = None) -> None:
        if frame.shape[1] != 2:
            location = f" in {path}" if path else ""
            raise ValueError(
                f"expected exactly two CSV columns{location}; found {frame.shape[1]}"
            )
        if frame.empty:
            location = f" {path}" if path else ""
            raise ValueError(f"dataset{location} is empty")
        if frame.isna().any(axis=None):
            location = f" in {path}" if path else ""
            raise ValueError(f"missing source or target value{location}")

        clean = frame.reset_index(drop=True).copy()
        clean.columns = ["source", "target"]
        clean["source"] = clean["source"].astype(str)
        clean["target"] = clean["target"].astype(str)
        self._frame = clean
        self.path = path

    @classmethod
    def from_csv(cls, path: str | Path) -> "PairDataset":
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"dataset not found: {resolved}")
        return cls(
            pd.read_csv(resolved, header=None, dtype=str, keep_default_na=False),
            path=resolved,
        )

    @property
    def row_count(self) -> int:
        return len(self._frame)

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy()

    def validate_example_count(self, count: int) -> None:
        if count < 0:
            raise ValueError("example count must be non-negative")
        if count > RESERVED_EXAMPLE_COUNT:
            raise ValueError(f"example count must not exceed {RESERVED_EXAMPLE_COUNT}")
        if count > self.row_count:
            raise ValueError(
                f"requested {count} examples from a dataset with {self.row_count} rows"
            )

    def source_id(self, row_id: int) -> int:
        self._validate_row_id(row_id)
        return row_id

    def target_id(self, row_id: int) -> int:
        self._validate_row_id(row_id)
        return self.row_count + row_id

    def row_id_from_target(self, target_id: int) -> int:
        row_id = target_id - self.row_count
        self._validate_row_id(row_id)
        return row_id

    def paired_id(self, value_id: int) -> int:
        if 0 <= value_id < self.row_count:
            return self.target_id(value_id)
        return self.source_id(self.row_id_from_target(value_id))

    def text_for_id(self, value_id: int) -> str:
        if 0 <= value_id < self.row_count:
            return self._frame.iloc[value_id]["source"]
        row_id = self.row_id_from_target(value_id)
        return self._frame.iloc[row_id]["target"]

    def records(self) -> Iterator[PairRecord]:
        for row_id, row in self._frame.iterrows():
            yield PairRecord(row_id, row["source"], row["target"])

    def examples(self, count: int) -> list[dict[str, str]]:
        self.validate_example_count(count)
        return [
            {"input": record.source, "output": record.target}
            for record in list(self.records())[:count]
        ]

    def example_inputs(self, count: int) -> dict[int, str]:
        self.validate_example_count(count)
        return {record.row_id: record.source for record in list(self.records())[:count]}

    def example_targets(self, count: int) -> dict[int, str]:
        self.validate_example_count(count)
        return {
            self.target_id(record.row_id): record.target
            for record in list(self.records())[:count]
        }

    def test_sources(self, count: int) -> dict[int, str]:
        self.validate_example_count(count)
        return {
            record.row_id: record.source
            for record in list(self.records())[RESERVED_EXAMPLE_COUNT:]
        }

    def ground_truth(self, count: int) -> dict[int, str]:
        self.validate_example_count(count)
        return {
            record.row_id: record.target
            for record in list(self.records())[RESERVED_EXAMPLE_COUNT:]
        }

    def active_sources(self, example_count: int) -> dict[int, str]:
        """Keep active examples and the fixed test set, retaining original IDs."""
        return self.example_inputs(example_count) | self.test_sources(example_count)

    def clustering_values(self, example_count: int) -> list[str]:
        """Return active values in order; use clustering_mapping for their IDs."""
        return list(self.clustering_mapping(example_count).values())

    def clustering_mapping(self, example_count: int) -> Mapping[int, str]:
        return self.active_sources(example_count) | self.example_targets(example_count)

    def _validate_row_id(self, row_id: int) -> None:
        if not 0 <= row_id < self.row_count:
            raise IndexError(f"row ID {row_id} outside [0, {self.row_count})")


def discover_csv_files(directory: str | Path, names: Sequence[str] | None = None) -> list[Path]:

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"data directory not found: {root}")
    requested = set(names or ())
    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    )
    if requested:
        files = [path for path in files if path.name in requested]
        missing = requested - {path.name for path in files}
        if missing:
            raise FileNotFoundError(f"datasets not found in {root}: {', '.join(sorted(missing))}")
    return files
