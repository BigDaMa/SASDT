from __future__ import annotations

import random
import string
from pathlib import Path
from typing import Iterable

import pandas as pd

from gdt.data import PairDataset, discover_csv_files


PROTECTED_ADDRESS_SEQUENCES = (
    "str",
    "street",
    "strasse",
    "straat",
    "ave",
    "avenue",
    "bvld",
    "rd",
    "road",
    "weg",
    "wy",
)


def shuffle_datasets(
    input_directory: str | Path,
    output_directory: str | Path,
    *,
    repetitions: int = 3,
    datasets: Iterable[str] | None = None,
    seed: int = 42,
) -> list[Path]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    random_state = random.Random(seed)
    written = []
    for path in discover_csv_files(input_directory, list(datasets) if datasets else None):
        frame = PairDataset.from_csv(path).frame
        for iteration in range(repetitions):
            shuffled = frame.sample(
                frac=1,
                random_state=random_state.randrange(0, 2**32),
            ).reset_index(drop=True)
            destination = output / f"{path.stem}_{iteration}.csv"
            shuffled.to_csv(destination, index=False, header=False)
            written.append(destination)
    return written


def expand_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    copies: int = 30,
    mutate: bool = False,
    seed: int = 42,
) -> Path:
    if copies < 0:
        raise ValueError("copies must be non-negative")
    source = PairDataset.from_csv(input_path).frame
    random_state = random.Random(seed)
    frames = [source]
    for _ in range(copies):
        current = source.copy(deep=True)
        if mutate:
            for row_id in range(len(current)):
                left, right = _mutate_pair(
                    current.iat[row_id, 0],
                    current.iat[row_id, 1],
                    random_state,
                )
                current.iat[row_id, 0] = left
                current.iat[row_id, 1] = right
        frames.append(current)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(destination, index=False, header=False)
    return destination


def split_dataset(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    chunk_size: int = 1000,
) -> list[Path]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    source_path = Path(input_path)
    frame = PairDataset.from_csv(source_path).frame
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for start in range(0, len(frame), chunk_size):
        destination = output / f"{source_path.stem}_{start // chunk_size + 1}_{chunk_size}.csv"
        frame.iloc[start : start + chunk_size].to_csv(destination, index=False, header=False)
        written.append(destination)
    return written


def _mutate_pair(left: str, right: str, random_state: random.Random) -> tuple[str, str]:
    protected = _protected_indices(left)
    digits = sorted({character for character in left if character.isdigit()})
    letters = sorted(
        {
            left[index].lower()
            for index in range(len(left))
            if left[index].isalpha() and index not in protected
        }
    )
    if not digits and not letters:
        return left, right
    use_digits = bool(digits) and (not letters or random_state.choice((True, False)))
    if use_digits:
        chosen = random_state.sample(digits, min(len(digits), random_state.randint(1, 2)))
        replacements = {
            value: random_state.choice(
                [candidate for candidate in string.digits if candidate != value]
            )
            for value in chosen
        }
        translate = lambda text: "".join(replacements.get(char, char) for char in text)
    else:
        chosen = random_state.sample(letters, min(len(letters), random_state.randint(1, 3)))
        replacements = {
            value: random_state.choice(
                [candidate for candidate in string.ascii_lowercase if candidate != value]
            )
            for value in chosen
        }

        def translate(text: str) -> str:
            result = []
            for character in text:
                replacement = replacements.get(character.lower()) if character.isalpha() else None
                result.append(
                    replacement.upper()
                    if replacement and character.isupper()
                    else replacement or character
                )
            return "".join(result)

    return translate(left), translate(right)


def _protected_indices(text: str) -> set[int]:
    lower = text.lower()
    protected: set[int] = set()
    for sequence in PROTECTED_ADDRESS_SEQUENCES:
        start = 0
        while (index := lower.find(sequence, start)) >= 0:
            protected.update(range(index, index + len(sequence)))
            start = index + 1
    return protected
