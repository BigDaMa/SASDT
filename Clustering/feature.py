from __future__ import annotations

from gdt.timing import output_time

from pathlib import Path
import re
import statistics
from typing import Dict, List, Sequence, Union

import pandas as pd


AVAIL_FEAT = [
    "LengthFeature",
    "NumTokensFeature",
    "RatioAlphaFeature",
    "RatioNumericFeature",
    "RatioAlphanumericFeature",
    "RatioNonAlnumFeature",
    "NonAlnumOffsetFeature",
    "TypeNonAlnumCharFeature",
]
FEATURE_DIRECTORY = Path("data_feature")


class Feature:
    AVAILABLE_FEATURES = tuple(AVAIL_FEAT)

    @staticmethod
    def normalize(values: Sequence[float]) -> List[float]:
        total = sum(values)
        return [float(value) / total for value in values] if total else [0.0] * len(values)

    __normalize_list__ = normalize

    @staticmethod
    def min_max_normalize(values: Sequence[Union[int, float]]) -> List[float]:
        numbers = [float(value) for value in values]
        if len(numbers) < 2 or min(numbers, default=0.0) == max(numbers, default=0.0):
            return [0.0] * len(numbers)
        minimum, maximum = min(numbers), max(numbers)
        return [(value - minimum) / (maximum - minimum) for value in numbers]

    __min_max_normalizer__ = min_max_normalize

    @staticmethod
    def min_max_abs_normalize(values: Sequence[Union[int, float]]) -> List[float]:
        numbers = [float(value) for value in values]
        magnitudes = [abs(value) for value in numbers]
        if len(numbers) < 2 or min(magnitudes, default=0.0) == max(magnitudes, default=0.0):
            return [0.0] * len(numbers)
        minimum, maximum = min(magnitudes), max(magnitudes)
        return [
            (abs(value) - minimum) * (1 if value >= 0 else -1) / (maximum - minimum)
            for value in numbers
        ]

    __min_max_abs_normalizer__ = min_max_abs_normalize

    @staticmethod
    def validate_feature(feature: str) -> bool:
        return feature in Feature.AVAILABLE_FEATURES

    @staticmethod
    def LengthFeature(value: str) -> float:
        return float(len(str(value)))

    @staticmethod
    def NumTokensFeature(value: str) -> float:
        return float(len(re.findall(r"[A-Za-z0-9]+|[^\w\s]", str(value))))

    @staticmethod
    def _ratio(value: str, predicate) -> float:
        text = str(value)
        return sum(predicate(character) for character in text) / len(text) if text else 0.0

    @staticmethod
    def RatioAlphaFeature(value: str) -> float:
        return Feature._ratio(value, str.isalpha)

    @staticmethod
    def RatioNumericFeature(value: str) -> float:
        return Feature._ratio(value, str.isnumeric)

    @staticmethod
    def RatioAlphanumericFeature(value: str) -> float:
        return Feature._ratio(value, str.isalnum)

    @staticmethod
    def RatioNonAlnumFeature(value: str) -> float:
        return Feature._ratio(value, lambda character: not character.isalnum())

    @staticmethod
    def NonAlnumOffsetFeature(value: str) -> float:
        text = str(value)
        offsets = []
        for index, character in enumerate(text):
            if character.isalnum():
                continue
            right_run = 1
            while index + right_run < len(text) and not text[index + right_run].isalnum():
                right_run += 1
            left_run = 1
            while index - left_run >= 0 and not text[index - left_run].isalnum():
                left_run += 1
            offsets.append(float(right_run - left_run))
        normalized = Feature.min_max_abs_normalize(offsets)
        return statistics.stdev(normalized) if len(normalized) > 1 else 0.0

    @staticmethod
    def TypeNonAlnumCharFeature(value: str) -> str:
        characters = {
            "d" if character == '"' else character
            for character in str(value)
            if not character.isalnum() and not character.isspace()
        }
        return " ".join(sorted(characters))

    @staticmethod
    def get_features(value: str, feature_names: Sequence[str]) -> list[float | str]:
        unknown = set(feature_names) - set(Feature.AVAILABLE_FEATURES)
        if unknown:
            raise ValueError(f"unknown features: {sorted(unknown)}")
        return [getattr(Feature, feature)(value) for feature in feature_names]

    getFeature = get_features

    @staticmethod
    def save_features_to_file(features: List[list], file: str) -> None:
        with output_time():
            FEATURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(features).to_csv(
                FEATURE_DIRECTORY / file,
                header=["OrigStr", *Feature.AVAILABLE_FEATURES],
                index=False,
            )

    @staticmethod
    def get_feature_for_strs(
        data: List[str],
        file: str,
        reset: bool = False,
    ) -> Dict[str, Dict[str, float | str]]:
        path = FEATURE_DIRECTORY / file
        if not reset and path.is_file():
            frame = pd.read_csv(path)
            return {
                str(row["OrigStr"]): {
                    feature: row[feature] for feature in Feature.AVAILABLE_FEATURES
                }
                for _, row in frame.iterrows()
            }
        result = {}
        rows = []
        for value in data:
            text = str(value)
            features = dict(
                zip(
                    Feature.AVAILABLE_FEATURES,
                    Feature.get_features(text, Feature.AVAILABLE_FEATURES),
                )
            )
            result[text] = features
            rows.append([text, *features.values()])
        Feature.save_features_to_file(rows, file)
        return result
