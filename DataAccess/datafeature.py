from __future__ import annotations

from gdt.timing import output_time

import re
from typing import Any, List

from Utils.config import Config
from gdt.diagnostics import record_ground_truth_error
from gdt.ground_truth import perfect_path


class DataFeature:
    data_features: dict[int, dict[str, Any]] = {}

    @classmethod
    def clear(cls) -> None:
        cls.data_features.clear()

    @classmethod
    def _values(cls, field: str) -> List[Any]:
        return [feature[field] for feature in cls.data_features.values()]

    @classmethod
    def _mapping(cls, field: str) -> dict[int, Any]:
        return {
            value_id: feature[field]
            for value_id, feature in cls.data_features.items()
        }

    @classmethod
    def get_all_strings_list(cls) -> List[str]:
        return cls._values("text")

    @classmethod
    def get_all_strings_dict(cls) -> dict[int, str]:
        return cls._mapping("text")

    @classmethod
    def get_all_collapses_list(cls) -> List[str]:
        return cls._values("collapsed")

    @classmethod
    def get_all_collapses_dict(cls) -> dict[int, str]:
        return cls._mapping("collapsed")

    @classmethod
    def get_all_compositions_list(cls) -> List[str]:
        return cls._values("composition")

    @classmethod
    def get_all_compositions_dict(cls) -> dict[int, str]:
        return cls._mapping("composition")

    @staticmethod
    def get_labels_from_string(value: str) -> List[str]:
        return re.findall(r"\[([^\]]+)\]", value)

    @classmethod
    def get_all_sem_types(cls, file: str) -> List[str]:
        if Config.perfect_cluster:
            type_path = perfect_path(file, "_types.txt")
            if type_path.is_file():
                return sorted(
                    {
                        line.strip()
                        for line in type_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    }
                )
            if not Config.perfect_split:
                record_ground_truth_error(type_path, "missing semantic types file")
            with output_time():
                if Config.verbose:
                    print(f"Type file {type_path} not found; deriving types from features.")
        return sorted(
            {
                label
                for feature in cls.data_features.values()
                for label in cls.get_labels_from_string(feature.get("composition", ""))
            }
        )
