from gdt.timing import output_time

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import json
from typing import Any


class IterPrintUtil:
    @staticmethod
    def print(value: Any, depth: int = 0) -> None:
        with output_time():
            print(IterPrintUtil.format(value, depth))

    @staticmethod
    def format(value: Any, depth: int = 0) -> str:
        return json.dumps(
            IterPrintUtil._json_value(value),
            ensure_ascii=False,
            indent=max(2, (depth + 1) * 2),
        )

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "type": type(value).__name__,
                **{
                    str(key): IterPrintUtil._json_value(item)
                    for key, item in asdict(value).items()
                },
            }
        if isinstance(value, Mapping):
            return {
                str(key): IterPrintUtil._json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [IterPrintUtil._json_value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return [
                IterPrintUtil._json_value(item)
                for item in sorted(value, key=lambda item: repr(item))
            ]
        return str(value)
