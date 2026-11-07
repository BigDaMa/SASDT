from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


class Position(ABC):
    @abstractmethod
    def get_index(self, value: str) -> int | None:
        """Resolve this symbolic position against ``value``."""


@dataclass(frozen=True, slots=True)
class AbsPosPosition(Position):
    idx: int | None

    def get_index(self, value: str) -> int | None:
        del value
        return self.idx

    def __str__(self) -> str:
        return f"AbsPosPosition({self.idx})"


@dataclass(frozen=True, slots=True)
class RegExPosPosition(Position):
    regex: str
    occurance: int  # Historical attribute spelling retained.
    direction: Literal["L", "R"]

    def __post_init__(self) -> None:
        if self.direction not in {"L", "R"}:
            raise ValueError("direction must be 'L' or 'R'")

    @property
    def occurrence(self) -> int:
        return self.occurance

    def get_regex(self) -> str:
        return self.regex

    def get_index(self, value: str) -> int | None:
        from Propagate.positionop import RegExPos

        return RegExPos(value, self.regex, self.occurance, self.direction)

    def get_as_tuple(self) -> tuple[str, int, str]:
        return self.regex, self.occurance, self.direction

    def __str__(self) -> str:
        return f"RegExPosPosition({self.regex}, {self.occurance}, {self.direction})"
