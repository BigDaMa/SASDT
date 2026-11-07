from __future__ import annotations

import re
from typing import Literal, Optional, Tuple

from Propagate.position import AbsPosPosition, Position


def get_str_from_pos(value: str, positions: Tuple[Position, Position]) -> str:
    left = positions[0].get_index(value)
    right = positions[1].get_index(value)
    if left is None:
        return ""
    if right is None:
        if isinstance(positions[1], AbsPosPosition):
            right = len(value)
        else:
            return ""
    start = AbsPos(value, left)
    end = AbsPos(value, right)
    if start > end:
        return ""
    return value[start:end]


def AbsPos(value: str, position: int) -> int:
    resolved = len(value) + position if position < 0 else position
    return min(max(resolved, 0), len(value))


def RegExPos(
    value: str,
    regex: str,
    occurance: int,
    direction: Literal["L", "R"],
) -> Optional[int]:
    matches = list(re.finditer(regex, value))
    index = occurance if occurance >= 0 else len(matches) + occurance
    if not 0 <= index < len(matches):
        return None
    match = matches[index]
    return match.start() if direction == "L" else match.end()
