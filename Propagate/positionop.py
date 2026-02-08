from typing import Literal, Optional, Tuple
import re

from Propagate.position import Position

def get_str_from_pos(v: str, pos: Tuple[Position, Position]) -> str:
    left = pos[0].get_index(v)
    right = pos[1].get_index(v)
    if right is None or right == -1:
        return v[left:]
    elif right < -1:
        return v[left:right+1]
    else:
        return v[left:right]

def AbsPos(v: str, pos: int) -> int:
    n = len(v)
    if pos < 0:
        pos = n + pos  # python-style negative index -> absolute
    if pos < 0:
        return 0
    if pos > n:
        return n
    return pos

def RegExPos(v: str, regex: str, occurance: int, direction: Literal["L", "R"]) -> Optional[int]:
    if direction == "L":
        match = re.finditer(regex, v)
        if occurance < 0: 
            match = list(match)[::-1]
            occurance = -occurance - 1

        for i, m in enumerate(match):
            if i == occurance:
                return m.start()
    else:
        match = re.finditer(regex, v)
        if occurance < 0: 
            match = list(match)[::-1]
            occurance = -occurance - 1
        for i, m in enumerate(match):
            if i == occurance:
                return m.end()
    return None