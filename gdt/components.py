from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Mapping


def component_key(label: str) -> tuple[str, int, str]:
    """Stable label order for tie-breaking, including numbered repeated types."""
    normalized = re.sub(r"[^a-zA-Z0-9]", " ", label).lower().strip()
    match = re.search(r"(\d+)$", normalized)
    number = int(match.group()) if match else 0
    base = normalized[:match.start()].strip() if match else normalized
    return re.sub(r"\s+", " ", base), number, label


@dataclass(frozen=True)
class ComponentSpan:
    label: str
    value: str
    start: int
    end: int


def locate_components(
    original: str, components: Mapping[str, str],
) -> tuple[ComponentSpan, ...] | None:
    """Place every nonempty component; mapping order carries no string order."""
    return select_component_spans(original, {
        str(label): {str(value): 1}
        for label, value in components.items() if value is not None and str(value) != ""
    })


def select_component_spans(
    original: str, candidates: Mapping[str, Mapping[str, int]],
) -> tuple[ComponentSpan, ...] | None:
    """Maximize support for one non-overlapping occurrence per type, in any order.

    Equal scores use stable label/value/position tie-breaking. Repeated substrings
    can use separate occurrences; a first-occurrence sort alone cannot do this.
    None means no complete placement exists, not that the split should be rejected.
    """
    options = {}
    for label, values in candidates.items():
        placements = []
        for value, support in (values or {"": 0}).items():
            start = original.find(value)
            while start >= 0:
                end = start + len(value)
                occupied = ((1 << len(value)) - 1) << start
                placements.append((ComponentSpan(label, value, start, end), support, occupied))
                if not value:
                    break
                start = original.find(value, start + 1)
        if not placements:
            return None
        options[label] = sorted(placements, key=lambda item: (
            -item[1], item[0].start, item[0].end, item[0].value,
        ))

    # Constrained types first reduces backtracking for repeated/overlapping values.
    labels = sorted(options, key=lambda label: (len(options[label]), component_key(label)))
    upper_bounds = [0] * (len(labels) + 1)
    for index in range(len(labels) - 1, -1, -1):
        upper_bounds[index] = upper_bounds[index + 1] + options[labels[index]][0][1]

    @lru_cache(maxsize=None)
    def choose(index: int, occupied: int) -> tuple[int, tuple[ComponentSpan, ...]] | None:
        if index == len(labels):
            return 0, ()
        best = None
        for span, support, mask in options[labels[index]]:
            if mask & occupied:
                continue
            if best is not None and support + upper_bounds[index + 1] <= best[0]:
                continue
            suffix = choose(index + 1, occupied | mask)
            if suffix is not None and (best is None or support + suffix[0] > best[0]):
                best = support + suffix[0], (span,) + suffix[1]
                if best[0] == upper_bounds[index]:
                    break
        return best

    result = choose(0, 0)
    if result is None:
        return None
    return tuple(sorted(result[1], key=lambda span: (span.start, span.end, component_key(span.label))))
