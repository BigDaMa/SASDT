from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


_BRACKETS = frozenset("()[]{}")


@dataclass(frozen=True, slots=True)
class RegexProcessor:
    r: str

    def getregex(self) -> str:

        return self.r

    def __str__(self) -> str:
        return self.r

    def regex_clean(self) -> "RegexProcessor":

        result: list[str] = []
        index = 0
        while index < len(self.r):
            atom, next_index = _read_atom(self.r, index)
            has_quantifier = next_index < len(self.r) and self.r[next_index] == "{"
            if atom in _BRACKETS or has_quantifier:
                result.append(atom)
                index = next_index
                continue

            count = 1
            cursor = next_index
            while cursor < len(self.r):
                candidate, candidate_end = _read_atom(self.r, cursor)
                if candidate != atom or candidate in _BRACKETS:
                    break
                if candidate_end < len(self.r) and self.r[candidate_end] == "{":
                    break
                count += 1
                cursor = candidate_end
            result.append(f"{atom}{{{count}}}" if count > 1 else atom)
            index = cursor
        return RegexProcessor("".join(result))

    def expand(self) -> list[str]:

        return _expand_alternatives(self.r)


def build_regex(data: Sequence[str]) -> str:

    if not data:
        raise ValueError("at least one string is required to build a regex")
    try:
        from grex import RegExpBuilder
    except ImportError as error:
        raise RuntimeError(
            "regex clustering requires the optional 'grex' package"
        ) from error

    return (
        RegExpBuilder.from_test_cases(list(data))
        .with_conversion_of_digits()
        .with_conversion_of_whitespace()
        .with_conversion_of_words()
        .with_conversion_of_repetitions()
        .with_minimum_substring_length(1)
        .with_capturing_groups()
        .build()
    )


def _read_atom(pattern: str, index: int) -> tuple[str, int]:
    if pattern[index] == "\\" and index + 1 < len(pattern):
        return pattern[index : index + 2], index + 2
    return pattern[index], index + 1


def _expand_alternatives(pattern: str) -> list[str]:
    alternatives = _split_top_level(pattern, "|")
    if len(alternatives) > 1:
        return [
            item
            for alternative in alternatives
            for item in _expand_alternatives(alternative)
        ]

    prefixes = [""]
    index = 0
    while index < len(pattern):
        if pattern[index] == "\\" and index + 1 < len(pattern):
            prefixes = [prefix + pattern[index : index + 2] for prefix in prefixes]
            index += 2
            continue
        if pattern[index] != "(":
            prefixes = [prefix + pattern[index] for prefix in prefixes]
            index += 1
            continue

        closing = _matching_parenthesis(pattern, index)
        inner = pattern[index + 1 : closing]
        inner_alternatives = _split_top_level(inner, "|")
        if len(inner_alternatives) == 1:
            fragments = [f"({inner})"]
        else:
            fragments = [
                fragment
                for alternative in inner_alternatives
                for fragment in _expand_alternatives(alternative)
            ]
        prefixes = [prefix + fragment for prefix in prefixes for fragment in fragments]
        index = closing + 1
    return prefixes


def _split_top_level(pattern: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    escaped = False
    for index, character in enumerate(pattern):
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced parenthesis in regex")
        elif character == delimiter and depth == 0:
            parts.append(pattern[start:index])
            start = index + 1
    if depth:
        raise ValueError("unbalanced parenthesis in regex")
    parts.append(pattern[start:])
    return parts


def _matching_parenthesis(pattern: str, opening: int) -> int:
    depth = 0
    escaped = False
    for index in range(opening, len(pattern)):
        character = pattern[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unbalanced parenthesis in regex")
