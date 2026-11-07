from __future__ import annotations


def is_valid_extraction(
    original: str,
    extracted: str,
    start_index: int | None,
    end_index: int | None,
) -> bool:
    if not extracted or start_index is None:
        return False
    length = len(original)
    start = length + start_index if start_index < 0 else start_index
    if end_index is None:
        end = length
    else:
        end = length + end_index if end_index < 0 else end_index
    if not (0 <= start < end <= length):
        return False
    if original[start:end] != extracted:
        return False
    if start > 0 and original[start - 1].isalnum() and original[start].isalnum():
        return False
    if end < length and original[end - 1].isalnum() and original[end].isalnum():
        return False
    return True
