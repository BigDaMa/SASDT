from __future__ import annotations

import re
from typing import Dict, List


ITEM_PATTERN = re.compile(r"\[ITEM\]", flags=re.IGNORECASE)


class RParser:
    @staticmethod
    def split_items(response: str) -> list[str]:
        parts = ITEM_PATTERN.split(response)
        if parts and not parts[0].strip():
            parts = parts[1:]
        if parts and not parts[-1].strip():
            parts = parts[:-1]
        return [part.strip() for part in parts]

    @staticmethod
    def parse_semantic_components(components: str) -> List[Dict[str, str]]:
        return list(RParser.parse_component_str(RParser.split_items(components)).values())

    @staticmethod
    def parse_component_str(
        component_str: List[str],
        escapespecchar: bool = True,
    ) -> Dict[int, Dict[str, str]]:
        components: Dict[int, Dict[str, str]] = {}
        for index, item in enumerate(component_str):
            parsed: Dict[str, str] = {}
            for line in item.splitlines():
                text = line.strip().strip("{}")
                if not text:
                    continue
                key, separator, value = text.partition(":")
                if escapespecchar:
                    key = re.sub(r"[^a-zA-Z0-9]", " ", key).lower()
                key = re.sub(r"\s+", " ", key).strip()
                if not key:
                    continue
                value = value.strip() if separator else ""
                if escapespecchar and value:
                    value = re.sub(r"[^a-zA-Z0-9]", " ", value)
                    value = re.sub(r"\s+", " ", value).strip()
                parsed[key] = value
            components[index] = parsed
        return components

    @staticmethod
    def parse_sem_types(sem_types: List[str]) -> str:
        values = [value for value in sem_types if value]
        if len(values) < 2:
            return values[0] if values else ""
        if len(values) == 2:
            return f"{values[0]} and {values[1]}"
        return f"{', '.join(values[:-1])}, and {values[-1]}"

    @staticmethod
    def key_normalize(key: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9]", " ", key).lower().strip()
        normalized = re.sub(r"\d+$", "", normalized)
        normalized = re.sub(r"(?:\(|\s)s(?:\)|\s)$", "", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def key_normalize_with_underscore(key: str) -> str:
        return re.sub(r"\d+$", "", key).lower().strip().rstrip("_")

    @staticmethod
    def remove_string_wrapper(result: str, original: str) -> str:
        if len(result) >= 2 and result[0] == result[-1] and result[0] in {'"', "'"}:
            unwrapped = result[1:-1]
            if unwrapped in original:
                return unwrapped
        return result

    @staticmethod
    def get_types_from_split(response: str) -> List[str]:
        types: list[str] = []
        for item in RParser.split_items(response):
            for line in item.splitlines():
                key, separator, _value = line.partition(":")
                key = key.strip()
                if separator and key and key not in types:
                    types.append(key)
        return types

    @staticmethod
    def require_further_split(
        input_components: Dict[int, Dict[str, str]],
        output_components: Dict[int, Dict[str, str]],
    ) -> bool:
        if not input_components:
            return False
        mismatches = 0
        comparable = 0
        for index in input_components.keys() & output_components.keys():
            comparable += 1
            inputs = input_components[index]
            input_values = set(inputs.values())
            for output_key, output_value in output_components[index].items():
                matching_keys = [
                    key
                    for key in inputs
                    if RParser.key_normalize(key) == RParser.key_normalize(output_key)
                ]
                if output_value not in input_values and any(
                    inputs[key] != output_value for key in matching_keys
                ):
                    mismatches += 1
                    break
        return comparable > 0 and mismatches / comparable > 0.5


class RParser_Prompt(RParser):
    @staticmethod
    def normalize_composition(item: str) -> str:
        """Ignore punctuation, retaining underscores only inside label brackets."""
        characters = []
        in_label = False
        for character in item:
            if character == "[":
                in_label = True
            if character.isalnum() or character in "[]" or (in_label and character == "_"):
                characters.append(character)
            if character == "]":
                in_label = False
        return "".join(characters)

    @staticmethod
    def is_composition(_original: str, item: str) -> bool:
        item = RParser_Prompt.normalize_composition(item)
        if not re.fullmatch(r"(?:\[[^\[\]]+\])+", item):
            return False
        return all(any(character.isalnum() for character in label) and label.upper() != "ITEM"
                   for label in re.findall(r"\[([^\]]+)\]", item))

    @staticmethod
    def parse_component_str(
        component_str: List[str],
        escapespecchar: bool = True,
    ) -> Dict[int, List[str]]:
        components: Dict[int, List[str]] = {}
        for index, item in enumerate(component_str):
            labels = []
            item = RParser_Prompt.normalize_composition(item)
            for match in re.findall(r"\[([^\]]+)\]", item):
                label = match.lower() if escapespecchar else match
                if any(character.isalnum() for character in label) and label.upper() != "ITEM":
                    labels.append(f"[{label}]")
            components[index] = labels
        return components

    @staticmethod
    def get_types_from_split(response: str) -> List[str]:
        types = []
        parsed = RParser_Prompt.parse_component_str(RParser.split_items(response))
        for labels in parsed.values():
            for label in labels:
                if label not in types:
                    types.append(label)
        return types
