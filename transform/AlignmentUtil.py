from __future__ import annotations

from collections import Counter, defaultdict
import itertools
from typing import Dict, List, Sequence, Tuple

from LLMUtil.ResponseParser import RParser


Components = Dict[int, Dict[str, str]]


class AlignmentUtil:
    @staticmethod
    def require_further_split(
        input_components: Components,
        output_components: Components,
        expand_input_key: Dict[str, Tuple[str, ...]],
    ) -> List[str]:
        common_ids = input_components.keys() & output_components.keys()
        if not common_ids:
            return []
        mismatched: Counter[str] = Counter()
        for value_id in common_ids:
            inputs = input_components[value_id]
            input_values = set(inputs.values())
            for output_key, output_value in output_components[value_id].items():
                related = [
                    input_key
                    for input_key in inputs
                    if RParser.key_normalize(input_key) == RParser.key_normalize(output_key)
                ]
                if output_value not in input_values and related:
                    mismatched[output_key] += 1
                    mismatched.update(related)
        threshold = len(common_ids) / 3
        result = [key for key, count in mismatched.items() if count > threshold]
        if not expand_input_key:
            input_keys = {key for value in input_components.values() for key in value}
            output_keys = {key for value in output_components.values() for key in value}
            if input_keys and output_keys and not input_keys & output_keys:
                result.extend(sorted(input_keys | output_keys))
        return list(dict.fromkeys(result))

    @staticmethod
    def align_output_component_type_to_input(
        output_components: Components,
        input_components: Components,
    ) -> Dict[str, List[str]]:
        common_ids = output_components.keys() & input_components.keys()
        counts: Counter[tuple[str, str]] = Counter()
        for value_id in common_ids:
            for output_key, output_value in output_components[value_id].items():
                for input_key, input_value in input_components[value_id].items():
                    if (
                        RParser.key_normalize(output_value)
                        == RParser.key_normalize(input_value)
                        and output_key != input_key
                    ):
                        counts[(output_key, input_key)] += 1
        alignment: dict[str, list[str]] = defaultdict(list)
        for (output_key, input_key), count in counts.items():
            if count > len(common_ids) / 2:
                alignment[output_key].append(input_key)
        return dict(alignment)

    @staticmethod
    def align_output_component_type_to_input_longest(
        output_components_orig: Components,
        input_components_orig: Components,
    ) -> Dict[str, List[str]]:
        if not output_components_orig:
            return {}
        maximum = max(map(len, output_components_orig.values()))
        selected_ids = {
            value_id
            for value_id, components in output_components_orig.items()
            if len(components) == maximum and value_id in input_components_orig
        }
        return AlignmentUtil.align_output_component_type_to_input(
            {value_id: output_components_orig[value_id] for value_id in selected_ids},
            {value_id: input_components_orig[value_id] for value_id in selected_ids},
        )

    @staticmethod
    def _get_constant_(
        keys: List[str],
        output_components: Components,
        input_components: Components,
    ) -> List[str]:
        result = list(keys or [])
        total = len(output_components)
        if not total:
            return result
        for index, key in enumerate(result):
            values = Counter(
                components[key]
                for components in output_components.values()
                if key in components
            )
            if values:
                value, count = values.most_common(1)[0]
                copied_from_input = sum(
                    components.get(key) == value
                    or value in components.values()
                    for value_id, components in input_components.items()
                    if value_id in output_components
                )
                if count >= total * 0.75 and copied_from_input < count * 0.75:
                    result[index] = f"[CONST]{value}"
        return result

    @staticmethod
    def clean_final_shortest_key(
        shortest_key: List[str],
        expanded_keys: Dict[str, Tuple[str, ...]],
        alignment: Dict[str, List[str]],
        examples: List[Dict[str, str]],
        input_components: Components,
        output_components: Components,
    ) -> List[str]:
        return AlignmentUtil._clean_key_sequence(
            shortest_key,
            expanded_keys,
            alignment,
            examples,
            input_components,
            output_components,
            require_majority=True,
        )

    @staticmethod
    def clean_final_longest_key(
        longest_key: List[str],
        expanded_keys: Dict[str, Tuple[str, ...]],
        alignment: Dict[str, List[str]],
        examples: List[Dict[str, str]],
        input_components: Components,
        output_components: Components,
    ) -> List[str]:
        return AlignmentUtil._clean_key_sequence(
            longest_key,
            expanded_keys,
            alignment,
            examples,
            input_components,
            output_components,
            require_majority=False,
        )

    @staticmethod
    def _clean_key_sequence(
        keys: Sequence[str],
        expanded_keys: Dict[str, Tuple[str, ...]],
        alignment: Dict[str, List[str]],
        examples: List[Dict[str, str]],
        input_components: Components,
        output_components: Components,
        *,
        require_majority: bool,
    ) -> List[str]:
        expanded: list[str] = []
        for key in keys:
            replacements = list(expanded_keys.get(key, ()))
            if not replacements:
                replacements = [item for item in alignment.get(key, []) if item not in keys]
            for replacement in replacements or [key]:
                if replacement not in expanded:
                    expanded.append(replacement)
        if not examples:
            return AlignmentUtil._get_constant_(
                expanded, output_components, input_components
            )

        orders = []
        presence = Counter()
        for value_id in output_components.keys() & input_components.keys():
            if not 0 <= value_id < len(examples):
                continue
            output = examples[value_id]["output"]
            values = output_components[value_id] | input_components[value_id]
            positions = {
                key: output.find(values[key])
                for key in expanded
                if key in values and values[key] and output.find(values[key]) >= 0
            }
            order = tuple(sorted(positions, key=positions.get))
            orders.append(order)
            presence.update(order)
        if not orders:
            return AlignmentUtil._get_constant_(
                expanded, output_components, input_components
            )
        if require_majority:
            allowed = {key for key, count in presence.items() if count >= len(orders) / 2}
            orders = [tuple(key for key in order if key in allowed) for order in orders]
        winner = Counter(orders).most_common(1)[0][0]
        return AlignmentUtil._get_constant_(
            list(winner), output_components, input_components
        )

    @staticmethod
    def align_component_id(
        input_components: Components,
        output_components: Components,
    ) -> Dict[str, str]:
        counts: Counter[tuple[str, str]] = Counter()
        common_ids = input_components.keys() & output_components.keys()
        for value_id in common_ids:
            for input_key, input_value in input_components[value_id].items():
                if not input_key.strip()[-1:].isdigit():
                    continue
                for output_key, output_value in output_components[value_id].items():
                    if (
                        RParser.key_normalize(input_key) == RParser.key_normalize(output_key)
                        and input_value == output_value
                    ):
                        counts[(output_key, input_key)] += 1
        return {
            output_key: input_key
            for (output_key, input_key), count in counts.items()
            if count > len(common_ids) / 2
        }

    @staticmethod
    def align_extra_split(output_keys: List[str], input_keys: List[str]) -> List[str]:
        output_normalized = {RParser.key_normalize(key) for key in output_keys}
        return [
            key
            for key in input_keys
            if key.strip()[-1:].isdigit()
            and RParser.key_normalize(key) in output_normalized
        ]

    @staticmethod
    def process_input_more_detailed(
        input_components: Components,
        output_components: Components,
    ) -> Dict[str, Tuple[str, ...]]:
        common_ids = input_components.keys() & output_components.keys()
        counts: Counter[tuple[str, tuple[str, ...]]] = Counter()
        for value_id in common_ids:
            inputs = input_components[value_id]
            for output_key, output_value in output_components[value_id].items():
                normalized_output = RParser.key_normalize(output_value)
                keys = list(inputs)
                for size in range(2, len(keys) + 1):
                    for combination in itertools.permutations(keys, size):
                        combined = "".join(
                            RParser.key_normalize(inputs[key]) for key in combination
                        )
                        if combined == normalized_output:
                            counts[(output_key, combination)] += 1
                            break
                    else:
                        continue
                    break
        result = {}
        for output_key in {key for key, _combination in counts}:
            options = [
                (combination, count)
                for (key, combination), count in counts.items()
                if key == output_key and count > len(common_ids) / 2
            ]
            if options:
                result[output_key] = min(options, key=lambda item: (len(item[0]), -item[1]))[0]
        return result

    @staticmethod
    def align_sem_with_synt(
        examples: List[Dict[str, str]],
        input_components: Components,
        output_components: Components,
        expanded_output_keys: Dict[str, Tuple[str, ...]],
    ) -> Dict[str, List[str]]:
        del examples
        excluded = set(expanded_output_keys) | {
            key for values in expanded_output_keys.values() for key in values
        }
        counts: Counter[tuple[str, tuple[str, ...]]] = Counter()
        common_ids = input_components.keys() & output_components.keys()
        for value_id in common_ids:
            inputs = input_components[value_id]
            for output_key, output_value in output_components[value_id].items():
                if output_key in excluded:
                    continue
                found = [
                    key for key, value in inputs.items() if value and value in output_value
                ]
                if found and output_key not in found:
                    found.sort(key=lambda key: output_value.index(inputs[key]))
                    counts[(output_key, tuple(found))] += 1
        return {
            output_key: list(keys)
            for (output_key, keys), count in counts.items()
            if count > len(common_ids) / 2
        }

    @staticmethod
    def align_with_synt_str(
        examples: List[Dict[str, str]],
        input_components: Components,
        output_components: Components,
        shortest_keys: List[str],
        expanded_output_keys: Dict[str, Tuple[str, ...]],
        aligned_keys: Dict[str, List[str]],
    ) -> List[str]:
        templates = []
        for value_id, outputs in output_components.items():
            if value_id not in input_components or not 0 <= value_id < len(examples):
                continue
            output = examples[value_id]["output"]
            inputs = input_components[value_id]
            cursor = 0
            template = []
            for key in shortest_keys:
                if key.startswith("[CONST]"):
                    continue
                candidates = list(expanded_output_keys.get(key, ()))
                candidates.extend(aligned_keys.get(key, []))
                candidates.append(key)
                selected = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate in outputs | inputs and (outputs | inputs)[candidate]
                    ),
                    None,
                )
                if selected is None:
                    continue
                value = (outputs | inputs)[selected]
                position = output.find(value, cursor)
                if position < 0:
                    continue
                if position > cursor:
                    template.append(f"[CONST]{output[cursor:position]}")
                template.append(key)
                cursor = position + len(value)
            if cursor < len(output):
                template.append(f"[CONST]{output[cursor:]}")
            templates.append(template)
        return AlignmentUtil._check_tuned_alignment(templates, examples)

    @staticmethod
    def _check_tuned_alignment(
        tuned_alignments: List[List[str]],
        examples: List[Dict[str, str]],
    ) -> List[str]:
        del examples
        if not tuned_alignments:
            return []
        counts = Counter(tuple(alignment) for alignment in tuned_alignments)
        return list(
            min(
                counts,
                key=lambda alignment: (
                    -counts[alignment],
                    -len(alignment),
                    sum(key.startswith("[CONST]") for key in alignment),
                    alignment,
                ),
            )
        )
