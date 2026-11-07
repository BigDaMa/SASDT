from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterator, List, Mapping, Sequence

from LLMUtil.ResponseParser import RParser
from gdt.components import component_key, locate_components


@dataclass(frozen=True, slots=True)
class SemanticType:
    """A reference to one occurrence of a semantic type in an input string."""

    name: str
    normalized_name: str
    occurrence: int = 0


@dataclass(frozen=True, slots=True)
class ConstantString:
    """Literal output text inserted around semantic components."""

    value: str


TransformationPart = SemanticType | ConstantString
ComponentReference = tuple[str, int]


class Transform:
    @staticmethod
    def transform(
        example_input_str: Dict[int, str],
        input_components: Dict[int, Dict[str, str]],
        example_output_str: Dict[int, str],
        output_components: Dict[int, Dict[str, str]],
        query_components: Dict[object, Dict[str, object]],
        examples: List[Dict[str, str]],
    ) -> Dict[int, Dict[str, object]]:
        """Infer and apply a function while preserving the historical return type."""

        transformations, _function = Transform.transform_with_function(
            example_input_str,
            input_components,
            example_output_str,
            output_components,
            query_components,
            examples,
        )
        return transformations

    @staticmethod
    def transform_with_function(
        example_input_str: Dict[int, str],
        input_components: Dict[int, Dict[str, str]],
        example_output_str: Dict[int, str],
        output_components: Dict[int, Dict[str, str]],
        query_components: Dict[object, Dict[str, object]],
        examples: List[Dict[str, str]],
    ) -> tuple[Dict[int, Dict[str, object]], List[TransformationPart]]:
        """Return both transformed queries and the inferred reusable function."""

        example_outputs = Transform._example_outputs(
            example_input_str,
            example_output_str,
            input_components,
            examples,
        )
        function = Transform.find_transformation_function(
            input_components,
            output_components,
            example_outputs,
        )

        transformations: Dict[int, Dict[str, object]] = {}
        for query_key, payload in query_components.items():
            value_id = int(payload["id"])
            original = str(payload.get("text", query_key))
            components = payload["components"]
            if not isinstance(components, dict):
                continue
            transformations[value_id] = {
                "transformation": Transform.apply_transformation_function(
                    function, components
                ).strip(),
                "orig_txt": original,
                "original": components,
            }
        return transformations, function

    @staticmethod
    def find_transformation_function(
        input_components: Mapping[int, Mapping[str, str]],
        output_components: Mapping[int, Mapping[str, str]],
        example_outputs: Mapping[int, str],
    ) -> List[TransformationPart]:
        """Select a complete template by output support, then total constant length.

        Every candidate keeps one example's actual output order and literal gaps.
        Equal-valued input components supply alternative references. Each usable
        pair casts at most one vote per complete candidate, by exact reproduction
        of its output. Constant-only templates rank below every template with an
        input reference. Accept the first semantic template matching all usable
        examples; otherwise prefer more matches, then fewer literal characters.
        Remaining ties retain the first candidate obtained in example order.
        """

        example_ids = [
            value_id
            for value_id in example_outputs
            if input_components.get(value_id) and output_components.get(value_id)
        ]
        if not example_ids:
            # Empty LM results supply no evidence for semantic references or
            # literals. An empty function still produces evaluable predictions.
            return []

        best_function: List[TransformationPart] = []
        best_rank: tuple[bool, int, int] | None = None
        seen = set()
        for value_id in example_ids:
            for function in Transform._example_templates(
                input_components[value_id], output_components[value_id],
                example_outputs[value_id], value_id,
            ):
                key = tuple(
                    ("constant", part.value) if isinstance(part, ConstantString)
                    else ("semantic", part.normalized_name, part.occurrence)
                    for part in function
                )
                if key in seen:
                    continue
                seen.add(key)
                support = sum(
                    Transform.apply_transformation_function(
                        function, input_components[example_id]
                    ).strip() == example_outputs[example_id].strip()
                    for example_id in example_ids
                )
                has_reference = any(isinstance(part, SemanticType) for part in function)
                if has_reference and support == len(example_ids):
                    return function
                constant_length = sum(
                    len(part.value) for part in function if isinstance(part, ConstantString)
                )
                rank = (has_reference, support, -constant_length)
                if best_rank is None or rank > best_rank:
                    best_function, best_rank = function, rank
        return best_function

    @staticmethod
    def apply_transformation_function(
        function: Sequence[TransformationPart],
        components: Mapping[str, object],
    ) -> str:
        """Apply available semantic references and skip types absent from this input."""

        output = []
        for part in function:
            if isinstance(part, ConstantString):
                output.append(part.value)
                continue
            value = Transform._component_value(components, part)
            if value is None:
                continue
            output.append(str(value))
        return "".join(output)

    @staticmethod
    def describe_transformation_function(
        function: Sequence[TransformationPart],
    ) -> list[dict[str, object]]:
        """Return a concise JSON-ready description for diagnostics and tests."""

        return [
            {"constant string": part.value}
            if isinstance(part, ConstantString)
            else {
                "semantic type": part.name,
                "occurrence": part.occurrence + 1,
            }
            for part in function
        ]

    @staticmethod
    def _example_outputs(
        example_input_str: Mapping[int, str],
        example_output_str: Mapping[int, str],
        input_components: Mapping[int, Mapping[str, str]],
        examples: Sequence[Mapping[str, str]],
    ) -> Dict[int, str]:
        paired_output_values = dict(zip(example_input_str, example_output_str.values()))
        result = {}
        for value_id in input_components:
            if 0 <= value_id < len(examples) and "output" in examples[value_id]:
                result[value_id] = str(examples[value_id]["output"])
            elif value_id in paired_output_values:
                result[value_id] = str(paired_output_values[value_id])
        return result

    @staticmethod
    def _example_templates(
        inputs: Mapping[str, str], outputs: Mapping[str, str],
        output: str, example_id: int,
    ) -> Iterator[List[TransformationPart]]:
        spans = locate_components(output, outputs)
        if not spans:
            # Missing/overlapping evidence cannot define a complete template.
            # Other examples can still supply a function; never abort the run.
            return
        slots = [(span.label, span.value) for span in spans]
        gaps = []
        cursor = 0
        for span in spans:
            gaps.append(output[cursor:span.start])
            cursor = span.end
        gaps.append(output[cursor:])
        options = []
        for output_type, value in slots:
            references = sorted(
                Transform._candidate_references(inputs, output_type, value),
                key=lambda reference: (
                    reference[0] != Transform._normalize_type(output_type),
                    reference[1], reference[0],
                ),
            )
            options.append([
                SemanticType(Transform._reference_name({example_id: inputs}, [example_id], ref),
                             ref[0], ref[1])
                for ref in references
            ] or [ConstantString(value)])
        for parts in product(*options):
            function: List[TransformationPart] = []
            for index, part in enumerate(parts):
                Transform._append_constant(function, gaps[index])
                Transform._append_part(function, part)
            Transform._append_constant(function, gaps[-1])
            yield function

    @staticmethod
    def _candidate_references(
        components: Mapping[str, str],
        output_type: str,
        output_value: str,
    ) -> set[ComponentReference]:
        del output_type
        return {
            Transform._reference_for_key(components, key)
            for key, value in components.items()
            if str(value) == output_value
        }

    @staticmethod
    def _reference_for_key(
        components: Mapping[str, object],
        selected_key: str,
    ) -> ComponentReference:
        del components
        # Occurrence identity comes from the type's suffix, never mapping order.
        return Transform._normalize_type(selected_key), max(component_key(selected_key)[1] - 1, 0)

    @staticmethod
    def _reference_name(
        input_components: Mapping[int, Mapping[str, str]],
        example_ids: Sequence[int],
        reference: ComponentReference,
    ) -> str:
        for value_id in example_ids:
            matching_keys = [
                str(key)
                for key in sorted(input_components[value_id], key=component_key)
                if Transform._reference_for_key(input_components[value_id], key) == reference
            ]
            if matching_keys:
                return matching_keys[0]
        return reference[0]

    @staticmethod
    def _component_value(
        components: Mapping[str, object],
        semantic_type: SemanticType,
    ) -> object | None:
        if semantic_type.name in components:
            return components[semantic_type.name]
        reference = semantic_type.normalized_name, semantic_type.occurrence
        for key in sorted(components, key=component_key):
            if Transform._reference_for_key(components, key) == reference:
                return components[key]
        return None

    @staticmethod
    def _constant_gaps(
        output: str,
        component_values: Sequence[str],
        example_id: int,
    ) -> list[str | None]:
        """Extract known gaps; missing boundaries do not cast a constant vote."""
        del example_id
        gaps = []
        cursor = 0
        previous_matched = True
        for value in component_values:
            position = output.find(value, cursor)
            if position < 0:
                gaps.append(None)
                previous_matched = False
                continue
            gaps.append(output[cursor:position] if previous_matched else None)
            cursor = position + len(value)
            previous_matched = True
        gaps.append(output[cursor:] if previous_matched else None)
        return gaps

    @staticmethod
    def _append_constant(
        function: List[TransformationPart],
        value: str,
    ) -> None:
        if value:
            Transform._append_part(function, ConstantString(value))

    @staticmethod
    def _append_part(
        function: List[TransformationPart],
        part: TransformationPart,
    ) -> None:
        if (
            isinstance(part, ConstantString)
            and function
            and isinstance(function[-1], ConstantString)
        ):
            previous = function[-1]
            assert isinstance(previous, ConstantString)
            function[-1] = ConstantString(previous.value + part.value)
        else:
            function.append(part)

    @staticmethod
    def _normalize_type(value: str) -> str:
        normalized = RParser.key_normalize(value)
        return normalized or value.strip().lower()
