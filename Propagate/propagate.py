from __future__ import annotations

from gdt.timing import output_time

from collections import Counter
from typing import Dict, List, Tuple

from gdt.components import locate_components, select_component_spans
from Propagate.position import Position
from Propagate.positionop import get_str_from_pos
from Propagate.tokenutil import is_valid_extraction
from Utils.CacheUtil import cache_result
from Utils.config import Config
from Utils.IterPrintUtil import IterPrintUtil


PositionPair = Tuple[Position, Position]


class Propagate:
    @staticmethod
    def can_assemble_without_overlap(original: str, strings: List[str]) -> bool:
        return locate_components(original, {
            str(index): value for index, value in enumerate(strings)
        }) is not None

    @staticmethod
    def _can_assemble_from_position(
        original: str,
        string_list: List[str],
        start_pos: int,
    ) -> bool:
        if not string_list:
            return True
        if original.find(string_list[0], start_pos) != start_pos:
            return False
        return Propagate.can_assemble_without_overlap(
            original[start_pos:], string_list
        )

    @staticmethod
    @cache_result(version="unordered-components-v2")
    def propagate(
        clusters: List[Dict[int, str]],
        positions: Dict[int, Dict[str, List[PositionPair]]],
        sem_components_selected: Dict[int, Dict[str, str]],
    ) -> Dict[int, Dict[str, str]]:
        propagated: Dict[int, Dict[str, str]] = {}
        for cluster_index, cluster in enumerate(clusters):
            operators = positions.get(cluster_index, {})
            printed_items = []
            for value_id, original in cluster.items():
                if value_id in sem_components_selected:
                    propagated[value_id] = dict(sem_components_selected[value_id])
                    candidates = {}
                else:
                    candidates = {
                        component_type: Propagate._extract_candidates(
                            original, operator_list
                        )
                        for component_type, operator_list in operators.items()
                    }
                    propagated[value_id] = Propagate._select_candidates(
                        original, candidates
                    )
                if Config.verbose:
                    with output_time():
                        printed_items.append(
                            {
                                "value id": value_id,
                                "original string": original,
                                "propagation": propagated[value_id],
                            }
                        )
            if Config.verbose:
                with output_time():
                    print(f"Propagation for cluster {cluster_index}:")
                    IterPrintUtil.print(printed_items)
        return propagated

    @staticmethod
    def _extract_candidates(
        original: str,
        operators: List[PositionPair],
    ) -> Counter[str]:
        candidates: Counter[str] = Counter()
        for pair in operators:
            extracted = get_str_from_pos(original, pair)
            start = pair[0].get_index(original)
            end = pair[1].get_index(original)
            if is_valid_extraction(original, extracted, start, end):
                # Validate the original boundaries before trimming the value.
                extracted = extracted.strip()
                if extracted:
                    candidates[extracted] += 1
        return candidates

    @staticmethod
    def _select_candidates(
        original: str,
        candidates: Dict[str, Counter[str]],
    ) -> Dict[str, str]:
        if not candidates:
            return {}
        spans = select_component_spans(original, candidates)
        if spans is not None:
            return {span.label: span.value for span in spans}
        # Preserve the existing fallback when no complete non-overlapping choice
        # exists, with ties independent of the order in which operators arrived.
        return {
            label: min(counter, key=lambda value: (
                -counter[value], original.find(value), len(value), value,
            )) if counter else ""
            for label, counter in candidates.items()
        }
