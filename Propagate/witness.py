from __future__ import annotations

from gdt.timing import output_time

from collections import Counter
import itertools
import re
from typing import Dict, Iterable, List, Tuple

from Propagate.position import AbsPosPosition, Position, RegExPosPosition
from Propagate.positionop import get_str_from_pos
from Propagate.regexutil import RegExUtil
from Propagate.tokenutil import is_valid_extraction
from Utils.CacheUtil import cache_result, perfect_operator_override
from Utils.config import Config
from Utils.IterPrintUtil import IterPrintUtil


PositionPair = Tuple[Position, Position]


def dedup(sequence: Iterable[PositionPair]) -> List[PositionPair]:
    return list(dict.fromkeys(sequence))


def WitnessPosition(spec: str, currspec: str) -> List[PositionPair]:
    if not currspec:
        return []
    candidates: list[PositionPair] = []
    for occurrence in re.finditer(re.escape(currspec), spec):
        start, end = occurrence.span()
        left = _absolute_positions(start, len(spec), is_end=False)
        left.extend(_regex_positions_at(spec, start))
        right = _absolute_positions(end, len(spec), is_end=True)
        right.extend(_regex_positions_at(spec, end))
        candidates.extend(itertools.product(dedup(left), dedup(right)))
    return dedup(candidates)


def WitnessAbsPos(
    spec: str,
    currspec: str,
) -> Tuple[Dict[int, List[AbsPosPosition]], Dict[int, List[AbsPosPosition]]]:
    left = _WitnessAbsPosLeft(spec, currspec)
    return left, _WitnessAbsPosRight(spec, currspec, left)


def _WitnessAbsPosLeft(spec: str, currspec: str) -> Dict[int, List[AbsPosPosition]]:
    if not currspec:
        return {}
    return {
        match.start(): _absolute_positions(match.start(), len(spec), is_end=False)
        for match in re.finditer(re.escape(currspec), spec)
    }


def _WitnessAbsPosRight(
    spec: str,
    currspec: str,
    leftidx: Dict[int, List[AbsPosPosition]],
) -> Dict[int, List[AbsPosPosition]]:
    return {
        start: _absolute_positions(start + len(currspec), len(spec), is_end=True)
        for start in leftidx
    }


def WitnessRegExPos(
    spec: str,
    currspec: str,
    absposLeft: Dict[int, List[AbsPosPosition]],
) -> Tuple[Dict[int, List[Position]], Dict[int, List[Position]]]:
    del absposLeft
    if not currspec:
        return {}, {}
    left: Dict[int, List[Position]] = {}
    right: Dict[int, List[Position]] = {}
    for match in re.finditer(re.escape(currspec), spec):
        left[match.start()] = _regex_positions_at(spec, match.start())
        right[match.start()] = _regex_positions_at(spec, match.end())
    return left, right


def _WitnessRegExPosLeft(spec: str, currspec: str) -> Dict[int, List[RegExPosPosition]]:
    if not currspec:
        return {}
    return {
        match.start(): _regex_positions_at(spec, match.start())
        for match in re.finditer(re.escape(currspec), spec)
    }


def _WitnessRegExPosRight(
    spec: str,
    currspec: str,
    lidxmap: Dict[int, List[RegExPosPosition]],
) -> Dict[int, List[RegExPosPosition]]:
    return {
        start: _regex_positions_at(spec, start + len(currspec))
        for start in lidxmap
    }


@cache_result()
def get_pos_dict(
    sem_components: Dict[int, Dict[str, str]],
    query_strs: List[str],
    query_str_idx: List[int],
    file: str,
) -> Dict[int, Dict[str, List[PositionPair]]]:
    del file
    query_by_id = dict(zip(query_str_idx, query_strs))
    result: Dict[int, Dict[str, List[PositionPair]]] = {}
    for value_id, components in sem_components.items():
        if value_id not in query_by_id:
            raise KeyError(f"semantic split ID {value_id} has no query string")
        result[value_id] = {
            component_type: WitnessPosition(query_by_id[value_id], component_value)
            for component_type, component_value in components.items()
            if component_value
        }
    return result


@perfect_operator_override()
def get_cluster_operators(
    mapped_clusters: List[Dict[int, str]],
    sem_components: Dict[int, Dict[str, str]],
    query_strs: List[str],
    query_str_idx: List[int],
    file: str,
    *,
    output_ids: set[int] | None = None,
) -> Dict[int, Dict[str, List[PositionPair]]]:
    """Return the operator-stage result consumed by propagation.

    The ground-truth provider replaces this complete result. Otherwise infer
    witness candidates and select them using the normal cluster validation.
    Output-only clusters whose splits are all available need neither step.
    """
    final_clusters = {
        index for index, cluster in enumerate(mapped_clusters)
        if cluster and cluster.keys() <= (output_ids or set())
        and cluster.keys() <= sem_components.keys()
    }
    final_ids = {
        value_id for index in final_clusters for value_id in mapped_clusters[index]
    }
    witness_components = {
        value_id: components for value_id, components in sem_components.items()
        if value_id not in final_ids
    }
    positions = get_pos_dict(witness_components, query_strs, query_str_idx, file)
    return validate_pos_dict(
        mapped_clusters, positions, sem_components, skip_clusters=final_clusters,
    )


def get_common_operators(pos_lists: List[List[PositionPair]]) -> List[PositionPair]:
    if not pos_lists:
        return []
    common = set(pos_lists[0])
    for positions in pos_lists[1:]:
        common.intersection_update(positions)
    return list(common)


@cache_result()
def validate_pos_dict(
    mapped_clusters: List[Dict[int, str]],
    comp_pos_dict: Dict[int, Dict[str, List[PositionPair]]],
    sem_components_selected: Dict[int, Dict[str, str]],
    *,
    skip_clusters: set[int] | None = None,
) -> Dict[int, Dict[str, List[PositionPair]]]:
    selected: Dict[int, Dict[str, List[PositionPair]]] = {}
    for cluster_index, cluster in enumerate(mapped_clusters):
        selected[cluster_index] = {}
        if skip_clusters and cluster_index in skip_clusters:
            continue
        witness_ids = [value_id for value_id in cluster if value_id in comp_pos_dict]
        candidates_by_type: dict[str, Counter[PositionPair]] = {}
        for value_id in witness_ids:
            for component_type, positions in comp_pos_dict[value_id].items():
                candidates_by_type.setdefault(component_type, Counter()).update(set(positions))

        for component_type, support in candidates_by_type.items():
            if not support:
                selected[cluster_index][component_type] = []
                continue
            maximum_witness_support = max(support.values())
            candidates = [
                position for position, count in support.items() if count == maximum_witness_support
            ]
            scores = {
                position: _score_position(
                    position,
                    component_type,
                    cluster,
                    sem_components_selected,
                )
                for position in candidates
            }
            best_score = max(scores.values(), default=0)
            selected[cluster_index][component_type] = [
                position for position, score in scores.items() if score == best_score and score > 0
            ]
        if Config.verbose:
            with output_time():
                print(f"Selected operators for cluster {cluster_index}:")
                IterPrintUtil.print(selected[cluster_index])
    return selected


def _score_position(
    position: PositionPair,
    component_type: str,
    cluster: Dict[int, str],
    semantic_components: Dict[int, Dict[str, str]],
) -> int:
    score = 0
    for value_id, original in cluster.items():
        extracted = get_str_from_pos(original, position)
        start = position[0].get_index(original)
        end = position[1].get_index(original)
        if not is_valid_extraction(original, extracted, start, end):
            continue
        expected = semantic_components.get(value_id, {}).get(component_type)
        if expected is None or extracted == expected:
            score += 1
    return score


def _absolute_positions(
    boundary: int,
    length: int,
    *,
    is_end: bool,
) -> List[AbsPosPosition]:
    positions = [AbsPosPosition(boundary)]
    if boundary == length and is_end:
        positions.append(AbsPosPosition(None))
    else:
        positions.append(AbsPosPosition(boundary - length))
    return list(dict.fromkeys(positions))


def _regex_positions_at(value: str, boundary: int) -> List[RegExPosPosition]:
    positions = []
    for pattern in RegExUtil.regexes:
        matches = list(re.finditer(pattern, value))
        for occurrence, match in enumerate(matches):
            negative_occurrence = occurrence - len(matches)
            if match.start() == boundary:
                positions.extend(
                    [
                        RegExPosPosition(pattern, occurrence, "L"),
                        RegExPosPosition(pattern, negative_occurrence, "L"),
                    ]
                )
            if match.end() == boundary:
                positions.extend(
                    [
                        RegExPosPosition(pattern, occurrence, "R"),
                        RegExPosPosition(pattern, negative_occurrence, "R"),
                    ]
                )
    return list(dict.fromkeys(positions))
