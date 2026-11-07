from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import cached_property
from pathlib import Path
import re

from gdt.config import Config
from gdt.components import locate_components
from gdt.data import PairDataset
from gdt.diagnostics import record_ground_truth_error
from Propagate.position import AbsPosPosition, RegExPosPosition


REGEX_ESCAPES = {
    "[ESCAPESPACE]": r"\s",
    "[ESCAPEWORD]": r"\b[A-Za-z]+\b",
    "[ESCAPENUMBER]": r"\d+",
    "[ESCAPECOMMA]": r",",
    "[ESCAPEPERIOD]": r"\.",
    "[ESCAPEAT]": r"@",
    "[ESCAPEDOUBLECOLON]": r":",
    "[ESCAPEBACKSLASH]": r"\\",
    "[ESCAPESLASH]": r"/",
    "[ESCAPEIN]": r"In\s",
    "[ESCAPEDOUBLEQUOTE]": r'\"',
    "[ESCAPELEFTPAREN]": r"\(",
    "[ESCAPERIGHTPAREN]": r"\)",
    "[ESCAPEBY]": r"by\s",
}

_session = ContextVar("ground_truth_session", default=None)


def perfect_path(file: str | Path, suffix: str, directory: str | Path | None = None) -> Path:
    stem = re.sub(r"_[012]$", "", Path(file).stem)
    root = Path(Config.perfect_directory if directory is None else directory)
    path = root / f"{stem}{suffix}"
    # The supplied corpus retains this historical singular type-file spelling.
    if stem == "Festivals" and suffix == "_types.txt" and not path.is_file():
        legacy = root / "Festival_types.txt"
        if legacy.is_file():
            return legacy
    return path


@contextmanager
def ground_truth_session():
    """Parse once per dataset run, without retaining another run's IDs or errors."""
    token = _session.set({})
    try:
        yield
    finally:
        _session.reset(token)


def provide_clusters(file: str) -> list[set[int]]:
    """Supply the same active-ID cluster result as the clustering step."""
    return [set(cluster) for cluster in _get_ground_truth(file).clusters]


def provide_splits(file: str, query: dict[int, str], *, prompt_only: bool = False) -> dict:
    """Supply LM compositions or components, preserving requested IDs and order."""
    return _get_ground_truth(file).split_for(query, prompt_only=prompt_only)


def provide_operators(file: str) -> dict[int, dict]:
    """Supply final cluster operators without witness inference or validation."""
    truth = _get_ground_truth(file)
    operators = truth.operators
    return {
        cluster_index: {label: list(pairs) for label, pairs in
                        operators.get(cluster_index, {}).items()}
        for cluster_index in range(len(truth.clusters))
    }


def _get_ground_truth(file: str) -> GroundTruth:
    path = Path(Config.data_directory) / file
    key = (path.resolve(), Path(Config.perfect_directory).resolve(), Config.num_examples)
    cache = _session.get()
    if cache is not None and key in cache:
        return cache[key]
    truth = GroundTruth(
        PairDataset.from_csv(path), Config.perfect_directory, Config.num_examples,
    )
    if cache is not None:
        cache[key] = truth
    return truth


class GroundTruth:
    """Load active truth lazily, retaining declared representatives for op lookup."""

    def __init__(self, dataset: PairDataset, directory: str | Path, example_count: int):
        self.dataset = dataset
        self.directory = directory
        self.active = dict(dataset.clustering_mapping(example_count))
        self.active_strings = set(self.active.values())
        # Inactive CSV values are recognized, but never added to the run's inputs.
        self.known = {text for row in dataset.records() for text in (row.source, row.target)}
        self.cluster_by_string: dict[str, int] = {}

    def _path(self, suffix: str) -> Path:
        return perfect_path(self.dataset.path.name, suffix, self.directory)

    def _error(self, suffix: str, reason: str, original: str | None = None) -> None:
        record_ground_truth_error(self._path(suffix), reason, original)

    def _blocks(self, suffix: str) -> list[list[str]]:
        path = self._path(suffix)
        try:
            content = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            self._error(suffix, f"cannot read ground truth: {error}")
            return []
        blocks = []
        for block in content.split("[ITEM]"):
            lines = block.strip("\r\n").splitlines()
            if lines:
                blocks.append(lines)
        return blocks

    @cached_property
    def clusters(self) -> list[set[int]]:
        groups: list[set[str]] = []
        for lines in self._blocks("_cluster.txt"):
            index = len(groups)
            group = set()
            for original in lines:
                if original in self.cluster_by_string:
                    if original in self.known and self.cluster_by_string[original] != index:
                        self._error("_cluster.txt", "multiple clusters; keeping first cluster", original)
                else:
                    group.add(original)
                    self.cluster_by_string[original] = index
            groups.append(group)
        for original in dict.fromkeys(self.active.values()):
            if original not in self.cluster_by_string:
                self._error("_cluster.txt", "missing string; using a separate cluster", original)
                self.cluster_by_string[original] = len(groups)
                groups.append({original})
        # Representatives can belong to another CSV variant. Retain their declared
        # membership for op lookup, but export only clusters of this run's active IDs.
        ids_by_group = {index: set() for index in range(len(groups))}
        for value_id, original in self.active.items():
            ids_by_group[self.cluster_by_string[original]].add(value_id)
        remap = {}
        clusters = []
        for index, ids in ids_by_group.items():
            if ids:
                remap[index] = len(clusters)
                clusters.append(ids)
        self.cluster_by_string = {
            original: remap[index] for original, index in self.cluster_by_string.items()
            if index in remap
        }
        return clusters

    @cached_property
    def splits(self) -> dict[str, dict[str, str]]:
        splits = {}
        for lines in self._blocks("_split.txt"):
            original = lines[0]
            if original not in self.known:
                continue
            # Held-out targets and unused reserved examples supply no component evidence.
            if original not in self.active_strings:
                continue
            components = {}
            for line in lines[1:]:
                if not line.strip():
                    continue
                label, separator, value = line.partition(":")
                if not separator or not label.strip():
                    self._error("_split.txt", f"malformed component line: {line!r}", original)
                    continue
                components[label.strip()] = value.strip()
            if original in splits and splits[original] != components:
                self._error("_split.txt", "conflicting splits; keeping first split", original)
            else:
                splits.setdefault(original, components)
        for original in dict.fromkeys(self.active.values()):
            if original not in splits:
                self._error("_split.txt", "missing string; using an empty split", original)
        return splits

    def split_for(self, query: dict[int, str], *, prompt_only: bool = False) -> dict:
        result = {}
        for value_id, original in query.items():
            components = self.splits.get(original, {})
            if prompt_only:
                spans = locate_components(original, components)
                if spans is None:
                    self._error("_split.txt", "cannot locate components for composition", original)
                result[value_id] = [f"[{span.label}]" for span in spans or ()]
            else:
                result[value_id] = dict(components)
        return result

    @cached_property
    def operators(self) -> dict[int, dict]:
        blocks = self._blocks("_op.txt")
        # Cluster membership is needed only to resolve the file's representatives.
        # Split evidence is consumed by the normal pipeline, not by this parser.
        _ = self.clusters
        result = {}
        first_original = {}
        for lines in blocks:
            original = lines[0]
            cluster = self.cluster_by_string.get(original)
            if cluster is None:
                # Undeclared representatives and groups without active IDs are unused.
                continue
            operators = {}
            try:
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    label, separator, encoded = line.partition(":")
                    if not separator or not label.strip():
                        raise ValueError(f"malformed operator line: {line!r}")
                    boundaries = encoded.split("[PARAMSPLIT]")
                    if len(boundaries) != 2:
                        raise ValueError(f"expected two [PARAMSPLIT]-separated boundaries: {line!r}")
                    operators[label.strip()] = [(parse_position(boundaries[0]),
                                                  parse_position(boundaries[1]))]
            except (ValueError, re.error) as error:
                self._error("_op.txt", f"invalid operations; ignored block: {error}", original)
                continue
            if cluster not in result:
                result[cluster] = operators
                first_original[cluster] = original
            elif result[cluster] != operators:
                self._error("_op.txt", f"conflicting operations for cluster {cluster}; "
                            f"keeping first set from {first_original[cluster]!r}", original)
        return result


def parse_position(encoded: str) -> AbsPosPosition | RegExPosPosition:
    """Parse witness-compatible positions; ``AbsPos(None)`` denotes a right EOS.

    Right endpoints are exclusive. Negative indices remain relative to the
    string length, so ``AbsPos(-1)`` is before the last character, not EOS.
    """
    name, separator, remainder = encoded.strip().partition("(")
    if not separator or not remainder.endswith(")"):
        raise ValueError(f"invalid position: {encoded}")
    arguments = remainder[:-1]
    if name in {"AbsPos", "AbsPosPosition"}:
        return AbsPosPosition(None if arguments.strip() == "None" else int(arguments.strip()))
    if name in {"RegPos", "RegExPos", "RegExPosPosition"}:
        pattern, occurrence, direction = (item.strip() for item in arguments.rsplit(",", 2))
        for token, replacement in REGEX_ESCAPES.items():
            pattern = pattern.replace(token, replacement)
        re.compile(pattern)
        return RegExPosPosition(pattern, int(occurrence), direction)
    raise ValueError(f"unknown position type: {name}")
