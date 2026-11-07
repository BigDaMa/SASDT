from __future__ import annotations

from gdt.timing import output_time

from collections import defaultdict
import itertools
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Set

import pandas as pd

from DataAccess.dataaccess import DataAccess
from Utils.config import Config
from Utils.IterPrintUtil import IterPrintUtil
from gdt.data import PairDataset
from gdt.ground_truth import GroundTruth


class ProcessTruth:
    @staticmethod
    def process(truthfile: str | Path, returntype: str = "set"):
        groups: list[set[int]] = []
        path = Path(truthfile)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                values = [value.strip() for value in line.split(",") if value.strip()]
                if not values:
                    continue
                try:
                    groups.append({int(value) for value in values})
                except ValueError as error:
                    raise ValueError(f"invalid ground-truth ID in {path}:{line_number}") from error
        if returntype == "set":
            return groups
        if returntype == "list":
            return [sorted(group) for group in groups]
        raise ValueError("returntype must be 'set' or 'list'")

    @staticmethod
    def process_labeled(datafile: str, numexamples: int | None = None) -> List[int]:
        count = DataAccess.NUM_EXAMPLES if numexamples is None else numexamples
        cluster_path = Path("data_cluster") / datafile
        data_path = Path(DataAccess.DATA_FOLDER) / datafile
        clustered = pd.read_csv(cluster_path, header=None)
        data = DataAccess.get_data_from_file(str(data_path.parent), data_path.name)
        values = list(data.iloc[:, 0]) + list(data.iloc[:count, 1])
        # Duplicate strings are legal; consume positions in their original order.
        positions: dict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(values):
            positions[str(value)].append(index)
        occurrence: dict[str, int] = defaultdict(int)
        sort_order = []
        for value in clustered.iloc[:, 0].astype(str):
            offset = occurrence[value]
            sort_order.append(
                positions[value][offset]
                if offset < len(positions[value])
                else len(values)
            )
            occurrence[value] += 1
        clustered = clustered.assign(_sort_order=sort_order).sort_values("_sort_order")
        return clustered.iloc[: len(values), 1].dropna().astype(int).tolist()

    @staticmethod
    def process_labeled_perfect(
        datafile: str,
        numexamples: int = 0,
        *,
        include_output_cluster: bool = False,
    ) -> List[Set[int]]:
        dataset = PairDataset.from_csv(Path(DataAccess.DATA_FOLDER) / datafile)
        truth = GroundTruth(dataset, Config.perfect_directory, DataAccess.NUM_EXAMPLES)
        clusters = [
            {value for value in cluster if value >= numexamples
             and (include_output_cluster or value < dataset.row_count)}
            for cluster in truth.clusters
        ]
        return [cluster for cluster in clusters if cluster]

    @staticmethod
    def group_indices_by_label(
        labels: Iterable[int],
        *,
        one_based: bool = True,
    ) -> List[Set[int]]:
        groups: dict[int, set[int]] = defaultdict(set)
        offset = 1 if one_based else 0
        for index, label in enumerate(labels):
            groups[int(label)].add(index + offset)
        return list(groups.values())

    @staticmethod
    def list_duplicates_of(sequence, item) -> list[int]:
        return [index for index, value in enumerate(sequence) if value == item]

    @staticmethod
    def evaluate(labelindex: List[Set[int]], truthtable: List[Set[int]]) -> dict:
        if not labelindex or not truthtable:
            return {"mean": 0.0}
        result = {}
        for group_number, predicted in enumerate(labelindex, start=1):
            scores = [
                len(predicted & truth) / len(truth) if truth else 0.0
                for truth in truthtable
            ]
            result[str(group_number)] = max(scores, default=0.0)
        result["mean"] = mean(result.values()) if result else 0.0
        return result

    @staticmethod
    def processTruthTable(truthtable: List[Set[int]]) -> List[Set[str]]:
        return [
            {f"{left}+{right}" for left, right in itertools.combinations(sorted(group), 2)}
            for group in truthtable
        ]

    processLabelIndex = processTruthTable

    @staticmethod
    def evaluate_pairwise(labelindex: List[Set[int]], truthtable: List[Set[int]]) -> dict:
        if not labelindex or not truthtable:
            return {"mean": {"p": 0.0, "r": 0.0}}
        truth_pairs = [
            set(itertools.combinations(sorted(group), 2)) for group in truthtable
        ]
        result = {}
        for group_number, predicted in enumerate(labelindex, start=1):
            predicted_pairs = set(itertools.combinations(sorted(predicted), 2))
            best_precision = 0.0
            best_recall = 0.0
            best_overlap = -1
            for expected_pairs in truth_pairs:
                overlap = len(predicted_pairs & expected_pairs)
                precision = overlap / len(predicted_pairs) if predicted_pairs else 0.0
                recall = overlap / len(expected_pairs) if expected_pairs else 0.0
                if overlap > best_overlap or (
                    overlap == best_overlap and precision + recall > best_precision + best_recall
                ):
                    best_overlap = overlap
                    best_precision = precision
                    best_recall = recall
            result[str(group_number)] = {"p": best_precision, "r": best_recall}
        rows = list(result.values())
        result["mean"] = {
            "p": mean(row["p"] for row in rows),
            "r": mean(row["r"] for row in rows),
        }
        return result

    @staticmethod
    def printres(dataset: str, featureset: List[str], result: dict) -> None:
        with output_time():
            print(f"Precision of {dataset}")
            print(f"Using feature {', '.join(featureset)}")
            print("Mean:")
            IterPrintUtil.print(result.get("mean", 0.0))

    @staticmethod
    def saveresandprint(
        fout: str,
        dataset: str,
        featureset: List[str],
        result: dict,
        initial: bool,
    ) -> None:
        with output_time():
            mode = "w" if initial else "a"
            with open(fout, mode, encoding="utf-8") as handle:
                handle.write(f"Precision of {dataset}\n")
                handle.write(f"Using feature {', '.join(featureset)}\n")
                handle.write(f"Mean precision: {result.get('mean', 0.0)}\n")
            ProcessTruth.printres(dataset, featureset, result)
