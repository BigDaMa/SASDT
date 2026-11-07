from __future__ import annotations

import itertools
from typing import Iterable, List, Set

import numpy as np


class ClusterEvaluator:
    @staticmethod
    def completeness(labels_true: Iterable[int], labels_pred: Iterable[int]) -> float:
        from sklearn.metrics import completeness_score

        return float(completeness_score(list(labels_true), list(labels_pred)))

    @staticmethod
    def vmeasure(labels_true: Iterable[int], labels_pred: Iterable[int]) -> float:
        from sklearn.metrics import v_measure_score

        return float(v_measure_score(list(labels_true), list(labels_pred)))

    @staticmethod
    def evaluate_vmeasure(
        labels_true_set: List[Set[int]],
        labels_pred_set: List[Set[int]],
    ) -> List[float]:
        true_labels, predicted_labels = ClusterEvaluator._label_vectors(
            labels_true_set, labels_pred_set
        )
        return [ClusterEvaluator.vmeasure(true_labels, predicted_labels)]

    @staticmethod
    def evaluate_mean_vmeasure(
        labels_true_set: List[Set[int]],
        labels_pred_set: List[Set[int]],
    ) -> float:
        scores = ClusterEvaluator.evaluate_vmeasure(labels_true_set, labels_pred_set)
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def evaluate_completeness(
        labels_true_set: List[Set[int]],
        labels_pred_set: List[Set[int]],
    ) -> List[float]:
        true_labels, predicted_labels = ClusterEvaluator._label_vectors(
            labels_true_set, labels_pred_set
        )
        return [ClusterEvaluator.completeness(true_labels, predicted_labels)]

    @staticmethod
    def evaluate_mean_completeness(
        labels_true_set: List[Set[int]],
        labels_pred_set: List[Set[int]],
    ) -> float:
        scores = ClusterEvaluator.evaluate_completeness(labels_true_set, labels_pred_set)
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def evaluate_pairwise_f1(
        labels_true_set: List[Set[int]],
        labels_pred_set: List[Set[int]],
    ) -> tuple[float, float, float]:
        true_pairs = ClusterEvaluator._pairs(labels_true_set)
        predicted_pairs = ClusterEvaluator._pairs(labels_pred_set)
        true_positive = len(true_pairs & predicted_pairs)
        precision = (
            true_positive / len(predicted_pairs) if predicted_pairs else 0.0
        )
        recall = true_positive / len(true_pairs) if true_pairs else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return precision, recall, f1

    @staticmethod
    def _pairs(groups: Iterable[Set[int]]) -> set[tuple[int, int]]:
        return {
            pair
            for group in groups
            for pair in itertools.combinations(sorted(group), 2)
        }

    @staticmethod
    def _label_vectors(
        truth: List[Set[int]],
        predicted: List[Set[int]],
    ) -> tuple[list[int], list[int]]:
        universe = sorted(set().union(*truth, *predicted)) if truth or predicted else []
        truth_map = {
            value_id: label for label, group in enumerate(truth) for value_id in group
        }
        predicted_map = {
            value_id: label for label, group in enumerate(predicted) for value_id in group
        }
        missing_truth = len(truth)
        missing_predicted = len(predicted)
        return (
            [
                truth_map.get(value_id, missing_truth + index)
                for index, value_id in enumerate(universe)
            ],
            [
                predicted_map.get(value_id, missing_predicted + index)
                for index, value_id in enumerate(universe)
            ],
        )
