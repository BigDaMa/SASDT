import numpy as np
from typing import Set, List
from sklearn.metrics import completeness_score

class ClusterEvaluator:

    completeness = completeness_score

    @staticmethod
    def vmeasure(labels_true: Set[int], labels_pred: Set[int]) -> float:
        def entropy(labels):
            _, counts = np.unique(labels, return_counts=True)
            probs = counts / counts.sum()
            return -np.sum(probs * np.log(probs))

        def conditional_entropy(labels_true, labels_pred):
            labels_true = np.array(list(labels_true))
            labels_pred = np.array(list(labels_pred))
            unique_labels_true, unique_labels_pred = np.unique(labels_true), np.unique(labels_pred)
            conditional_entropy = 0.0
            for c in unique_labels_pred:
                sub_labels_true = labels_true[labels_pred == c]
                conditional_entropy += (len(sub_labels_true) / len(labels_true)) * entropy(sub_labels_true)
            return conditional_entropy

        H_C = entropy(labels_pred)
        H_K = entropy(labels_true)
        H_K_given_C = conditional_entropy(labels_true, labels_pred)
        H_C_given_K = conditional_entropy(labels_pred, labels_true)
        
        homogeneity = 1 - H_K_given_C / H_K if H_K != 0 else 1.0
        completeness = 1 - H_C_given_K / H_C if H_C != 0 else 1.0
        v_measure = 2 * (homogeneity * completeness) / (homogeneity + completeness) if (homogeneity + completeness) != 0 else 0.0
        
        return v_measure

    def evaluate_vmeasure(labels_true_set: List[Set[int]], labels_pred_set: List[Set[int]]) -> List[float]:
        v_measure_scores = []
        for labels_pred in labels_pred_set:
            scores = [ClusterEvaluator.vmeasure(labels_true, labels_pred) for labels_true in labels_true_set]
            final_v_measure = np.max(scores) if scores else 0.0
            v_measure_scores.append(final_v_measure)
        
        return v_measure_scores

    @staticmethod
    def evaluate_mean_vmeasure(labels_true_set: List[Set[int]], labels_pred_set: List[Set[int]]) -> float:
        v_measure_scores = ClusterEvaluator.evaluate_vmeasure(labels_true_set, labels_pred_set)
        return np.mean(v_measure_scores) if v_measure_scores else 0.0

    

    @staticmethod
    def evaluate_completeness(labels_true_set: List[Set[int]], labels_pred_set: List[Set[int]]) -> float:
        completeness_scores = []
        for labels_pred in labels_pred_set:
            scores = [ClusterEvaluator.completness(labels_true, list(labels_pred)) for labels_true in labels_true_set]
            final_completeness = np.max(scores) if scores else 0.0
            completeness_scores.append(final_completeness)
        
        return completeness_scores

    @staticmethod
    def evaluate_mean_completeness(labels_true_set: List[Set[int]], labels_pred_set: List[Set[int]]) -> float:
        completeness_scores = ClusterEvaluator.evaluate_completeness(labels_true_set, labels_pred_set)
        return np.mean(completeness_scores) if completeness_scores else 0.0