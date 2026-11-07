from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from Clustering.processtruth import ProcessTruth
from DataAccess.dataaccess import DataAccess


class MiniLMCluster:
    numeric_feature_count = 0

    @classmethod
    def custom_distance(cls, left, right, alpha: float = 0.5) -> float:
        numeric_left = left[: cls.numeric_feature_count] if cls.numeric_feature_count else None
        numeric_right = right[: cls.numeric_feature_count] if cls.numeric_feature_count else None
        vector_left = left[cls.numeric_feature_count :]
        vector_right = right[cls.numeric_feature_count :]
        vector_distance = cosine_distances([vector_left], [vector_right])[0][0]
        if numeric_left is None:
            return float(vector_distance)
        numeric_distance = euclidean_distances([numeric_left], [numeric_right])[0][0]
        return float(alpha * vector_distance + (1 - alpha) * numeric_distance)

    @staticmethod
    def get_text_embeddings(model, data: pd.DataFrame) -> List[List[float]]:
        return [model.encode(str(text)) for text in data.iloc[:, 0]]

    @staticmethod
    def cluster(clusterer, data, truth: List[Set[int]]) -> Dict[str, object]:
        labels = clusterer.fit(data).labels_
        predicted = ProcessTruth.group_indices_by_label(labels, one_based=False)
        result = ProcessTruth.evaluate_pairwise(predicted, truth)
        for key, value in result.items():
            if key == "mean":
                continue
            precision, recall = value["p"], value["r"]
            value["f1"] = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        result["mean"]["f1"] = float(
            np.mean([value["f1"] for key, value in result.items() if key != "mean"])
        ) if len(result) > 1 else 0.0
        return result

    @classmethod
    def run_directory(
        cls,
        data_directory: str | Path,
        truth_directory: str | Path,
        *,
        feature_directory: str | Path | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> dict[str, Dict[str, object]]:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        results = {}
        for path in sorted(Path(data_directory).glob("*.csv")):
            frame = DataAccess.get_data_from_file(str(path.parent), path.name)
            embeddings = cls.get_text_embeddings(model, frame)
            combined = embeddings
            if feature_directory:
                feature_path = Path(feature_directory) / f"{path.stem}_feature.csv"
                numeric = pd.read_csv(feature_path, header=None).iloc[:, :-1].values.tolist()
                cls.numeric_feature_count = len(numeric[0]) if numeric else 0
                combined = np.hstack((numeric, embeddings))
            truth = ProcessTruth.process(Path(truth_directory) / f"{path.stem}_processed.csv")
            clusterer = DBSCAN(eps=0.4, min_samples=8, metric=cls.custom_distance)
            results[path.name] = cls.cluster(clusterer, combined, truth)
        return results
