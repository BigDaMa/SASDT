from __future__ import annotations

from gdt.timing import output_time

from itertools import combinations
from pathlib import Path
import time
from typing import List

import pandas as pd
from sklearn.cluster import DBSCAN

from Clustering.feature import Feature
from Clustering.processtruth import ProcessTruth
from DataAccess.dataaccess import DataAccess
from Utils.config import Config
from Utils.IterPrintUtil import IterPrintUtil


ACTIVE_FEATURE_LIST = Path(__file__).with_name("feature_list.txt")
FEATURE_DIRECTORY = Path("data_feature")
TRUTH_DIRECTORY = Path("truth")
NOMINAL_FEATURES = {"TypeNonAlnumCharFeature"}


class Cluster:
    @staticmethod
    def getUniqueFeatures(clusteralg: str = "DBSCAN") -> List[str]:
        with ACTIVE_FEATURE_LIST.open("r", encoding="utf-8") as handle:
            features = [line.strip() for line in handle if line.strip()]
        if clusteralg not in {"DenseClus", "all"}:
            features = [feature for feature in features if feature not in NOMINAL_FEATURES]
        unknown = set(features) - set(Feature.AVAILABLE_FEATURES)
        if unknown:
            raise ValueError(f"unknown features in {ACTIVE_FEATURE_LIST}: {sorted(unknown)}")
        return features

    @staticmethod
    def getFeatureList(clusteralg: str = "DBSCAN") -> List[List[str]]:
        features = Cluster.getUniqueFeatures(clusteralg)
        if clusteralg == "DenseClus":
            return [features]
        return [
            list(group)
            for size in range(1, len(features) + 1)
            for group in combinations(features, size)
        ]

    @staticmethod
    def getFeatureIndexDict(clusteralg: str = "DBSCAN") -> dict[str, int]:
        return {
            feature: index
            for index, feature in enumerate(Cluster.getUniqueFeatures(clusteralg))
        }

    @staticmethod
    def featureCleanAndSave(file: str, df: pd.DataFrame | None = None) -> None:
        path = Path(file)
        data = pd.read_csv(path, header=None) if df is None else df.copy()
        features = Cluster.getUniqueFeatures()
        if "LengthFeature" in features:
            column = features.index("LengthFeature")
            data[column] = Feature.min_max_normalize(data[column].tolist())
        with output_time():
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(path, header=False, index=False)

    @staticmethod
    def getFeaturesAndCache(fin: str, fout: str) -> None:
        source = pd.read_csv(fin, header=None, usecols=[0])
        features = Cluster.getUniqueFeatures("all")
        rows = [Feature.get_features(value, features) for value in source.iloc[:, 0]]
        frame = pd.DataFrame(rows, columns=features)
        if "LengthFeature" in frame:
            frame["LengthFeature"] = Feature.min_max_normalize(
                frame["LengthFeature"].tolist()
            )
        output = Path(fout)
        with output_time():
            output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output, index=False)

    @staticmethod
    def loadFeaturesAsDF(fin: str) -> pd.DataFrame:
        frame = pd.read_csv(fin, header=0)
        features = Cluster.getUniqueFeatures()
        missing = set(features) - set(frame.columns)
        if missing:
            raise ValueError(f"feature file {fin} is missing columns: {sorted(missing)}")
        return frame[features]

    @staticmethod
    def runCluster(file: str, clusteralg: str = "DBSCAN") -> List[set[int]]:
        started = time.perf_counter()
        dataset = DataAccess.get_dataset(DataAccess.DATA_FOLDER, file)
        values = dataset.clustering_mapping(DataAccess.NUM_EXAMPLES)
        value_ids = list(values)
        feature_path = FEATURE_DIRECTORY / (
            f"{Path(file).stem}_reserved5_n{DataAccess.NUM_EXAMPLES}_features.csv"
        )
        if not feature_path.is_file():
            Feature.get_feature_for_strs(list(values.values()), feature_path.name, reset=True)

        feature_frame = Cluster.loadFeaturesAsDF(str(feature_path))
        clusterer = Cluster._build_clusterer(clusteralg)
        if clusteralg == "DenseClus":
            clusterer.fit(feature_frame)
            labels = clusterer.score()
        else:
            labels = clusterer.fit_predict(feature_frame.values)
        clusters = [
            {value_ids[index] for index in cluster}
            for cluster in ProcessTruth.group_indices_by_label(labels, one_based=False)
        ]

        truth_path = TRUTH_DIRECTORY / f"{Path(file).stem}_processed.csv"
        with output_time():
            if truth_path.is_file() and Config.verbose:
                metrics = ProcessTruth.evaluate_pairwise(
                    clusters, ProcessTruth.process(truth_path)
                )
                with output_time():
                    print(f"{file}: pairwise metrics:")
                with output_time():
                    IterPrintUtil.print(metrics["mean"])
            with output_time():
                if Config.verbose:
                    print(f"{file}: clustered in {time.perf_counter() - started:.2f}s")
        return clusters

    @staticmethod
    def _build_clusterer(clusteralg: str):
        if clusteralg == "DBSCAN":
            return DBSCAN(eps=0.1, min_samples=4)
        if clusteralg == "HDBSCAN":
            try:
                from sklearn.cluster import HDBSCAN
            except ImportError as error:
                raise RuntimeError(
                    "HDBSCAN requires a scikit-learn build that provides it"
                ) from error
            return HDBSCAN(min_cluster_size=3, min_samples=5)
        if clusteralg == "DenseClus":
            try:
                from denseclus import DenseClus
            except ImportError as error:
                raise RuntimeError("DenseClus is not installed") from error
            return DenseClus(
                umap_combine_method="intersection_union_mapper",
                n_neighbors=3,
                min_samples=3,
                min_cluster_size=5,
            )
        raise ValueError(f"unknown clustering algorithm: {clusteralg}")


ActiveFeatureList = str(ACTIVE_FEATURE_LIST)
