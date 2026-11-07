from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import pandas as pd

from gdt.data import PairDataset
from Utils.CacheUtil import cache_result


class DataAccess:
    DATA_FOLDER = "data_shuffled"
    DATA_SHUFFLED_FOLDER = "data_shuffled"
    DATA_MIX_FOLDER = "data_mix"
    DATA_STRATIFIED_FOLDER = "data_stratified"
    DATA_BACKUP_FOLDER = "data_backup"
    NUM_EXAMPLES = 5

    @staticmethod
    def get_dataset(directory: str, file: str) -> PairDataset:
        return PairDataset.from_csv(Path(directory) / file)

    @staticmethod
    def get_data_from_file(directory: str, file: str) -> pd.DataFrame:
        return DataAccess.get_dataset(directory, file).frame

    @staticmethod
    def get_examples(
        data: pd.DataFrame,
        numexamples: int | None = None,
    ) -> List[Dict[str, str]]:
        count = DataAccess.NUM_EXAMPLES if numexamples is None else numexamples
        return PairDataset(data).examples(count)

    @staticmethod
    def split_data(
        data: pd.DataFrame,
        numexamples: int = 5,
    ) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
        dataset = PairDataset(data)
        return (
            dataset.examples(numexamples),
            list(dataset.test_sources(numexamples).values()),
            list(dataset.ground_truth(numexamples).values()),
        )

    @staticmethod
    def get_data_for_clustering(
        data: pd.DataFrame,
        numexamples: int | None = None,
    ) -> List[str]:
        count = DataAccess.NUM_EXAMPLES if numexamples is None else numexamples
        return PairDataset(data).clustering_values(count)

    @staticmethod
    @cache_result()
    def map_clustered_idx_to_str(
        split_clusters: Iterable[Iterable[int]],
        data_cluster: List[str] | Mapping[int, str],
    ) -> List[Dict[int, str]]:
        return [
            {value_id: data_cluster[value_id] for value_id in cluster}
            for cluster in split_clusters
        ]
