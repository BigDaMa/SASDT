from __future__ import annotations

from gdt.timing import output_time
from gdt.data import PairDataset

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Set, Tuple, Union

import numpy as np

from DataAccess.dataaccess import DataAccess
from DataAccess.datafeature import DataFeature
from LLMUtil.LLMUtil import OLLAMAUtil, OpenAIUtil_Prompt
from Utils.CacheUtil import cache_result
from Utils.config import Config


SPARSE_CLUSTER_THRESHOLD = 3
SPARSE_CLUSTER_RATIO = 0.5
SPARSE_CLUSTER_ABS_NUM = 2

# Historical module constants retained for external notebooks.
__SPARSE_CLUSTER_THRESHOLD__ = SPARSE_CLUSTER_THRESHOLD
__SPARSE_CLUSTER_RATIO__ = SPARSE_CLUSTER_RATIO
__SPARSE_CLUSTER_ABS_NUM__ = SPARSE_CLUSTER_ABS_NUM


@dataclass(slots=True)
class _PreparedData:
    source_count: int
    sources: dict[int, str]
    example_targets: list[str]
    output_ids: list[int]


class CollapseCluster:
    _embedding_model: Any = None
    _embedding_model_name = "local-qwen-emb"

    @staticmethod
    def class_based_collapse(text: str) -> str:
        collapsed = (
            unicodedata.normalize("NFKD", str(text))
            .encode("ascii", "ignore")
            .decode("ascii")
            .strip()
        )
        collapsed = re.sub(r"\s+", " ", collapsed)
        collapsed = re.sub(r"[A-Z]", "A", collapsed)
        collapsed = re.sub(r"[a-z]", "a", collapsed)
        collapsed = re.sub(r"\d", "0", collapsed)
        return re.sub(r"(A+)|(a+)|(0+)", lambda match: match.group(0)[0], collapsed)

    @classmethod
    def get_embedding(cls, text: str) -> List[float]:
        if cls._embedding_model is None:
            from Utils.ModelSelect import ModelSelect

            cls._embedding_model = ModelSelect.get_clustering_model_by_name(
                cls._embedding_model_name
            )
        return cls._embedding_model.encode(text)

    @staticmethod
    def collapse_repeated_segments(value: str) -> str:
        # Adjacent structural tokens are redundant after character-class collapse.
        return re.sub(r"(?<!\S)(\S+)(?:\s+\1)+(?!\S)", r"\1", value)

    @staticmethod
    def get_rare_nonword(
        data_feature: Dict[int, Dict[str, Any]],
        keep_nonword_num: int = 2,
    ) -> List[str]:
        counts = Counter(
            character
            for feature in data_feature.values()
            for character in re.findall(
                r"[^\w\s]", feature.get("collapsed_relax", feature["collapsed"])
            )
        )
        if len(counts) <= keep_nonword_num:
            return []
        minimum = min(counts.values())
        return sorted(character for character, count in counts.items() if count == minimum)

    @staticmethod
    def relax_collapse(
        data_feature: Dict[int, Dict[str, Any]],
        index_list: List[int],
        rare_nonword: List[str],
    ) -> Dict[int, Dict[str, Any]]:
        relaxed: Dict[int, Dict[str, Any]] = {}
        selected = set(index_list)
        for value_id, feature in data_feature.items():
            if value_id not in selected:
                continue
            text = feature.get("collapsed_relax", feature["collapsed"])
            for character in rare_nonword:
                text = text.replace(character, "")
            text = re.sub(r"(?<=[a-zA-Z0-9])A", "a", text)
            text = re.sub(r"\s+", " ", text.strip())
            text = re.sub(r"(A+)|(a+)|(0+)", lambda match: match.group(0)[0], text)
            updated = feature.copy()
            updated["collapsed_relax"] = CollapseCluster.collapse_repeated_segments(text)
            relaxed[value_id] = updated
        return relaxed

    @staticmethod
    def collapse_cluster(data_feature: Dict[int, Dict[str, Any]]) -> List[Set[int]]:
        groups: dict[str, set[int]] = defaultdict(set)
        for value_id, feature in data_feature.items():
            groups[feature.get("collapsed_relax", feature["collapsed"])].add(value_id)
        return list(groups.values())

    @staticmethod
    def group_indices_by_label(labels: Iterable[int], cluster: List[int]) -> List[Set[int]]:
        groups: dict[int, set[int]] = defaultdict(set)
        for value_id, label in zip(cluster, labels):
            # DBSCAN's -1 denotes independent noise, not one semantic cluster.
            group_label = label if label != -1 else -(value_id + 1_000_000)
            groups[int(group_label)].add(value_id)
        return list(groups.values())

    @staticmethod
    def get_knee_point(vectors: List[List[float]], k: int = 10) -> float:
        if len(vectors) < 2:
            return 0.5
        candidates = np.linspace(0.1, 1.0, max(2, k))
        cluster_counts = []
        min_samples = min(SPARSE_CLUSTER_ABS_NUM, len(vectors))
        for epsilon in candidates:
            labels = _dbscan(
                eps=float(epsilon),
                min_samples=min_samples,
                metric="cosine",
            ).fit_predict(vectors)
            cluster_counts.append(len(set(labels)))
        changes = np.abs(np.diff(cluster_counts))
        return float(candidates[int(np.argmax(changes)) + 1]) if changes.any() else 0.5

    @staticmethod
    def split_cluster_by_embedding(
        cluster: List[int],
        data_feature: Dict[int, Dict[str, Any]],
    ) -> List[Set[int]]:
        if len(cluster) < 2:
            return [set(cluster)]
        embeddings = [list(data_feature[value_id]["embedding"]) for value_id in cluster]
        epsilon = CollapseCluster.get_knee_point(embeddings)
        min_samples = min(SPARSE_CLUSTER_THRESHOLD, len(cluster))
        labels = _dbscan(
            eps=epsilon,
            min_samples=min_samples,
            metric="cosine",
        ).fit_predict(embeddings)
        return CollapseCluster.group_indices_by_label(labels, cluster)

    @staticmethod
    def split_cluster_by_composition(
        cluster: List[int],
        data_feature: Dict[int, Dict[str, Any]],
    ) -> List[Set[int]]:
        groups: dict[str, set[int]] = defaultdict(set)
        for value_id in cluster:
            groups[data_feature[value_id].get("composition", "")].add(value_id)
        return list(groups.values())

    @staticmethod
    def merge_sparse_clusters(
        clusters: List[Set[int]],
        data_feature: Dict[int, Dict[str, Any]],
    ) -> List[Set[int]]:
        dense = [set(cluster) for cluster in clusters if len(cluster) >= SPARSE_CLUSTER_THRESHOLD]
        sparse = [set(cluster) for cluster in clusters if len(cluster) < SPARSE_CLUSTER_THRESHOLD]
        if not dense:
            return sparse

        centroids = [
            np.mean([data_feature[value_id]["embedding"] for value_id in cluster], axis=0)
            for cluster in dense
        ]
        for sparse_cluster in sparse:
            sparse_centroid = np.mean(
                [data_feature[value_id]["embedding"] for value_id in sparse_cluster],
                axis=0,
            )
            nearest = int(np.argmin(_cosine_distances([sparse_centroid], centroids)[0]))
            dense[nearest].update(sparse_cluster)
            centroids[nearest] = np.mean(
                [data_feature[value_id]["embedding"] for value_id in dense[nearest]],
                axis=0,
            )
        return dense

    @staticmethod
    def get_labeled_clusters(clusters: List[Set[int]]) -> Tuple[int, List[int]]:
        labels = {
            value_id: label
            for label, cluster in enumerate(clusters)
            for value_id in cluster
        }
        return len(clusters), [labels[value_id] for value_id in sorted(labels)]

    @staticmethod
    @cache_result()
    def run_cluster_emb(file: str) -> List[Set[int]]:
        prepared = CollapseCluster._prepare(file)
        features = CollapseCluster._build_features(prepared.sources, embeddings=True)
        clusters = CollapseCluster.split_cluster_by_embedding(
            list(features), features
        )
        return CollapseCluster._with_outputs(clusters, prepared.output_ids)

    @staticmethod
    @cache_result()
    def run_cluster_emb_collapse(file: str) -> List[Set[int]]:
        prepared = CollapseCluster._prepare(file)
        features = CollapseCluster._build_features(prepared.sources, embeddings=True)
        embedding_clusters = CollapseCluster.split_cluster_by_embedding(
            list(features), features
        )
        result: list[set[int]] = []
        for cluster in embedding_clusters:
            collapsed = CollapseCluster.collapse_cluster(
                {value_id: features[value_id] for value_id in cluster}
            )
            result.extend(CollapseCluster._merge_nearest(collapsed, features))
        return CollapseCluster._with_outputs(result, prepared.output_ids)

    @staticmethod
    @cache_result()
    def run_cluster_collapse(file: str) -> List[Set[int]]:
        prepared = CollapseCluster._prepare(file)
        features = CollapseCluster._build_features(prepared.sources, embeddings=False)
        return CollapseCluster._with_outputs(
            CollapseCluster.collapse_cluster(features), prepared.output_ids
        )

    @staticmethod
    @cache_result()
    def run_cluster_collapse_emb(file: str) -> List[Set[int]]:
        prepared = CollapseCluster._prepare(file)
        features = CollapseCluster._build_features(prepared.sources, embeddings=True)
        clusters = CollapseCluster.collapse_cluster(features)
        sparse = [cluster for cluster in clusters if len(cluster) < SPARSE_CLUSTER_THRESHOLD]
        while (
            len(sparse) > SPARSE_CLUSTER_ABS_NUM
            and clusters
            and len(sparse) / len(clusters) >= SPARSE_CLUSTER_RATIO
        ):
            rare = CollapseCluster.get_rare_nonword(features)
            if not rare:
                break
            for value_id, feature in CollapseCluster.relax_collapse(
                features, list(features), rare
            ).items():
                features[value_id] = feature
            next_clusters = CollapseCluster.collapse_cluster(features)
            if next_clusters == clusters:
                break
            clusters = next_clusters
            sparse = [cluster for cluster in clusters if len(cluster) < SPARSE_CLUSTER_THRESHOLD]

        split: list[set[int]] = []
        for cluster in clusters:
            if len(cluster) < SPARSE_CLUSTER_THRESHOLD:
                split.append(set(cluster))
            else:
                split.extend(
                    CollapseCluster.split_cluster_by_embedding(list(cluster), features)
                )
        merged = CollapseCluster.merge_sparse_clusters(split, features)
        return CollapseCluster._with_outputs(merged, prepared.output_ids)

    @staticmethod
    @cache_result(restore_features=True)
    def run_cluster_collapse_embsplit(file: str) -> List[Set[int]]:
        prepared = CollapseCluster._prepare(file)
        features = CollapseCluster._build_features(prepared.sources, embeddings=True)
        DataFeature.data_features.update(features)
        split: list[set[int]] = []
        for cluster in CollapseCluster.collapse_cluster(features):
            split.extend(
                CollapseCluster.split_cluster_by_embedding(list(cluster), features)
                if len(cluster) > 1
                else [set(cluster)]
            )
        return CollapseCluster._with_outputs(split, prepared.output_ids)

    @staticmethod
    @cache_result(restore_features=True)
    def run_cluster_collapse_promptsplit(
        file: str,
    ) -> Union[List[Set[int]], Dict[int, Dict[str, str]]]:
        return CollapseCluster._run_prompt_split(file)

    @staticmethod
    @cache_result(restore_features=True)
    def run_cluster_collapse_promptsplit_gpt(
        file: str,
    ) -> Union[List[Set[int]], Dict[int, Dict[str, str]]]:
        return CollapseCluster._run_prompt_split(file)

    @staticmethod
    def _prepare(file: str) -> _PreparedData:
        frame = DataAccess.get_data_from_file(DataAccess.DATA_FOLDER, file)
        dataset = PairDataset(frame)
        source_count = len(frame)
        return _PreparedData(
            source_count=source_count,
            sources=dataset.active_sources(DataAccess.NUM_EXAMPLES),
            example_targets=list(dataset.example_targets(DataAccess.NUM_EXAMPLES).values()),
            output_ids=[source_count + index for index in range(DataAccess.NUM_EXAMPLES)],
        )

    @staticmethod
    def _build_features(
        sources: Dict[int, str],
        *,
        embeddings: bool,
    ) -> Dict[int, Dict[str, Any]]:
        features: Dict[int, Dict[str, Any]] = {}
        for value_id, text in sources.items():
            feature: Dict[str, Any] = {
                "text": text,
                "collapsed": CollapseCluster.class_based_collapse(text),
                "composition": "",
            }
            if embeddings:
                feature["embedding"] = CollapseCluster.get_embedding(text)
            features[value_id] = feature
        return features

    @staticmethod
    def _with_outputs(
        clusters: Iterable[Set[int]],
        output_ids: List[int],
    ) -> List[Set[int]]:
        result = [set(cluster) for cluster in clusters if cluster]
        result.append(set(output_ids))
        return result

    @staticmethod
    def _merge_nearest(
        clusters: List[Set[int]],
        features: Dict[int, Dict[str, Any]],
    ) -> List[Set[int]]:
        clusters = [set(cluster) for cluster in clusters]
        while len(clusters) > 1:
            sparse_index = next(
                (
                    index
                    for index, cluster in enumerate(clusters)
                    if len(cluster) < SPARSE_CLUSTER_THRESHOLD
                ),
                None,
            )
            if sparse_index is None:
                break
            sparse = clusters[sparse_index]
            sparse_centroid = np.mean(
                [features[value_id]["embedding"] for value_id in sparse], axis=0
            )
            candidates = [index for index in range(len(clusters)) if index != sparse_index]
            nearest = min(
                candidates,
                key=lambda index: _cosine_distances(
                    [sparse_centroid],
                    [
                        np.mean(
                            [features[value_id]["embedding"] for value_id in clusters[index]],
                            axis=0,
                        )
                    ],
                )[0][0],
            )
            clusters[nearest].update(sparse)
            clusters.pop(sparse_index)
        return clusters

    @staticmethod
    def _run_prompt_split(
        file: str,
    ) -> List[Set[int]]:
        if DataAccess.NUM_EXAMPLES < 1:
            raise ValueError("prompt-based clustering requires at least one example")
        DataFeature.clear()
        prepared = CollapseCluster._prepare(file)
        source_features = CollapseCluster._build_features(
            prepared.sources, embeddings=False
        )
        DataFeature.data_features.update(source_features)
        output_features = {
            value_id: {
                "text": text,
                "collapsed": CollapseCluster.class_based_collapse(text),
                "composition": "",
            }
            for value_id, text in zip(prepared.output_ids, prepared.example_targets)
        }

        example_source_ids = list(range(min(DataAccess.NUM_EXAMPLES, prepared.source_count)))
        examples = {
            **{value_id: source_features[value_id]["text"] for value_id in example_source_ids},
            **{value_id: output_features[value_id]["text"] for value_id in prepared.output_ids},
        }
        split = OpenAIUtil_Prompt.get_split_prompt(
            examples,
            file,
            example_source_ids,
            prepared.output_ids,
            prompt="label",
            always_split_further=False,
        )
        for value_id, labels in split.items():
            target = source_features.get(value_id) or output_features.get(value_id)
            if target is not None:
                target["composition"] = CollapseCluster._labels_to_composition(labels)

        needs_semantic_split = not all(
            _is_digit_only(text) for text in prepared.sources.values()
        )
        remaining_ids = [value_id for value_id in source_features if value_id not in split]
        if needs_semantic_split and remaining_ids:
            example_splits = [
                (source_features[value_id]["text"], source_features[value_id]["composition"])
                for value_id in example_source_ids
                if source_features[value_id]["composition"]
            ]
            semantic_types = sorted(
                {
                    match.group(0)
                    for _text, composition in example_splits
                    for match in re.finditer(r"\[[^\]]+\]", composition)
                }
            )
            remaining = {
                value_id: source_features[value_id]["text"] for value_id in remaining_ids
            }
            from gdt.models import is_gpt_model

            if is_gpt_model(Config.model):
                labels_by_id = OpenAIUtil_Prompt.get_labels_for_types(
                    remaining, file, semantic_types,
                )
            else:
                # Pass the full pool so each prompt can select its own examples.
                # Held-out targets never enter this pool.
                example_pool = example_splits + [
                    (feature["text"], feature["composition"])
                    for feature in output_features.values()
                    if feature["composition"]
                ]
                slm_types = sorted({
                    label for _text, composition in example_pool
                    for label in re.findall(r"\[[^\]]+\]", composition)
                })
                OLLAMAUtil.set_model(Config.model)
                OLLAMAUtil.set_mode(Config.mode)
                labels_by_id = OLLAMAUtil.get_labels_for_types(
                    remaining, file, slm_types, example_pool,
                )
            for value_id in remaining_ids:
                source_features[value_id]["composition"] = (
                    CollapseCluster._labels_to_composition(labels_by_id[value_id])
                )

        structural_clusters = CollapseCluster.collapse_cluster(source_features)
        clusters: list[set[int]] = []
        for cluster in structural_clusters:
            if needs_semantic_split and len(cluster) > 1:
                clusters.extend(
                    CollapseCluster.split_cluster_by_composition(
                        list(cluster), source_features
                    )
                )
            else:
                clusters.append(set(cluster))
        DataFeature.data_features.update(output_features)
        result = CollapseCluster._with_outputs(clusters, prepared.output_ids)
        if Config.verbose:
            with output_time():
                CollapseCluster._print_clusters(result)
        return result

    @staticmethod
    def _labels_to_composition(labels: Iterable[str]) -> str:
        normalized = []
        for label in labels:
            value = str(label).strip().strip("[]")
            value = re.sub(r"\s+", "_", value)
            if value:
                normalized.append(f"[{value}]")
        return "".join(normalized)

    @staticmethod
    def _response_to_composition(response: str) -> str:
        labels = re.findall(r"\[([^\]]+)\]", response)
        if not labels:
            labels = [
                segment.strip()
                for segment in re.split(r"[\n,]+", response)
                if segment.strip()
            ]
        return CollapseCluster._labels_to_composition(labels)

    @staticmethod
    def _print_clusters(clusters: List[Set[int]]) -> None:
        for cluster in clusters:
            with output_time():
                print("===== CLUSTER =====")
            for value_id in sorted(cluster):
                feature = DataFeature.data_features.get(value_id, {})
                with output_time():
                    print(f"{value_id}: {feature.get('composition', '')}")


def _is_digit_only(text: str) -> bool:
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", text)
    return bool(cleaned) and cleaned.isdigit()


def _dbscan(**kwargs):
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as error:
        raise RuntimeError(
            "embedding clustering requires scikit-learn; install project dependencies"
        ) from error
    return DBSCAN(**kwargs)


def _cosine_distances(left, right):
    try:
        from sklearn.metrics.pairwise import cosine_distances
    except ImportError as error:
        raise RuntimeError(
            "embedding clustering requires scikit-learn; install project dependencies"
        ) from error
    return cosine_distances(left, right)
