from __future__ import annotations

from typing import Callable, List, Literal

from Utils.CacheUtil import perfect_cluster_override


class ClusterMeta:
    COLLAPSE_STRATEGIES = {
        "collapse+emb": "run_cluster_collapse_emb",
        "emb+collapse": "run_cluster_emb_collapse",
        "emb": "run_cluster_emb",
        "collapse": "run_cluster_collapse",
        "collapse+embsplit": "run_cluster_collapse_embsplit",
        "collapse+prompt_label": "run_cluster_collapse_promptsplit",
        "collapse+prompt_label_gpt": "run_cluster_collapse_promptsplit_gpt",
    }

    @staticmethod
    @perfect_cluster_override()
    def run_cluster(
        file: str,
        method: Literal["collapse", "normal"] = "collapse",
        clusteralg: str = "DBSCAN",
        config: str = "collapse+emb",
    ) -> List[set[int]]:
        if method == "normal":
            from Clustering.cluster import Cluster
            from DataAccess.dataaccess import DataAccess

            clusters = Cluster.runCluster(file, clusteralg)
            row_count = len(DataAccess.get_data_from_file(DataAccess.DATA_FOLDER, file))
            output_ids = {
                row_count + index for index in range(DataAccess.NUM_EXAMPLES)
            }
            return [
                restricted
                for cluster in clusters
                if (restricted := set(cluster) - output_ids)
            ] + [output_ids]
        if method != "collapse":
            raise ValueError(f"unknown clustering method: {method}")

        from Clustering.collapsecluster import CollapseCluster

        method_name = ClusterMeta.COLLAPSE_STRATEGIES.get(config)
        if method_name is None:
            choices = ", ".join(sorted(ClusterMeta.COLLAPSE_STRATEGIES))
            raise ValueError(f"unknown collapse configuration {config!r}; choose one of: {choices}")
        strategy: Callable[[str], List[set[int]]] = getattr(CollapseCluster, method_name)
        return strategy(file)
