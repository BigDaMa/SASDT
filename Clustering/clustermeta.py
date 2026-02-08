from typing import List, Literal
from Clustering.cluster import Cluster
from Clustering.collapsecluster import CollapseCluster
from Utils.CacheUtil import perfect_cluster_override

class ClusterMeta:

    @staticmethod
    @perfect_cluster_override()
    def run_cluster(file: str, method: Literal["collapse", "normal"] = "collapse", clusteralg: str = 'DBSCAN', config: str = "collapse+emb") -> List[List[int]]:
        if method == "collapse":
            if config == "collapse+emb":
                return CollapseCluster.run_cluster_collapse_emb(file)
            elif config == "emb+collapse":
                return CollapseCluster.run_cluster_emb_collapse(file)
            elif config == "emb":
                return CollapseCluster.run_cluster_emb(file)
            elif config == "collapse":
                return CollapseCluster.run_cluster_collapse(file)
            elif config == "collapse+embsplit":
                return CollapseCluster.run_cluster_collapse_embsplit(file)
            elif config == "collapse+prompt_label":
                return CollapseCluster.run_cluster_collapse_promptsplit(file)
            elif config == "collapse+prompt_label_gpt":
                return CollapseCluster.run_cluster_collapse_promptsplit_gpt(file)
            else:
                print("Unknown config, using collapse + emb")
                return CollapseCluster.run_cluster_collapse_emb(file)
        else:
            return Cluster.runCluster(file, clusteralg)