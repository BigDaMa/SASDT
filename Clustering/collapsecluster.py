import re
from typing import List, Dict, Set, Tuple, Union
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from kneefinder import KneeFinder
import unidecode
from collections import Counter

from DataAccess.dataaccess import DataAccess
from LLMUtil.LLMUtil import OpenAIUtil_Prompt, OLLAMAUtil, PromptUtil
from Utils.CacheUtil import cache_result
from Utils.config import Config
from Utils.ModelSelect import ModelSelect
from DataAccess.datafeature import DataFeature

try:
    model = ModelSelect.get_clustering_model_by_name("local-qwen-emb")
except:
    model = None

__DATA_DIR__ = 'data'
__CACHE_DIR__ = 'cache'
__TRUTH_DIR__ = 'truth'

__SPARSE_CLUSTER_THRESHOLD__ = 3
__SPARSE_CLUSTER_RATIO__ = 0.5
__SPARSE_CLUSTER_ABS_NUM__ = 2

class CollapseCluster:

    @staticmethod
    def class_based_collapse(text: str) -> str:
        """
        Collapse the text based on class-based regex
        :param text: The input text to collapse.
        :return: The collapsed text.
        """
        text = unidecode.unidecode(text)  # Normalize text to ASCII
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'[A-Z]', "A", text)  # Replace uppercase letters with 'A'
        text = re.sub(r'[a-z]', "a", text)  # Replace lowercase letters with 'a'
        text = re.sub(r'\d', "0", text)  # Replace digits with '0'
        # Collapse consecutive A's, a's and 0's
        text = re.sub(r'(A+)', 'A', text)  # Collapse consecutive 'A's
        text = re.sub(r'(a+)', 'a', text)  # Collapse consecutive 'a's
        text = re.sub(r'(0+)', '0', text)  # Collapse consecutive '0's
        return text

    @staticmethod
    def get_embedding(text: str) -> List[float]:
        """
        Get the embedding for the input text using the MiniLM model.
        
        :param text: The input text to embed.
        :return: The embedding of the input text.
        """
        return model.encode(text)

    @staticmethod
    def collapse_repeated_segments(s: str) -> str:
        segments: List[Tuple[str,int,int]] = []
        i = 0
        n = len(s)

        while i < n:
            if s[i].isspace():
                i += 1
                continue
            start = i
            while i < n and s[i].isalnum():
                i += 1
            while i < n and not s[i].isalnum() and not s[i].isspace():
                i += 1
            segments.append((s[start:i], start, i))

        result = []
        idx = 0
        while idx < len(segments):
            seg, st, en = segments[idx]
            if idx + 1 < len(segments) and segments[idx + 1][0] == seg:
                result.append(seg)
                idx += 2  
            else:
                result.append(seg)
                idx += 1

        out = []
        last = 0
        for seg in result:
            pos = s.find(seg, last)
            out.append(s[last:pos])
            out.append(seg)
            last = pos + len(seg)
        out.append(s[last:])

        return ''.join(out)

    @staticmethod
    def get_rare_nonword(data_feature: Dict[int, Dict[str, List[float]]], keep_nonword_num = 2) -> List[str]:
        nonword_count = dict()
        for i, feature in data_feature.items():
            if 'collapsed_relax' in feature.keys():
                collapsed_text = feature['collapsed_relax']
            else:
                collapsed_text = feature['collapsed']
            nonword_chars = re.findall(r'[^\w\s]', collapsed_text)
            for char in nonword_chars:
                if char not in nonword_count:
                    nonword_count[char] = 0
                nonword_count[char] += 1
        rare_nonword = [char for char, count in nonword_count.items() if count == min(nonword_count.values())]
        num_nonword = len(nonword_count.keys())
        if not rare_nonword or num_nonword <= keep_nonword_num:
            return []
        return rare_nonword

    @staticmethod
    def relax_collapse(data_feature: Dict[int, Dict[str, List[float]]], index_list: List[int], rare_nonword: List[str]) -> Dict[str, Dict[str, List[float]]]:
        relaxed_data_feature = dict()
        for i, feature in data_feature.items():
            if i not in index_list:
                continue
            if 'collapsed_relax' in feature.keys():
                collapsed_text = feature['collapsed_relax']
            else:
                collapsed_text = feature['collapsed']
            for nonword in rare_nonword:
                collapsed_text = collapsed_text.replace(nonword, '')

            collapsed_text = re.sub(r'(?<=[a-zA-Z0-9])A', 'a', collapsed_text)
            
            collapsed_text = collapsed_text.strip()
            collapsed_text = re.sub(r'\s+', ' ', collapsed_text)  # Replace multiple spaces with a single space
            collapsed_text = re.sub(r'(A+)', 'A', collapsed_text)  # Collapse consecutive 'A's
            collapsed_text = re.sub(r'(a+)', 'a', collapsed_text)  # Collapse consecutive 'a's
            collapsed_text = re.sub(r'(0+)', '0', collapsed_text)  # Collapse consecutive '0's
            collapsed_text = CollapseCluster.collapse_repeated_segments(collapsed_text) # Collapse repeated segments

            relaxed_data_feature[i] = feature.copy()
            relaxed_data_feature[i]['collapsed_relax'] = collapsed_text
        return relaxed_data_feature

    @staticmethod
    def collapse_cluster(data_feature: Dict[str, Dict[str, List[float]]]) -> List[Set[int]]:
        clusters = dict()
        for i, feature in data_feature.items():
            if 'collapsed_relax' in feature.keys():
                collapsed_text = feature['collapsed_relax']
            else:
                collapsed_text = feature['collapsed']
            if collapsed_text not in clusters:
                clusters[collapsed_text] = list()
            clusters[collapsed_text].append(i)
        return list(clusters.values())

    @staticmethod
    def group_indices_by_label(labels: List[int], cluster: List[int]) -> List[Set[int]]:
        label_index = {}
        for index, label in enumerate(labels):
            if label not in label_index:
                label_index[label] = set()
            label_index[label].add(cluster[index])
        return list(label_index.values())

    @staticmethod
    def get_knee_point(vectors: List[List[float]], k: int = 10) -> float:
        test_eps = np.linspace(0.1, 1.0, k)
        n_clust = []
        for eps in test_eps:
            db = DBSCAN(eps=eps, min_samples=__SPARSE_CLUSTER_ABS_NUM__, metric='cosine').fit(vectors)
            n_clust.append(len(set(db.labels_)))
        kf = KneeFinder(data_x=test_eps, data_y=n_clust, clean_data=False)
        kf.find_knee()
        if kf.knee is not None:
            eps = kf.knee[0]
        else:
            eps = 0.5
        return eps

    @staticmethod
    def split_cluster_by_embedding(cluster: List[int], data_feature: Dict[str, Dict[str, List[float]]]) -> List[Set[int]]:
        cluster_embeddings = [data_feature[i]['embedding'] for i in cluster]
        cluster_embeddings = [list(embedding) for embedding in cluster_embeddings]
        eps = CollapseCluster.get_knee_point(cluster_embeddings)
        cluster_model = DBSCAN(eps=eps, min_samples=__SPARSE_CLUSTER_THRESHOLD__, metric='cosine')
        labels = cluster_model.fit_predict(cluster_embeddings)
        return CollapseCluster.group_indices_by_label(labels, cluster)
    
    @staticmethod
    def split_cluster_by_composition(cluster: List[int], data_feature: Dict[str, Dict[str, str]]) -> List[Set[int]]:
        composition_map = dict()
        for i in cluster:
            composition = data_feature[i]['composition']
            if composition not in composition_map:
                composition_map[composition] = set()
            composition_map[composition].add(i)
        return list(composition_map.values())
    
    @staticmethod
    def merge_sparse_clusters(clusters: List[Set[int]], data_feature: Dict[str, Dict[str, List[float]]]) -> List[Set[int]]:
        sparse_clusters = [c for c in clusters if len(c) < __SPARSE_CLUSTER_THRESHOLD__]
        non_sparse_clusters = [c for c in clusters if len(c) >= __SPARSE_CLUSTER_THRESHOLD__]
        non_sparse_clusters_centroid = []
        if non_sparse_clusters:
            for cluster in non_sparse_clusters:
                embeddings = [data_feature[i]['embedding'] for i in cluster]
                centroid = np.mean(embeddings, axis=0)
                non_sparse_clusters_centroid.append(centroid)
            for sparse_cluster in sparse_clusters:
                for point in sparse_cluster:
                    point_embedding = data_feature[point]['embedding']
                    if non_sparse_clusters_centroid:
                        distances = cosine_distances([point_embedding], non_sparse_clusters_centroid)[0]
                        min_index = np.argmin(distances)
                        non_sparse_clusters[min_index].add(point)
                        non_sparse_clusters_centroid[min_index] = np.mean([data_feature[i]['embedding'] for i in non_sparse_clusters[min_index]], axis=0)
        else:
            cluster = DBSCAN(eps = 1, min_samples = 3)
            clusters_flat = [i for c in clusters for i in c]
            labels = cluster.fit_predict([data_feature[i]['embedding'] for i in clusters_flat])
            non_sparse_clusters = CollapseCluster.group_indices_by_label(labels, clusters_flat)
        return non_sparse_clusters

    @staticmethod
    def get_labeled_clusters(clusters: List[Set[int]]) -> List[List[int]]:
        numclusters = len(clusters)
        labeled_clusters = dict()
        final_cluster = []
        for i, cluster in enumerate(clusters):
            for item in cluster:
                if item not in labeled_clusters:
                    labeled_clusters[item] = i
        final_cluster = [labeled_clusters[item] for item in sorted(labeled_clusters.keys())]
        return numclusters, final_cluster

    @staticmethod
    @cache_result()
    def run_cluster_emb(file: str) -> List[Set[int]]:
        data_orig = DataAccess.get_data_from_file(DataAccess.DATA_FOLDER, file)
        data = DataAccess.get_data_for_clustering(data_orig)[:-DataAccess.NUM_EXAMPLES]
        output_labels = [len(data) + i for i in range(DataAccess.NUM_EXAMPLES)]
        data = pd.DataFrame(data, columns=None)
        data_features = dict()
        for i, row in data.iterrows():
            text = row[0]
            collapsed_text = CollapseCluster.class_based_collapse(text)
            embedding = CollapseCluster.get_embedding(text)
            data_features[i] = {
                'text': text,
                'collapsed': collapsed_text,
                'embedding': embedding
            }
        knee_point = CollapseCluster.get_knee_point([data_features[i]['embedding'] for i in range(len(data_features))])
        clusterer = DBSCAN(eps=knee_point, min_samples=__SPARSE_CLUSTER_THRESHOLD__, metric='cosine')
        labels = clusterer.fit_predict([data_features[i]['embedding'] for i in range(len(data_features))])
        clusters = CollapseCluster.group_indices_by_label(labels, list(data_features.keys()))
        clusters = [set(c) for c in clusters]
        clusters.append(set(output_labels))
        return clusters

    @staticmethod
    @cache_result()
    def run_cluster_emb_collapse(file: str) -> List[Set[int]]:
        data_orig = DataAccess.get_data_from_file(DataAccess.DATA_FOLDER, file)
        data = DataAccess.get_data_for_clustering(data_orig)[:-DataAccess.NUM_EXAMPLES]
        output_labels = [len(data) + i for i in range(DataAccess.NUM_EXAMPLES)]
        data = pd.DataFrame(data, columns=None)
        data_features = dict()
        for i, row in data.iterrows():
            text = row[0]
            collapsed_text = CollapseCluster.class_based_collapse(text)
            embedding = CollapseCluster.get_embedding(text)
            data_features[i] = {
                'text': text,
                'collapsed': collapsed_text,
                'embedding': embedding
            }
        knee_point = CollapseCluster.get_knee_point([data_features[i]['embedding'] for i in range(len(data_features))])
        clusterer = DBSCAN(eps=knee_point, min_samples=__SPARSE_CLUSTER_THRESHOLD__, metric='cosine')
        labels = clusterer.fit_predict([data_features[i]['embedding'] for i in range(len(data_features))])
        clusters = CollapseCluster.group_indices_by_label(labels, list(data_features.keys()))
        split_clusters = []
        for cluster in clusters:
            if len(cluster) <= __SPARSE_CLUSTER_THRESHOLD__:
                split_clusters.append(set(cluster))
                continue
            curr_features = {i: data_features[i] for i in cluster}
            collapse_clusters = CollapseCluster.collapse_cluster(curr_features)
            if len(collapse_clusters) == 1:
                split_clusters.append(set(cluster))
                continue
            while True:
                sparse_indices = [i for i, c in enumerate(collapse_clusters) if len(c) < __SPARSE_CLUSTER_THRESHOLD__]
                
                if not sparse_indices:
                    break  # No more sparse clusters to merge
                
                for sparse_idx in sparse_indices:
                    if len(collapse_clusters[sparse_idx]) == 0:
                        continue  # Skip empty clusters
                    
                    sparse_cluster = collapse_clusters[sparse_idx]
                    sparse_centroid = np.mean([data_features[idx]['embedding'] for idx in sparse_cluster], axis=0)
                    
                    min_distance = float('inf')
                    best_merge_idx = -1
                    
                    for other_idx, other_cluster in enumerate(collapse_clusters):
                        if other_idx == sparse_idx or len(other_cluster) == 0:
                            continue
                        
                        other_centroid = np.mean([data_features[idx]['embedding'] for idx in other_cluster], axis=0)
                        
                        distance = cosine_distances([sparse_centroid], [other_centroid])[0][0]
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_merge_idx = other_idx
                    
                    if best_merge_idx != -1:
                        collapse_clusters[best_merge_idx].extend(collapse_clusters[sparse_idx])
                        collapse_clusters[sparse_idx] = []  # Clear the merged cluster
                
                collapse_clusters = [c for c in collapse_clusters if len(c) > 0]
            
            for c in collapse_clusters:
                if c:
                    split_clusters.append(set(c))
        split_clusters.append(set(output_labels))

        return split_clusters

    @staticmethod
    @cache_result()
    def run_cluster_collapse(file: str) -> List[Set[int]]:
        data_orig = DataAccess.get_data_from_file(DataAccess.DATA_FOLDER, file)
        data = DataAccess.get_data_for_clustering(data_orig)[:-DataAccess.NUM_EXAMPLES]
        output_labels = [len(data) + i for i in range(DataAccess.NUM_EXAMPLES)]
        data = pd.DataFrame(data, columns=None)
        data_features = dict()
        for i, row in data.iterrows():
            text = row[0]
            collapsed_text = CollapseCluster.class_based_collapse(text)
            embedding = CollapseCluster.get_embedding(text)
            data_features[i] = {
                'text': text,
                'collapsed': collapsed_text,
                'embedding': embedding
            }
        clusters = CollapseCluster.collapse_cluster(data_features)
        clusters = [set(c) for c in clusters]
        clusters.append(set(output_labels))
        return clusters

    @staticmethod
    @cache_result()
    def run_cluster_collapse_emb(file: str) -> List[Set[int]]:
        data_orig = DataAccess.get_data_from_file(DataAccess.DATA_FOLDER, file)
        data = DataAccess.get_data_for_clustering(data_orig)[:-DataAccess.NUM_EXAMPLES]
        output_labels = [len(data) + i for i in range(DataAccess.NUM_EXAMPLES)]
        data = pd.DataFrame(data, columns=None)
        data_features = dict()
        for i, row in data.iterrows():
            text = row[0]
            collapsed_text = CollapseCluster.class_based_collapse(text)
            embedding = CollapseCluster.get_embedding(text)
            data_features[i] = {
                'text': text,
                'collapsed': collapsed_text,
                'embedding': embedding
            }
        clusters = CollapseCluster.collapse_cluster(data_features)
        sparse_clusters = [x for x in clusters if len(x) < __SPARSE_CLUSTER_THRESHOLD__]
        iteration = 1
        while(len(sparse_clusters) > __SPARSE_CLUSTER_ABS_NUM__ and len(sparse_clusters) * 1.0 / len(clusters) >= __SPARSE_CLUSTER_RATIO__):
            iteration += 1
            sparse_clusters_flat = [item for sublist in sparse_clusters for item in sublist]
            rare_nonword = CollapseCluster.get_rare_nonword(data_features, keep_nonword_num=2)
            if rare_nonword:
                relaxed_data_features = CollapseCluster.relax_collapse(data_features, list(data_features.keys()), rare_nonword)
                for i in relaxed_data_features.keys():
                    data_features[i]['collapsed_relax'] = relaxed_data_features[i]['collapsed_relax']
                clusters = CollapseCluster.collapse_cluster(data_features)
                sparse_clusters = [x for x in clusters if len(x) < __SPARSE_CLUSTER_THRESHOLD__]
            else:
                break
        split_clusters = []
        for cluster in clusters:
            if len(cluster) < __SPARSE_CLUSTER_THRESHOLD__:
                split_clusters.append(set(cluster))
            else:
                split_clusters.extend(CollapseCluster.split_cluster_by_embedding(list(cluster), data_features))
        split_clusters = CollapseCluster.merge_sparse_clusters(split_clusters, data_features)
        split_clusters.append(set(output_labels))
        numcluster, labeled_clusters = CollapseCluster.get_labeled_clusters(split_clusters)
        cluster_counts = Counter(labeled_clusters)
        num_sparse_clusters = sum(1 for count in cluster_counts.values() if count < __SPARSE_CLUSTER_THRESHOLD__)
        sparse_clusters_ratio = num_sparse_clusters * 1.0 / len(cluster_counts) if cluster_counts else 0.0
        ratio_instance_in_sparse_clusters = sum(count for count in cluster_counts.values() if count < __SPARSE_CLUSTER_THRESHOLD__) * 1.0 / sum(count for count in cluster_counts.values()) if sum(count for count in cluster_counts.values()) else 0.0
        
        return split_clusters
    
    @staticmethod
    @cache_result()
    def run_cluster_collapse_embsplit(file: str) -> List[Set[int]]:
        data_orig = DataAccess.get_data_from_file(DataAccess.DATA_FOLDER, file)
        data = DataAccess.get_data_for_clustering(data_orig)[:-DataAccess.NUM_EXAMPLES]
        output_labels = [len(data) + i for i in range(DataAccess.NUM_EXAMPLES)]
        data = pd.DataFrame(data, columns=None)
        data_features = dict()
        for i, row in data.iterrows():
            text = row[0]
            collapsed_text = CollapseCluster.class_based_collapse(text)
            embedding = CollapseCluster.get_embedding(text)
            DataFeature.data_features[i] = {
                'text': text,
                'collapsed': collapsed_text,
                'embedding': embedding
            }
        clusters = CollapseCluster.collapse_cluster(data_features)
        split_clusters = []
        for cluster in clusters:
            if len(cluster) == 1:
                split_clusters.append(set(cluster))
            else:
                split_clusters.extend(CollapseCluster.split_cluster_by_embedding(list(cluster), data_features))
        split_clusters.append(set(output_labels))
        return split_clusters
    
    @staticmethod
    @cache_result()
    def run_cluster_collapse_promptsplit(file: str) -> Union[List[Set[int]], Dict[int, Dict[str, str]]]:
        DataFeature.data_features = {}
        data_orig = DataAccess.get_data_from_file(DataAccess.DATA_FOLDER, file)
        data_full = DataAccess.get_data_for_clustering(data_orig)
        data = data_full[:-5]
        example_outputs = data_full[-5:]
        num_all_data = len(data)
        output_labels = [num_all_data + i for i in range(DataAccess.NUM_EXAMPLES)]
        data = pd.DataFrame(data, columns=None)
        need_sem_split = True
        
        for i, row in data.iterrows():
            text = row[0]
            cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text)
            if cleaned_text.isdigit():
                need_sem_split = False
                break
        
        for i, row in data.iterrows():
            text = row[0]
            collapsed_text = CollapseCluster.class_based_collapse(text)
            DataFeature.data_features[i] = {
                'text': text,
                'collapsed': collapsed_text,
                'composition': ""
            }
        output_data_features = {}
        for i, row in enumerate(data_full[len(data):len(data) + DataAccess.NUM_EXAMPLES]):
            if i < DataAccess.NUM_EXAMPLES:
                text = row
                collapsed_text = CollapseCluster.class_based_collapse(text)
                output_data_features[num_all_data + i] = {
                    'text': text,
                    'collapsed': collapsed_text,
                    'composition': ""
                }



        examples = {i: DataFeature.data_features[i]['text'] for i in range(DataAccess.NUM_EXAMPLES)} | {num_all_data+i: example_outputs[i] for i in range(DataAccess.NUM_EXAMPLES)}
        input_ids = [i for i in range(num_all_data)]
        output_ids = output_labels
        split = OpenAIUtil_Prompt.get_split_prompt(examples, file, input_ids, output_ids, prompt = "label", always_split_further = False)
        for k, v in split.items():
            if k in DataFeature.data_features.keys():
                compstr = ""
                for label in v:
                    if label.startswith("[") and label.endswith("]"):
                        compstr += "{}".format("_".join(label.split()))
                    else:
                        compstr += "[{}]".format("_".join(label.split()))
                if not compstr.endswith("]"):
                    compstr = compstr[:-1]
                DataFeature.data_features[k]['composition'] = "".join(compstr)
            elif k in output_data_features.keys():
                compstr = ""
                for label in v:
                    if label.startswith("[") and label.endswith("]"):
                        compstr += "{}".format("_".join(label.split()))
                    else:
                        compstr += "[{}]".format("_".join(label.split()))
                if not compstr.endswith("]"):
                    compstr = compstr[:-1]
                output_data_features[k]['composition'] = "".join(compstr)

        if need_sem_split:
            example_splits = [(examples[i], DataFeature.data_features[i]['composition']) for i in split.keys() if i < len(data)]
            queries = {i: DataFeature.data_features[i]['text'] for i in DataFeature.data_features.keys() if i not in split.keys()}
            prompts = [PromptUtil.build_ollama_query(example_splits, q) for q in queries.values()]
            all_splits = OLLAMAUtil.get_response_batch(prompts)

            processed_splits = []
            for split_string in all_splits:
                segments = re.findall(r'[a-zA-Z0-9_\s]+', split_string)
                
                formatted_segments = []
                for segment in segments:
                    trimmed = segment.strip()
                    if trimmed:  # Only process non-empty segments
                        formatted = re.sub(r'\s+', '_', trimmed)
                        formatted_segments.append(formatted)
                
                if formatted_segments:
                    formatted_string = ''.join(f'[{segment}]' for segment in formatted_segments)
                else:
                    formatted_string = ''
                
                processed_splits.append(formatted_string)
            
            for k, v in zip(queries.keys(), processed_splits):
                if k in DataFeature.data_features.keys():
                    DataFeature.data_features[k]['composition'] = v
                elif k in output_data_features.keys():
                    output_data_features[k]['composition'] = v

        clusters = CollapseCluster.collapse_cluster(DataFeature.data_features)
        split_clusters = []
        if need_sem_split:
            for cluster in clusters:
                if len(cluster) == 1:
                    split_clusters.append(set(cluster))
                else:
                    split_clusters.extend(CollapseCluster.split_cluster_by_composition(list(cluster), DataFeature.data_features))
        else:
            for cluster in clusters:
                split_clusters.append(set(cluster))
        split_clusters.append(set(output_labels))
        for k, v in output_data_features.items():
            DataFeature.data_features[k] = v
        return split_clusters
    
    @staticmethod
    @cache_result()
    def run_cluster_collapse_promptsplit_gpt(file: str) -> Union[List[Set[int]], Dict[int, Dict[str, str]]]:
        DataFeature.data_features = {}
        data_orig = DataAccess.get_data_from_file(DataAccess.DATA_FOLDER, file)
        data_full = DataAccess.get_data_for_clustering(data_orig)
        data = data_full[:-5]
        example_outputs = data_full[-5:]
        num_all_data = len(data)
        output_labels = [num_all_data + i for i in range(DataAccess.NUM_EXAMPLES)]
        data = pd.DataFrame(data, columns=None)
        need_sem_split = True
        
        for i, row in data.iterrows():
            text = row[0]
            cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text)
            if cleaned_text.isdigit():
                need_sem_split = False
                break
        
        for i, row in data.iterrows():
            text = row[0]
            collapsed_text = CollapseCluster.class_based_collapse(text)
            DataFeature.data_features[i] = {
                'text': text,
                'collapsed': collapsed_text,
                'composition': ""
            }
        output_data_features = {}
        for i, row in enumerate(data_full[-DataAccess.NUM_EXAMPLES:]):
            if i < DataAccess.NUM_EXAMPLES:
                text = row
                collapsed_text = CollapseCluster.class_based_collapse(text)
                output_data_features[num_all_data + i] = {
                    'text': text,
                    'collapsed': collapsed_text,
                    'composition': ""
                }



        examples = {i: DataFeature.data_features[i]['text'] for i in range(DataAccess.NUM_EXAMPLES)} | {num_all_data+i: example_outputs[i] for i in range(DataAccess.NUM_EXAMPLES)}
        input_ids = [i for i in range(num_all_data)]
        output_ids = output_labels
        split = OpenAIUtil_Prompt.get_split_prompt(examples, file, input_ids, output_ids, prompt = "label", always_split_further = False)
        for k, v in split.items():
            if k in DataFeature.data_features.keys():
                compstr = ""
                for label in v:
                    if label.startswith("[") and label.endswith("]"):
                        compstr += "{}".format("_".join(label.split()))
                    else:
                        compstr += "[{}]".format("_".join(label.split()))
                if not compstr.endswith("]"):
                    compstr = compstr[:-1]
                DataFeature.data_features[k]['composition'] = "".join(compstr)
            elif k in output_data_features.keys():
                compstr = ""
                for label in v:
                    if label.startswith("[") and label.endswith("]"):
                        compstr += "{}".format("_".join(label.split()))
                    else:
                        compstr += "[{}]".format("_".join(label.split()))
                if not compstr.endswith("]"):
                    compstr = compstr[:-1]
                output_data_features[k]['composition'] = "".join(compstr)

        if need_sem_split:
            example_splits = [(examples[i], DataFeature.data_features[i]['composition']) for i in split.keys() if i < len(data)]
            queries = {i: DataFeature.data_features[i]['text'] for i in DataFeature.data_features.keys() if i not in split.keys()}
            prompts = [PromptUtil.build_ollama_query(example_splits, q) for q in queries.values()]
            all_splits = OpenAIUtil_Prompt.get_sem_component_labels(prompts)

            processed_splits = []
            for split_string in all_splits:
                segments = re.findall(r'[a-zA-Z0-9_\s]+', split_string)
                
                formatted_segments = []
                for segment in segments:
                    trimmed = segment.strip()
                    if trimmed:  # Only process non-empty segments
                        formatted = re.sub(r'\s+', '_', trimmed)
                        formatted_segments.append(formatted)
                
                if formatted_segments:
                    formatted_string = ''.join(f'[{segment}]' for segment in formatted_segments)
                else:
                    formatted_string = ''
                
                processed_splits.append(formatted_string)
            
            for k, v in zip(queries.keys(), processed_splits):
                if k in DataFeature.data_features.keys():
                    DataFeature.data_features[k]['composition'] = v
                elif k in output_data_features.keys():
                    output_data_features[k]['composition'] = v

        clusters = CollapseCluster.collapse_cluster(DataFeature.data_features)
        split_clusters = []
        if need_sem_split:
            for cluster in clusters:
                if len(cluster) == 1:
                    split_clusters.append(set(cluster))
                else:
                    split_clusters.extend(CollapseCluster.split_cluster_by_composition(list(cluster), DataFeature.data_features))
        else:
            for cluster in clusters:
                split_clusters.append(set(cluster))
        split_clusters.append(set(output_labels))
        for k, v in output_data_features.items():
            DataFeature.data_features[k] = v
        return split_clusters