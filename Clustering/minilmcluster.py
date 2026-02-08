from typing import Dict, List, Set
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import os
import time
import numpy as np

from DataAccess.dataaccess import DataAccess
from processtruth import group_indices_by_label, evaluate_pairwise

__DATA_DIR__ = 'data'
__CACHE_DIR__ = 'cache'
__TRUTH_DIR__ = 'truth'

__LEN_NUM_FEATURE__ = 0

class MiniLMCluster:

    @staticmethod
    def custom_distance(x, y, alpha = 0.5):
        num_x = x[:__LEN_NUM_FEATURE__] if len(x) > __LEN_NUM_FEATURE__ and __LEN_NUM_FEATURE__ > 0 else None
        num_y = y[:__LEN_NUM_FEATURE__] if len(y) > __LEN_NUM_FEATURE__ and __LEN_NUM_FEATURE__ > 0 else None
        vec_x = x[__LEN_NUM_FEATURE__:]
        vec_y = y[__LEN_NUM_FEATURE__:]
        vec_distance = cosine_distances([vec_x], [vec_y])[0][0]
        num_distance = euclidean_distances([num_x], [num_y])[0][0] if num_x is not None and num_y is not None else 0.0
        if num_x is None and num_y is None:
            return vec_distance  # If no numerical features, return only vector distance
        combined_distance = alpha * vec_distance + (1 - alpha) * num_distance
        return combined_distance

    @staticmethod
    def embed_text(model, text):
        return model.encode(text)

    @staticmethod
    def get_text_embeddings(model, data: pd.DataFrame) -> List[List[float]]:
        return [MiniLMCluster.embed_text(model, text) for text in data.iloc[:, 0].tolist()]

    @staticmethod
    def cluster(clusterer, data: List[List[float]], truth: List[Set[int]]) -> Dict[str, float]:
        labels = clusterer.fit(data).labels_
        labelIndex = group_indices_by_label(labels)
        labelIndex = [set([int(i - 1) for i in list(x) if i != np.nan]) for x in labelIndex]
        result = evaluate_pairwise(labelIndex, truth)
        f1 = 0
        numitem = 0
        for clus, res in result.items():
            if(res['p'] == 0 and res['r'] == 0):
                continue
            if(clus != 'mean'):
                currf1 = 0 if res['p'] + res['r'] == 0 else float(float(2 * res['p'] * res['r']) / float(res['p'] + res['r']))
                result[clus]['f1'] = currf1
                f1 += currf1
                numitem += 1
        result['mean']['f1'] = float(f1) / float(numitem) if numitem > 0 else 0
        return result

for file in os.listdir(DataAccess.DATA_FOLDER):
    if file.endswith('.csv'):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        clusterer = DBSCAN(eps = 0.4, min_samples = 8, metric=MiniLMCluster.custom_distance)
        data = pd.read_csv(os.path.join(DataAccess.DATA_FOLDER, file), header=None)
        startemb = time.time()
        data_embedding = MiniLMCluster.get_text_embeddings(model, data)
        endemb = time.time()
        
        startpredeval = time.time()
        truthtable = pd.read_csv(os.path.join(__TRUTH_DIR__, file.replace('.csv', '_processed.csv')), header=None).values.tolist()
        truthtable = [set(int(val) for val in row if pd.notna(val) and val != '') for row in truthtable]
        featuretable = pd.read_csv(os.path.join(__CACHE_DIR__, file.replace('.csv', '_feature.csv')), header=None).values.tolist()
        # Drop the last column from featuretable
        featuretable = [row[:-1] for row in featuretable]
        __LEN_NUM_FEATURE__ = len(featuretable[0]) if featuretable else 0
        combined_data = np.hstack((featuretable, data_embedding))
        
        result = MiniLMCluster.cluster(clusterer, combined_data, truthtable)
        endpredeval = time.time()