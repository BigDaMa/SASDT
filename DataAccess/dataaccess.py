from typing import Tuple, List, Dict
import pandas as pd
import os

from Utils.CacheUtil import cache_result

class DataAccess:
    DATA_MIX_FOLDER = 'data_mix'
    DATA_STRATIFIED_FOLDER = 'data_stratified'
    DATA_BACKUP_FOLDER = 'data_backup'
    DATA_FOLDER = 'data_shuffled'
    DATA_SHUFFLED_FOLDER = 'data_shuffled'
    NUM_EXAMPLES = 5

    @staticmethod
    def get_data_from_file(dir: str, file: str) -> pd.DataFrame:
        file_path = os.path.join(dir, file)
        return pd.read_csv(file_path, header = None)
    
    @staticmethod
    def get_examples(data: pd.DataFrame, numexamples: int = NUM_EXAMPLES) -> List[Dict[str, str]]:
        examples = []
        for i in range(0, numexamples):
            examples.append({
                "input": data.iloc[i, 0],
                "output": data.iloc[i, 1]
            })
        return examples

    @staticmethod
    def split_data(data: pd.DataFrame, numexamples: int = 5) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
        examples = []
        texts = data['text'].tolist()
        labels = data['label'].tolist()

        for i in range(0, numexamples):
            examples.append({
                "input": data.iloc[i, 0],
                "output": data.iloc[i, 1]
            })
        query = data.iloc[numexamples:, 0].tolist()
        ground_truth = data.iloc[numexamples:, 1].tolist()

        return examples, query, ground_truth
    
    @staticmethod
    def get_mock_data() -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]], List[List[str]]]:
        input_components = {"Sternbuschweg 52, 47057 Duisburg": {'street': 'Sternbuschweg', 'house number': '52', 
                    'postal code': '47057', 'city': 'Duisburg'},
                    "77 Massachusetts Avenue, Cambridge, MA 02139, USA": {'house number': '77', 'street': 'Massachusetts Avenue', 'city': 'Cambridge', 'state': 'MA', 'postal code': '02139', 'country': 'USA'}}
        output_components = {"Sternbuschweg 52": {'street': 'Sternbuschweg', 'house number': '52'},
                            "77 Massachusetts Avenue": {'street': 'Massachusetts Avenue', 'house number': '77'}}

        test = [
            ["Bd de la Plaine 2, 1050 Ixelles", "Karolinenplatz 5, 64289 Darmstadt", "Europaplatz 1, 10557 Berlin"],
            ["88 Altona St, Kensington, VIC 3031, Australia", "333 Biscayne Boulevard Way, Miami, Florida 33131, USA", "2331 Broadway, Sacramento, CA 95818, USA"],
            ["C. Alba de Tormes, Km2, 37188 Carbajosa de la Sagrada, Salamanca, Spain", "Av. Complutense, s/n, Moncloa - Aravaca, 28040 Madrid", "Via Castello di Verrazzano, 1, 50022 Greve in Chianti FI"],
            ["Bd de la Plaine 2", "Karolinenplatz 5", "Massachusetts Ave 77"]
        ]

        return input_components, output_components, test
    
    @staticmethod
    def get_data_for_clustering(data: pd.DataFrame, numexamples: int = NUM_EXAMPLES) -> List[str]:

        data_list = list()
        data_list = list(data.iloc[:, 0]) + list(data.iloc[:numexamples, 1])
        return data_list
    
    @staticmethod
    @cache_result()
    def map_clustered_idx_to_str(split_clusters: List[List[int]], data_cluster: List[str]) -> List[Dict[int, str]]:
        mapped_clusters = []
        for cluster in split_clusters:
            mapped_cluster = {idx: data_cluster[idx] for idx in cluster}
            mapped_clusters.append(mapped_cluster)
        return mapped_clusters