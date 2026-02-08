from typing import List
import re
from Utils.config import Config

class DataFeature:
    data_features = {}

    @staticmethod
    def get_all_strings_list() -> List[str]:
        return [DataFeature.data_features[i]['text'] for i in DataFeature.data_features.keys()]
    
    @staticmethod
    def get_all_strings_dict() -> dict:
        return {i: DataFeature.data_features[i]['text'] for i in DataFeature.data_features.keys()}
    
    @staticmethod
    def get_all_collapses_list() -> List[str]:
        return [DataFeature.data_features[i]['collapse'] for i in DataFeature.data_features.keys()]
    
    @staticmethod
    def get_all_collapses_dict() -> dict:
        return {i: DataFeature.data_features[i]['collapse'] for i in DataFeature.data_features.keys()}
    
    @staticmethod
    def get_all_compositions_list() -> List[str]:
        return [DataFeature.data_features[i]['composition'] for i in DataFeature.data_features.keys()]
    
    @staticmethod
    def get_all_compositions_dict() -> dict:
        return {i: DataFeature.data_features[i]['composition'] for i in DataFeature.data_features.keys()}
    
    @staticmethod
    def get_labels_from_string(s: str) -> List[str]:
        labels = []
        # Extract all the exact strings between square brackets in s
        matches = re.findall(r'\[([^\]]+)\]', s)
        for match in matches:
            labels.append(match)
        return labels

    @staticmethod
    def get_all_sem_types(file: str) -> List[str]:
        sem_types = set()
        if Config.perfect_cluster:
            type_file = f"{Config.perfect_directory}/{file.split('/')[-1].split('.')[0]}_types.txt"
            try:
                with open(type_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        sem_types.add(line.strip())
            except FileNotFoundError:
                print(f"Type file {type_file} not found. Falling back to extracting from data features.")
                for i in DataFeature.data_features.keys():
                    composition = DataFeature.data_features[i]['composition']
                    labels = DataFeature.get_labels_from_string(composition)
                    for label in labels:
                        sem_types.add(label)
        else:
            for i in DataFeature.data_features.keys():
                composition = DataFeature.data_features[i]['composition']
                labels = DataFeature.get_labels_from_string(composition)
                for label in labels:
                    sem_types.add(label)
        return list(sem_types)