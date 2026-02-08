from statistics import mean
from typing import List, Set
import os
import itertools
import pandas as pd

from DataAccess.dataaccess import DataAccess

__CLUS_DIR__ = 'data_cluster'
__DATA_DIR__ = 'data'

class ProcessTruth:

    @staticmethod
    def process(truthfile: str, returntype = 'set') -> List[Set[int]]:
        truth = pd.read_csv(truthfile, header=None)
        if returntype == 'set':
            truthTable = [set(int(val) for val in row if pd.notna(val) and val != '') for row in truth.values]
        elif returntype == 'list':
            truthTable = [list(int(val) for val in row if pd.notna(val) and val != '') for row in truth.values]

        return truthTable

    @staticmethod
    def process_labeled(datafile: str, numexamples: int = DataAccess.NUM_EXAMPLES) -> List[int]:
        truth_column = 1
        data_cluster = pd.read_csv(os.path.join(__CLUS_DIR__, datafile), header=None)
        data = pd.read_csv(os.path.join(DataAccess.DATA_FOLDER, datafile), header=None)
        
        data_list = list(data.iloc[:, 0]) + list(data.iloc[:numexamples, 1])
        
        position_map = {value: idx for idx, value in enumerate(data_list)}
        
        data_cluster['sort_order'] = data_cluster[data_cluster.columns[0]].map(position_map)
        data_cluster_sorted = data_cluster.sort_values('sort_order').drop('sort_order', axis=1).reset_index(drop=True)
        
        return data_cluster_sorted.iloc[:len(data_list), truth_column].dropna().astype(int).tolist()

    @staticmethod
    def group_indices_by_label(labels: List[int]) -> List[Set[int]]:
        index_dict = {}
        
        for index, label in enumerate(labels):
            if label not in index_dict:
                index_dict[label] = set()
            index_dict[label].add(index + 1)
        
        result = list(index_dict.values())
        
        return result

    @staticmethod
    def list_duplicates_of(seq, item):
        start_at = -1
        locs = []
        while True:
            try:
                loc = seq.index(item, start_at + 1)
            except ValueError:
                break
            else:
                locs.append(loc)
                start_at = loc
        return locs

    @staticmethod
    def evaluate(labelindex: List[Set[int]], truthtable: List[Set[int]]) -> dict:
        resultDict = dict()
        grpcnt = 1
        for labelset in labelindex:
            intersectList = [len(labelset.intersection(x)) for x in truthtable]
            maxintersect = max(intersectList)
            locs = ProcessTruth.list_duplicates_of(intersectList, maxintersect)
            ratio = [(intersectList[i] * 1.00) / (len(truthtable[i]) * 1.00) for i in locs]
            maxratio = max(ratio)
            resultDict[str(grpcnt)] = maxratio
            grpcnt += 1
        avgratio = mean(list(resultDict.values()))
        resultDict['mean'] = avgratio

        return resultDict

    @staticmethod
    def processTruthTable(truthtable: List[Set[int]]) -> List[Set[str]]:
        truthSetList = []
        for truthset in truthtable:
            combset = {str(tup[0]) + '+' + str(tup[1]) for tup in itertools.combinations(truthset, 2)}
            truthSetList.append(combset)

        return truthSetList

    @staticmethod
    def processLabelIndex(labelindex: List[Set[int]]) -> List[Set[str]]:
        return ProcessTruth.processTruthTable(labelindex)

    @staticmethod
    def evaluate_pairwise(labelindex: List[Set[int]], truthtable: List[Set[int]]) -> dict:
        resultdict = dict()
        grpcnt = 1
        # Create pairwise combinations for labelindex and truthtable
        labelindex_comb = [set(itertools.combinations(sorted(labelset), 2)) for labelset in labelindex]
        truthtable_comb = [set(itertools.combinations(sorted(truthset), 2)) for truthset in truthtable]
        for labelset in labelindex_comb:
            intersectList = [len(labelset.intersection(x)) for x in truthtable_comb]
            maxintersect = max(intersectList)
            locs = ProcessTruth.list_duplicates_of(intersectList, maxintersect)
            recall = [(intersectList[i] * 1.00) / (len(truthtable_comb[i]) * 1.00) for i in locs]
            maxrecall = max(recall)
            precision = [(intersectList[i] * 1.00) / (len(labelset) * 1.00) if len(labelset) > 0 else 0 for i in locs]
            maxprecision = max(precision)
            resultdict[str(grpcnt)] = {'p': maxprecision, 'r': maxrecall}
            grpcnt += 1
        for k, v in resultdict.items():
            if v['p'] is None or 'p' not in v.keys():
                v['p'] = 0
            if v['r'] is None or 'r' not in v.keys():
                v['r'] = 0
        meanprecision = mean(v['p'] for v in resultdict.values())
        meanrecall = mean(v['r'] for v in resultdict.values())
        resultdict['mean'] = {'p': meanprecision, 'r': meanrecall}
        return resultdict

    @staticmethod
    def printres(dataset: str, featureset: List[str], resultDict: dict):
        print('Precision of ' + dataset)
        print('Using feature ' + ', '.join(featureset) + '\n')
        for k, v in resultDict.items():
            if(k.isnumeric()):
                print('Cluster ' + k + ': ' + str(v) +'\n')
            elif(k == 'mean'):
                print('Mean: ' + str(v) + '\n')
        print('====================\n')

    @staticmethod
    def saveresandprint(fout: str, dataset: str, featureset: List[str], resultDict: dict, initial: bool):
        if(initial): openmethod = 'w+'
        else: openmethod = 'a'

        with open(fout, openmethod) as f:
            f.write('Precision of ' + dataset + '\n')
            f.write('Using feature ' + ', '.join(featureset) + '\n')
            for k, v in resultDict.items():
                if(k.isnumeric()):
                    f.write('Precision of cluster ' + k + ': ' + str(v) +'\n')
                elif(k == 'mean'):
                    f.write('Mean presicion: ' + str(v) + '\n')
            f.write('====================\n')
        print('Precision of ' + dataset)
        print('Using feature ' + ', '.join(featureset) + '\n')
        print('Mean presicion: ' + str(resultDict['mean']) + '\n')
        print('====================\n')

