from itertools import combinations
from typing import List
import os
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN
from denseclus import DenseClus

from Clustering.feature import Feature
from Clustering.processtruth import ProcessTruth
import time
import warnings

warnings.filterwarnings("ignore")

ActiveFeatureList = 'flist.txt'
__DATA_DIR__ = 'data'
__FEAT_DIR__ = 'data_feature'
__TRUTH_DIR__ = 'truth'
__RES_DIR__ = 'result'
__NOM_VAR__ = ['TypeNonAlnumCharFeature']

class Cluster:

    @staticmethod
    def getUniqueFeatures(clusteralg: str = 'DBSCAN') -> List[str]:
        featureList = []
        with open(os.path.join('.', ActiveFeatureList), 'r') as f:
            for line in f:
                if(not (clusteralg == 'DenseClus' or clusteralg == 'all') and line.strip() == 'TypeNonAlnumCharFeature'):
                    continue
                featureList.append(line.strip())
        return featureList

    @staticmethod
    def getFeatureList(clusteralg: str = 'DBSCAN') -> List[str]:
        featureList = Cluster.getUniqueFeatures(clusteralg)
        fullFeatureList = []

        if clusteralg == 'DenseClus':
            fullFeatureList = [featureList]
        else:
            for r in range(1, len(featureList) + 1):
                comb = combinations(featureList, r)
                fullFeatureList.extend([list(c) for c in comb])

        return fullFeatureList

    @staticmethod
    def getFeatureIndexDict(clusteralg: str = 'DBSCAN') -> dict:
        featureList = Cluster.getUniqueFeatures(clusteralg)
        featureIndexDict = {x: featureList.index(x) for x in featureList}

        return featureIndexDict

    @staticmethod
    def featureCleanAndSave(file: str, df: pd.DataFrame = None) -> None:
        if(df is None and type(file) == str):
            data = pd.read_csv(file, header = None, skipinitialspace = True)
        else:
            data = df
        AVAIL_FEAT = Cluster.getUniqueFeatures()
        if('LengthFeature' in AVAIL_FEAT):
            leng_col_header = AVAIL_FEAT.index('LengthFeature')
            leng_col = data[leng_col_header].values
            leng_col = Feature.__min_max_normalizer__(leng_col)
            data[leng_col_header] = leng_col

        data.to_csv(file, header = False, index = False)

    @staticmethod
    def getFeaturesAndCache(fin: str, fout: str) -> None:
        fi = pd.read_csv(fin, header = None, usecols = [0])
        featureList = Cluster.getUniqueFeatures('all')

        featureVec = pd.DataFrame(columns = range(len(featureList)))
        for val in fi.iloc[:, 0].values:
            featureVector = Feature.getFeature(val, featureList)
            featureVec.loc[len(featureVec)] = featureVector
        Cluster.featureCleanAndSave(fout, featureVec)


    @staticmethod
    def loadFeaturesAsDF(fin: str) -> pd.DataFrame:
        file = pd.read_csv(fin, header = 0)
        available_features = Cluster.getUniqueFeatures()
        file = file[available_features]

        return file

    @staticmethod
    def runCluster(file: str, clusteralg = 'DBSCAN') -> List[List[int]]:
        featureFile = file[:-4] + '_features.csv'

        uniqFeatures = Cluster.getUniqueFeatures(clusteralg)
        featureCombination = Cluster.getFeatureList(clusteralg)
        featureIndexDict = Cluster.getFeatureIndexDict(clusteralg)

        result_p = pd.DataFrame(columns = uniqFeatures, index = [file] + ['best'])
        result_r = pd.DataFrame(columns = uniqFeatures, index = [file] + ['best'])
        result_f1 = pd.DataFrame(columns = uniqFeatures, index = [file] + ['best'])


        starttime = time.time()
        print('Processing ' + file + '...')
        initial = True
        featureVec = Cluster.loadFeaturesAsDF(os.path.join(__FEAT_DIR__, featureFile))

        featurekey = ''
        for feat in uniqFeatures:
            featurekey += str(featureIndexDict[feat])
        featureColumn = featureVec[uniqFeatures]
        if clusteralg == 'DBSCAN':
            clusterer = DBSCAN(eps = 0.1, min_samples = 4)
        elif clusteralg == 'HDBSCAN':
            clusterer = HDBSCAN(min_cluster_size = 3, min_samples = 5)
        elif clusteralg == 'DenseClus':
            umap_combine_method = 'intersection_union_mapper'
            n_neighbors = 3
            min_samples = 3
            min_cluster_size = 5
            clusterer = DenseClus(
                umap_combine_method=umap_combine_method,
                n_neighbors = 3,
                min_samples = 3,
                min_cluster_size = 5
            )
        if clusteralg == 'DenseClus':
            try:
                clusterer.fit(featureColumn)
            except Exception as e:
                print(f"Error fitting DenseClus on {file} with features {uniqFeatures}: {e}")
                print(featureColumn)
                return
        else:
            clusterer.fit(featureColumn.values)
        if clusteralg == 'DenseClus':
            labels = clusterer.score()
        else:
            labels = clusterer.labels_

        truthfile = file[:-4] + '_processed.csv'
        truthtable = ProcessTruth.process(os.path.join(__TRUTH_DIR__, truthfile))

        labelIndex = ProcessTruth.group_indices_by_label(labels)
        result = ProcessTruth.evaluate_pairwise(labelIndex, truthtable)
        initial = False
        result_p.loc[file, featurekey] = result['mean']['p']
        result_r.loc[file, featurekey] = result['mean']['r']
        result_f1.loc[file, featurekey] = 0 if (result['mean']['p'] + result['mean']['r']) == 0 else \
            (2.00 * result['mean']['p'] * result['mean']['r']) / (1.00 * (result['mean']['p'] + result['mean']['r']))
        endtime = time.time()
        print('Time taken for ' + file + ': ' + str(endtime - starttime) + ' seconds')
        for i in range(len(result_p.columns)):
            result_p.iloc[-1, i] = result_p.iloc[:-1, i].mean()
            result_r.iloc[-1, i] = result_r.iloc[:-1, i].mean()
            result_f1.iloc[-1, i] = result_f1.iloc[:-1, i].mean()

        maxavg_p = result_p.loc['best'].max()
        maxavg_r = result_r.loc['best'].max()
        maxavg_f1 = result_f1.loc['best'].max()
        maxlist = list()
        for idx in result_f1.index:
            maxlist.append(result_f1.loc[idx].max())
        for i in range(len(maxlist)):
            for j in range(len(result_f1.iloc[i, :].values)):
                if(result_f1.iloc[i, j] == maxlist[i]):
                    colname = result_f1.columns[j]
            collist = [uniqFeatures[int(cn)] for cn in colname]
        colname = ''
        for i in range(len(result_f1.loc['best'].values)):
            if(result_f1.iloc[-1, i] == maxavg_f1):
                colname = result_f1.columns[i]
                break

        collist = [uniqFeatures[int(i)] for i in colname]

        return labelIndex