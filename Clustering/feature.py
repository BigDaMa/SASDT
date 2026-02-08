from typing import Dict, List, Union
import os
import statistics
import pandas as pd 


AVAIL_FEAT = ['LengthFeature', 'NumTokensFeature', 'RatioAlphaFeature', 'RatioNumericFeature', 'RatioAlphanumericFeature', 'RatioNonAlnumFeature', 'NonAlnumOffsetFeature', 'TypeNonAlnumCharFeature']
__FEAT_DIR__ = "data_feature"

class Feature:

    @staticmethod
    def __normalize_list__(raw: List[Union[int, float]]) -> List[Union[int, float]]:
        return [float(i) / sum(raw) for i in raw]

    @staticmethod
    def __min_max_normalizer__(orig: List[Union[int, float]]) -> List[Union[int, float]]:
        orig = [float(x) for x in orig]
        if(len(orig) == 0): return [0.0]
        if(len(orig) == 1): return [0.0]
        minval = min(orig)
        maxval = max(orig)
        if(maxval == 0 and minval == 0): return [0.0]
        if(maxval == minval): return [0.0]
        normalized = []
        for num in orig:
            normed = ((num - minval) * 1.00) / ((maxval - minval) * 1.00)
            normalized.append(normed)
        return normalized

    @staticmethod
    def __min_max_abs_normalizer__(orig: List[Union[int, float]]) -> List[Union[int, float]]:
        orig = [float(x) for x in orig]
        orig_abs = [abs(x) for x in orig]
        if(len(orig) == 0): return [0.0]
        if(len(orig) == 1): return [0.0]
        minval = min(orig_abs)
        maxval = max(orig_abs)
        if(maxval == 0 and minval == 0): return [0.0]
        if(maxval == minval): return [0.0]

        normalized = []
        for num in orig:
            if(num > 0): sign = 1
            else: sign = -1
            normed = ((abs(num) - minval) * sign * 1.00) / ((maxval - minval) * 1.00)
            normalized.append(normed)
        return normalized

    @staticmethod
    def validate_feature(s: List[str]) -> bool:
        if (s in AVAIL_FEAT):
            return True
        return False

    @staticmethod
    def LengthFeature(s: str) -> float:
        s = str(s)
        return len(s)

    @staticmethod
    def NumTokensFeature(s: str) -> float:
        s = str(s)
        tokens = [token for token in s.split() if token and (token.isalnum() or len(token) == 1 and not token.isspace())]
        return len(tokens)

    @staticmethod
    def RatioAlphaFeature(s: str) -> float:
        s = str(s)
        return (float)(len([char for char in s if char.isalpha()]) * 1.00) / (len(s) * 1.00)

    @staticmethod
    def RatioNumericFeature(s: str) -> float:
        s = str(s)
        return (float)(len([char for char in s if char.isnumeric()]) * 1.00) / (len(s) * 1.00)

    @staticmethod
    def RatioAlphanumericFeature(s: str) -> float:
        s = str(s)
        return (float)(len([char for char in s if char.isalnum()]) * 1.00) / (len(s) * 1.00)

    @staticmethod
    def RatioNonAlnumFeature(s: str) -> float:
        s = str(s)
        return (float)(len([char for char in s if not char.isalnum()]) * 1.00) / (len(s) * 1.00)

    @staticmethod
    def NonAlnumOffsetFeature(s: str) -> float:
        s = str(s)
        nonalnum = {char for char in s if not char.isalnum()}
        nonalnumcharoffset = []
        for char, i in zip(s, range(len(s))):
            offset = 0
            if(char in nonalnum):
                if(i == 0):
                    cnt = 1
                    while(i + cnt < len(s) and not s[i + cnt].isalnum()): 
                        offset += 1
                        cnt += 1
                    offset += 1
                elif (i == (len(str(s)) - 1)):
                    cnt = 1
                    while(i - cnt >= 0 and not s[i - cnt].isalnum()):
                        offset -= 1
                        cnt += 1
                    offset -= 1
                else:
                    cnt = 1
                    while(i + cnt < len(s) and not s[i + cnt].isalnum()): 
                        offset += 1
                        cnt += 1
                    offset += 1
                    cnt = 1
                    while(i - cnt >= 0 and not s[i - cnt].isalnum()):
                        offset -= 1
                        cnt += 1
                    offset -= 1
                nonalnumcharoffset.append(offset)
                nonalnumcharoffset = Feature.__min_max_abs_normalizer__(nonalnumcharoffset)
        return  0.0 if len(nonalnumcharoffset) <= 1 else (float)(statistics.stdev(nonalnumcharoffset))

    @staticmethod
    def TypeNonAlnumCharFeature(s: str) -> str:
        s = str(s)
        nonalnum = set()
        for char in s:
            if(not char.isalnum() and not char.isspace()):
                if char == '"':
                    char = 'd'
                    nonalnum.add(char)
                else:
                    nonalnum.add(char)
        if(len(nonalnum) == 0): final = ''
        elif(len(nonalnum) == 1): final = set(list(nonalnum)[0])
        else: final = ' '.join(sorted(nonalnum))
        return final

    @staticmethod
    def getFeature(s: str, FeatureList: List[str]) -> list:
        featureVector = []

        for feat in FeatureList:
            if(Feature.validate_feature(feat)):
                method = getattr(Feature, feat)
                if(feat == 'TypeNonAlnumCharFeature'):
                    featureVector.append((str)(method(s)))
                else:
                    featureVector.append((float)(method(s)))

        return featureVector
    
    @staticmethod
    def save_features_to_file(features: List[list], file: str) -> None:
        features_df = pd.DataFrame(features)
        features_df.to_csv(os.path.join(__FEAT_DIR__, file), header=["OrigStr"] + AVAIL_FEAT, index=None)

    @staticmethod
    def get_feature_for_strs(data: List[str], file: str, reset = False) -> Dict[str, List[float]]:
        if not reset and os.path.exists(os.path.join(__FEAT_DIR__, file)):
            features_df = pd.read_csv(os.path.join(__FEAT_DIR__, file), header=0)
            return features_df.to_dict(orient="list")

        features = []
        feature_dict = {}
        for s in data:
            features.append([s] + Feature.getFeature(s, AVAIL_FEAT))
            feature_dict[s] = features[1:]

        Feature.save_features_to_file(features, file)
        return feature_dict