import itertools
from typing import Dict, List, Tuple
from Propagate.position import Position, RegExPosPosition
from Propagate.positionop import get_str_from_pos
from Utils.CacheUtil import cache_result
from Propagate.tokenutil import is_valid_extraction

class Propagate:

    @staticmethod
    def can_assemble_without_overlap(original: str, string_list: List[str]) -> bool:
        if not string_list:
            return True
        
        non_empty_strings = [s for s in string_list if s]
        if not non_empty_strings:
            return True
        
        first_string = non_empty_strings[0]
        start_positions = []
        start = 0
        while True:
            pos = original.find(first_string, start)
            if pos == -1:
                break
            start_positions.append(pos)
            start = pos + 1
        
        if not start_positions:
            return False
        
        for start_pos in start_positions:
            if Propagate._can_assemble_from_position(original, non_empty_strings, start_pos):
                return True
        
        return False
    
    @staticmethod
    def _can_assemble_from_position(original: str, string_list: List[str], start_pos: int) -> bool:
        current_pos = start_pos
        
        for i, string in enumerate(string_list):
            found_pos = original.find(string, current_pos)
            
            if found_pos == -1:
                return False
            
            if i == 0:
                if found_pos != start_pos:
                    return False
            else:
                if found_pos < current_pos:
                    return False
            
            current_pos = found_pos + len(string)
        
        return True

    @staticmethod
    @cache_result()
    def propagate(clusters: List[Dict[int, str]], positions: Dict[int, Dict[str, List[Tuple[Position, Position]]]], sem_components_selected: Dict[int, Dict[str, str]]) -> Dict[int, Dict[str, str]]:
        propagationDict = {}
        propagationCandidateDict = {}
        for clusteridx in positions.keys():
            currcluster = clusters[clusteridx]
            currpositions = positions[clusteridx]
            for idx, querystr in currcluster.items():
                if idx in sem_components_selected.keys():
                    propagationDict[idx] = sem_components_selected[idx]
                    continue
                if idx not in propagationDict.keys():
                    propagationDict[idx] = dict()
                if idx not in propagationCandidateDict.keys():
                    propagationCandidateDict[idx] = {}
                for semtype, poslist in currpositions.items():
                    candidate = dict()
                    for pospair in poslist:
                        idxcheck = True
                        extractedstr = get_str_from_pos(querystr, pospair)
                        startidx = pospair[0].get_index(querystr)
                        endidx = pospair[1].get_index(querystr)
                        if isinstance(pospair[0], RegExPosPosition) and startidx is None:
                            idxcheck = False
                        if isinstance(pospair[1], RegExPosPosition) and endidx is None:
                            idxcheck = False
                        # try:
                        if idxcheck and extractedstr and is_valid_extraction(querystr, extractedstr, startidx, endidx) and extractedstr not in candidate.keys():
                            candidate[extractedstr] = 1
                        elif idxcheck and extractedstr and is_valid_extraction(querystr, extractedstr, startidx, endidx) and extractedstr in candidate.keys():
                            candidate[extractedstr] += 1
                    if len(candidate) > 1:
                        dellist = []
                        for canstr in candidate.keys():
                            if not canstr[0].isalnum() or not canstr[-1].isalnum():
                                dellist.append(canstr)
                        for dels in dellist:
                            del candidate[dels]
                    candidate = dict(sorted(candidate.items(), key=lambda item: item[1], reverse=True))
                    propagationCandidateDict[idx][semtype] = candidate
            for idx in propagationCandidateDict.keys():
                for semtype in propagationCandidateDict[idx].keys():
                    if len(propagationCandidateDict[idx][semtype]) == 1:
                        propagationDict[idx][semtype] = list(propagationCandidateDict[idx][semtype].keys())[0]
                    else:
                        propagationDict[idx][semtype] = ""
            for idx in propagationDict.keys():
                if all(semstr for semstr in propagationDict[idx].values()):
                    continue
                selectedstr = list()
                origstr = ""
                for cluster in clusters:
                    if idx in cluster.keys():
                        origstr = cluster[idx]
                        break
                check_dict = dict()
                list_idx_dict = dict()
                list_idx = 0
                for semtype in propagationDict[idx].keys():
                    if propagationDict[idx][semtype]:
                        selectedstr.append(propagationDict[idx][semtype])
                    else:
                        selectedstr.append("[{}]".format(semtype))
                        list_idx_dict["[{}]".format(semtype)] = list_idx
                        check_dict["[{}]".format(semtype)] = propagationCandidateDict[idx][semtype]
                    list_idx += 1
                for strs in itertools.product(*check_dict.values()):
                    temp_selectedstr = selectedstr.copy()
                    for i in range(len(strs)):
                        temp_selectedstr[list_idx_dict[list(check_dict.keys())[i]]] = strs[i]
                    if Propagate.can_assemble_without_overlap(origstr, temp_selectedstr):
                        for i in range(len(strs)):
                            propagationDict[idx][list(check_dict.keys())[i][1:-1]] = strs[i]
                        break
        return propagationDict