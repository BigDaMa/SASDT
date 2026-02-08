import itertools
from typing import Dict, List, Tuple
import re

from Propagate.regexutil import RegExUtil as REU
from Propagate.position import Position, AbsPosPosition, RegExPosPosition
from Propagate.positionop import get_str_from_pos
from Propagate.tokenutil import is_valid_extraction
from Utils.CacheUtil import cache_result, perfect_operator_override
from Utils.config import Config

def dedup(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def WitnessPosition(spec: str, currspec: str) -> List[Tuple[Position, Position]]:
    posList = list()
    absposleft, absposright = WitnessAbsPos(spec, currspec)
    regexleft, regexright = WitnessRegExPos(spec, currspec, absposleft)
    for startpos, endpos in zip(absposleft.keys(), absposright.keys()):
        leftpos = list()
        rightpos = list()
        if startpos in absposleft:
            leftpos.extend(absposleft[startpos])
        if startpos in regexleft:
            leftpos.extend(regexleft[startpos])
        if endpos in absposright:
            rightpos.extend(absposright[endpos])
        if endpos in regexright:
            rightpos.extend(regexright[endpos])
        if leftpos and rightpos:
            posList.extend(list(itertools.product(leftpos, rightpos)))
    return posList
            
def WitnessAbsPos(spec: str, currspec: str) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    absposleft = _WitnessAbsPosLeft(spec, currspec)
    absposright = _WitnessAbsPosRight(spec, currspec, absposleft)
    return absposleft, absposright

def _WitnessAbsPosLeft(spec: str, currspec: str) -> Dict[int, List[AbsPosPosition]]:
    idxList = dict()
    idx = 0
    while(idx != -1):
        idx = spec.find(currspec, idx)
        if idx != -1:
            if idx not in idxList:
                idxList[idx] = []
            idxList[idx].append(AbsPosPosition(idx))
            idxList[idx].append(AbsPosPosition(idx - len(spec)))
            idx = idx + len(currspec)
    return idxList

def _WitnessAbsPosRight(spec: str, currspec: str, leftidx: Dict[int, List[AbsPosPosition]]) -> Dict[int, List[AbsPosPosition]]:
    idxList = dict()
    for idx, values in leftidx.items():
        if idx not in idxList:
            idxList[idx] = []
        for val in values:
            currval = val.idx if val.idx >= 0 else len(spec) + val.idx

            if currval >= 0 and currval + len(currspec) <= len(spec):
                currvalright = currval + len(currspec)
                idxList[idx].append(AbsPosPosition(currvalright))
                if currvalright - len(spec) < 0:
                    idxList[idx].append(AbsPosPosition(currvalright - len(spec)))
                else:
                    idxList[idx].append(AbsPosPosition(None))

    return idxList

def WitnessRegExPos(spec: str, currspec: str, absposLeft: Dict[int, List[AbsPosPosition]]) -> Tuple[Dict[int, List[RegExPosPosition]], Dict[int, List[RegExPosPosition]]]:
    idxList = list()
    lidxmap = _WitnessRegExPosLeft(spec, currspec)
    for aidx in absposLeft.keys():
        if aidx in lidxmap.keys():
            lidxmap[aidx].extend(absposLeft[aidx])
        else:
            lidxmap[aidx] = absposLeft[aidx]
    ridxmap = _WitnessRegExPosRight(spec, currspec, lidxmap)
    return lidxmap, ridxmap

def _WitnessRegExPosLeft(spec: str, currspec: str) -> Dict[int, List[RegExPosPosition]]:
    """
    Return the position of a character in a string that matches a regex.
    """
    regexList = REU.regexes
    idxmap = dict()
    absposleft = [occurance.start() for occurance in re.finditer(re.escape(currspec), spec)]
    for r in regexList:
        matchstr = re.findall(r, spec)
        nummatches = len(matchstr)
        i = 0
        for regocc in re.finditer(r, spec):
            if(regocc.start() in absposleft):
                if regocc.start() not in idxmap.keys():
                    idxmap[regocc.start()] = []
                idxmap[regocc.start()].append(RegExPosPosition(r, i, "L"))
                idxmap[regocc.start()].append(RegExPosPosition(r, i - nummatches, "L"))
            elif(regocc.end() in absposleft):
                if regocc.end() not in idxmap.keys():
                    idxmap[regocc.end()] = []
                idxmap[regocc.end()].append(RegExPosPosition(r, i, "R"))
                idxmap[regocc.end()].append(RegExPosPosition(r, i - nummatches, "R"))
            i += 1
    return idxmap

def _WitnessRegExPosRight(spec: str, currspec: str, lidxmap: Dict[int, List[RegExPosPosition]]) -> Dict[int, List[RegExPosPosition]]:
    regexList = REU.regexes
    idxmap = dict()
    for lidx in lidxmap.keys():
        absposright = lidx + len(currspec)
        for r in regexList:
            matchstr = re.findall(r, spec)
            nummatches = len(matchstr)
            i = 0
            for regocc in re.finditer(r, spec):
                if(regocc.start() == absposright):
                    if lidx not in idxmap.keys():
                        idxmap[lidx] = []
                    idxmap[lidx].append(RegExPosPosition(r, i, "L"))
                    idxmap[lidx].append(RegExPosPosition(r, i - nummatches, "L"))
                elif(regocc.end() == absposright):
                    if absposright not in idxmap.keys():
                        idxmap[absposright] = []
                    idxmap[absposright].append(RegExPosPosition(r, i, "R"))
                    idxmap[absposright].append(RegExPosPosition(r, i - nummatches, "R"))
                i += 1

    return idxmap

@perfect_operator_override()
@cache_result()
def get_pos_dict(sem_components: Dict[int, Dict[str, str]], query_strs: List[str], query_str_idx: List[int], file: str) -> Dict[int, Dict[str, List[Tuple[Position, Position]]]]:
    comp_pos_dict = dict()
    for idx, component_dict in sem_components.items():
        if idx not in comp_pos_dict:
            comp_pos_dict[idx] = dict()
        orig_str = query_strs[query_str_idx.index(idx)]
        for compkey, compstr in component_dict.items():
            if compkey not in comp_pos_dict[idx]:
                comp_pos_dict[idx][compkey] = list()
            pos = WitnessPosition(orig_str, compstr)
            comp_pos_dict[idx][compkey].extend(pos)
    return comp_pos_dict

def get_common_operators(pos_lists: List[List[Tuple[Position, Position]]]) -> List[Tuple[Position, Position]]:
    if not pos_lists:
        return []
    common_pos = set(pos_lists[0])
    for pos_list in pos_lists[1:]:
        common_pos.intersection_update(set(pos_list))
    return list(common_pos)

@cache_result()
def validate_pos_dict(mapped_clusters: List[Dict[int, str]], comp_pos_dict: Dict[int, Dict[str, List[Tuple[Position, Position]]]], sem_components_selected: Dict[int, Dict[str, str]]) -> Dict[str, List[Tuple[Position, Position]]]:
    grouped_pos_list = list()
    for cluster in mapped_clusters:
        curr_pos_list = list()
        for idx in comp_pos_dict.keys():
            if idx in cluster.keys():
                curr_pos_list.append(idx)
        if curr_pos_list:
            grouped_pos_list.append(curr_pos_list)
        else:
            grouped_pos_list.append([])
    com_pos_dict_grouped = list()
    for idx_list in grouped_pos_list:
        if len(idx_list) == 0:
            com_pos_dict_grouped.append(dict())
            continue
        elif len(idx_list) == 1:
            com_pos_dict_grouped.append(comp_pos_dict[idx_list[0]])
            continue
        curr_pos_dict = comp_pos_dict[idx_list[0]]
        for idx in idx_list[1:]:
            next_pos_dict = comp_pos_dict[idx]
            new_pos_dict = dict()
            for compkey in curr_pos_dict.keys():
                if compkey in next_pos_dict.keys():
                    common_pos = get_common_operators([curr_pos_dict[compkey], next_pos_dict[compkey]])
                    if common_pos:
                        new_pos_dict[compkey] = common_pos
                    else:
                        new_pos_dict[compkey] = curr_pos_dict[compkey]
            curr_pos_dict = new_pos_dict
        com_pos_dict_grouped.append(curr_pos_dict)
    valid_comp_pos_dict = dict()
    for idx in range(len(com_pos_dict_grouped)):
        valid_comp_pos_dict[idx] = dict()
        for compkey, pos in com_pos_dict_grouped[idx].items():
            valid_comp_pos_dict[idx][compkey] = [True for _ in pos]
    for clusteridx, (cluster, pos_dict) in enumerate(zip(mapped_clusters, com_pos_dict_grouped)):
        for idx, orig_str in cluster.items():
            if idx in sem_components_selected.keys():
                continue
            else:
                for compkey, comppos in pos_dict.items():
                    for pos in comppos:
                        extracted_str = get_str_from_pos(orig_str, pos)
                        try: 
                            extraction_valid = is_valid_extraction(orig_str, extracted_str, pos[0].get_index(orig_str), pos[1].get_index(orig_str))
                        except Exception as e:
                            extraction_valid = False
                        if extraction_valid:
                            if valid_comp_pos_dict[clusteridx][compkey][comppos.index(pos)] == True:
                                valid_comp_pos_dict[clusteridx][compkey][comppos.index(pos)] = 1
                            elif isinstance(valid_comp_pos_dict[clusteridx][compkey][comppos.index(pos)], int):
                                valid_comp_pos_dict[clusteridx][compkey][comppos.index(pos)] += 1
                        else:
                            pass
    final_comp_pos_dict = dict()
    for idx, pos_dict_key in valid_comp_pos_dict.items():
        if idx not in final_comp_pos_dict.keys():
            final_comp_pos_dict[idx] = dict()
        for compkey, mask in pos_dict_key.items():
            if isinstance(mask, list):
                max_count = -1
                for m in mask:
                    if isinstance(m, int) and m > max_count:
                        max_count = m
                if max_count > 0:
                    final_comp_pos_dict[idx][compkey] = [pos for i, pos in enumerate(com_pos_dict_grouped[idx][compkey]) if isinstance(mask[i], int) and mask[i] == max_count]
                else:
                    final_comp_pos_dict[idx][compkey] = []
            else:
                final_comp_pos_dict[idx][compkey] = []
    return final_comp_pos_dict