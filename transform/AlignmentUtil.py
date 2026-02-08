import itertools
import re
from typing import Dict, List, Tuple

from LLMUtil.ResponseParser import RParser
from DataAccess.dataaccess import DataAccess

class AlignmentUtil:
    def __init__(self):
        pass

    @staticmethod
    def require_further_split(input_components: Dict[str, Dict[str, str]], output_components: Dict[str, Dict[str, str]], expand_input_key: Dict[str, Tuple[str]]) -> list:
        input_keys = set()
        output_keys = set()
        for i in input_components.keys():
            for ikey in input_components[i].keys():
                input_keys.add(ikey)
                if ikey in expand_input_key:
                    for ikey_e in expand_input_key[ikey]:
                        input_keys.add(ikey_e)
            for okey in output_components[i].keys():
                output_keys.add(okey)
        if not expand_input_key and not input_keys.intersection(output_keys):
            return list(input_keys.union(output_keys))
        comp_keys_for_further_split = list()
        num_require_further_split = 0
        for idx in input_components.keys():
            curr_input_components = input_components[idx]
            curr_output_components = output_components[idx]
            for okey, oval in curr_output_components.items():
                if okey in curr_input_components and oval != curr_input_components[okey] and not any(oval == v for v in curr_input_components.values()):
                    comp_keys_for_further_split.append(okey)
                    num_require_further_split += 1
                    break
                if okey in curr_input_components and curr_input_components[okey] and oval != curr_input_components[okey] and not any(oval == v for v in curr_input_components.values()):
                    comp_keys_for_further_split.append(okey)
                    num_require_further_split += 1
                    break
                if okey not in curr_input_components:
                    if okey and okey.strip()[-1].isdigit():
                        okey_normalize = RParser.key_normalize(okey)
                        for ikey in curr_input_components.keys():
                            outbreak = False
                            if ikey and ikey.strip()[-1].isdigit():
                                ikey_normalize = RParser.key_normalize(ikey)
                            if okey_normalize == ikey_normalize and oval != curr_input_components[ikey] and not any(oval == v for v in curr_input_components.values()):
                                comp_keys_for_further_split.append(okey)
                                comp_keys_for_further_split.append(ikey)
                                num_require_further_split += 1
                                outbreak = True
                                break
                        if outbreak:
                            break
                    else:
                        for ikey in curr_input_components.keys():
                            outbreak = False
                            if ikey and ikey.strip()[-1].isdigit():
                                ikey_normalize = RParser.key_normalize(ikey)
                            else:
                                ikey_normalize = ikey.strip()
                            if okey == ikey_normalize and oval != curr_input_components[ikey] and not any(oval == v for v in curr_input_components.values()):
                                comp_keys_for_further_split.append(okey)
                                comp_keys_for_further_split.append(ikey)
                                num_require_further_split += 1
                                outbreak = True
                                break
                        if outbreak:
                            break
        return comp_keys_for_further_split if (num_require_further_split * 1.0) / (DataAccess.NUM_EXAMPLES * 1.0) > 0.33 else []

    @staticmethod
    def align_output_component_type_to_input(output_components: Dict[str, Dict[str, str]], input_components: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
        alignment_count = dict()
        for idx in output_components.keys():
            curr_output_components = output_components[idx]
            curr_input_components = input_components[idx]
            for okey, oval in curr_output_components.items():
                for ikey, ival in curr_input_components.items():
                    normalized_oval = RParser.key_normalize(oval)
                    normalized_ival = RParser.key_normalize(ival)
                    if normalized_oval == normalized_ival and okey != ikey:
                        if okey not in alignment_count:
                            alignment_count[okey] = dict()
                        if ikey not in alignment_count[okey]:
                            alignment_count[okey][ikey] = 0
                        alignment_count[okey][ikey] += 1
        alignment = dict()
        for okey, ikeycount in alignment_count.items():
            for ikey, count in ikeycount.items():
                if count > DataAccess.NUM_EXAMPLES // 2:
                    if okey not in alignment:
                        alignment[okey] = list()
                    alignment[okey].append(ikey)
        return alignment
    
    @staticmethod
    def align_output_component_type_to_input_longest(output_components_orig: Dict[str, Dict[str, str]], input_components_orig: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
        alignment_count = dict()
        output_components = dict()
        input_components = dict()
        max_output_len = max(len(v) for v in output_components_orig.values())
        max_output_len_idx = [k for k, v in output_components_orig.items() if len(v) == max_output_len]
        output_components = {k: output_components_orig[k] for k in max_output_len_idx}
        input_components = {k: input_components_orig[k] for k in max_output_len_idx}
        for idx in output_components.keys():
            curr_output_components = output_components[idx]
            curr_input_components = input_components[idx]
            for okey, oval in curr_output_components.items():
                for ikey, ival in curr_input_components.items():
                    normalized_oval = RParser.key_normalize(oval)
                    normalized_ival = RParser.key_normalize(ival)
                    if normalized_oval == normalized_ival and okey != ikey:
                        if okey not in alignment_count:
                            alignment_count[okey] = dict()
                        if ikey not in alignment_count[okey]:
                            alignment_count[okey][ikey] = 0
                        alignment_count[okey][ikey] += 1
        alignment = dict()
        for okey, ikeycount in alignment_count.items():
            for ikey, count in ikeycount.items():
                if count > DataAccess.NUM_EXAMPLES // 2:
                    if okey not in alignment:
                        alignment[okey] = list()
                    alignment[okey].append(ikey)
        return alignment
    
    @staticmethod
    def _get_constant_(final_shortest_key: List[str], output_components: Dict[str, Dict[str, str]]) -> None:
        final_shortest_key_constant = dict()
        for key in final_shortest_key:
            for oc in output_components.values():
                if key in oc.keys():
                    val = oc[key]
                    if (key, val) not in final_shortest_key_constant.keys():
                        final_shortest_key_constant[(key, val)] = 0
                    final_shortest_key_constant[(key, val)] += 1
        constant_dict = dict()
        for (key, val), occurence in final_shortest_key_constant.items():
            if occurence >= DataAccess.NUM_EXAMPLES * 0.75:
                constant_dict[key] = "[CONST]" + val
        for i in range(len(final_shortest_key)):
            if final_shortest_key[i] in constant_dict.keys():
                final_shortest_key[i] = constant_dict[final_shortest_key[i]]
        return final_shortest_key

    @staticmethod
    def clean_final_shortest_key(shortest_key: List[str], expanded_keys: Dict[str, Tuple[str]], alignment: Dict[str, List[str]], examples: List[Dict[str, str]], input_components: Dict[str, Dict[str, str]], output_components: Dict[str, Dict[str, str]]) -> List[str]:
        if not alignment and not expanded_keys:
            final_shortest_key = AlignmentUtil._get_constant_(shortest_key, output_components)
            return final_shortest_key
        markremove = dict()
        for key in shortest_key:
            if key in alignment:
                markremove[key] = list()
                for aligned_key in alignment[key]:
                    if aligned_key not in shortest_key:
                        markremove[key].append(aligned_key)
        final_shortest_key = list()
        for key in shortest_key:
            if key not in markremove:
                if key in expanded_keys:
                    for expanded_key in expanded_keys[key]:
                        if expanded_key not in final_shortest_key:
                            final_shortest_key.append(expanded_key)
                else:
                    final_shortest_key.append(key)
            else:
                for aligned_key in markremove[key]:
                    if aligned_key not in final_shortest_key:
                        final_shortest_key.append(aligned_key)
        component_order = list()
        for i in output_components.keys():
            i = int(i)
            curr_output_components = output_components[i]
            curr_input_components = input_components[i]
            curr_output = examples[i]['output']
            curr_order = list()
            for key in final_shortest_key:
                if (key in curr_output_components and curr_output_components[key] in curr_output) or \
                   (key in curr_input_components and curr_input_components[key] in curr_output):
                    curr_order.append(key)
            curr_order = sorted(curr_order, key=lambda x: curr_output.index(curr_output_components[x]) if x in curr_output_components else curr_output.index(curr_input_components[x]) if x in curr_input_components else -1)
            component_order.append(curr_order)
        component_order_dict = dict()
        key_to_delete = set()
        for order_list in component_order:
            for key in order_list:
                if key not in component_order_dict:
                    component_order_dict[key] = 0
                component_order_dict[key] += 1
        for key in component_order_dict.keys():
            if component_order_dict[key] < DataAccess.NUM_EXAMPLES // 2:
                key_to_delete.add(key)
        final_ordered_keys_dict = dict()
        for order_list in component_order:
            order_list_new = tuple([key for key in order_list if key not in key_to_delete])
            if order_list_new not in final_ordered_keys_dict:
                final_ordered_keys_dict[order_list_new] = 0
            final_ordered_keys_dict[order_list_new] += 1
        final_shortest_key = None
        final_shortest_key_occurence = -1
        for key, occurence in final_ordered_keys_dict.items():
            if not final_shortest_key or occurence > final_shortest_key_occurence:
                final_shortest_key = list(key)
                final_shortest_key_occurence = occurence
        final_shortest_key = AlignmentUtil._get_constant_(final_shortest_key, output_components)

        return final_shortest_key
    
    @staticmethod
    def clean_final_longest_key(longest_key: List[str], expanded_keys: Dict[str, Tuple[str]], alignment: Dict[str, List[str]], examples: List[Dict[str, str]], input_components: Dict[str, Dict[str, str]], output_components: Dict[str, Dict[str, str]]) -> List[str]:
        if not alignment and not expanded_keys and len(examples) == 1:
            return longest_key
        if not alignment and not expanded_keys:
            final_longest_key = AlignmentUtil._get_constant_(longest_key, output_components)
            return final_longest_key
        markremove = dict()
        for key in longest_key:
            if key in alignment:
                markremove[key] = list()
                for aligned_key in alignment[key]:
                    if aligned_key not in longest_key:
                        markremove[key].append(aligned_key)
        final_longest_key = list()
        for key in longest_key:
            if key not in markremove:
                if key in expanded_keys:
                    for expanded_key in expanded_keys[key]:
                        if expanded_key not in final_longest_key:
                            final_longest_key.append(expanded_key)
                else:
                    final_longest_key.append(key)
            else:
                for aligned_key in markremove[key]:
                    if aligned_key not in final_longest_key:
                        final_longest_key.append(aligned_key)
        component_order = list()
        for i in output_components.keys():
            i = int(i)
            curr_output_components = output_components[i]
            curr_input_components = input_components[i]
            curr_output = examples[i]['output']
            curr_order = list()
            for key in final_longest_key:
                if (key in curr_output_components and curr_output_components[key] in curr_output) or \
                   (key in curr_input_components and curr_input_components[key] in curr_output):
                    curr_order.append(key)
            curr_order = sorted(curr_order, key=lambda x: curr_output.index(curr_output_components[x]) if x in curr_output_components else curr_output.index(curr_input_components[x]) if x in curr_input_components else -1)
            component_order.append(curr_order)
        component_order_dict = dict()
        key_to_delete = set()
        for order_list in component_order:
            for key in order_list:
                if key not in component_order_dict:
                    component_order_dict[key] = 0
                component_order_dict[key] += 1
        final_ordered_keys_dict = dict()
        for order_list in component_order:
            order_list_new = tuple([key for key in order_list if key not in key_to_delete])
            if order_list_new not in final_ordered_keys_dict:
                final_ordered_keys_dict[order_list_new] = 0
            final_ordered_keys_dict[order_list_new] += 1
        final_longest_key = None
        final_longest_key_occurence = -1
        for key, occurence in final_ordered_keys_dict.items():
            if not final_longest_key or occurence < final_longest_key_occurence:
                final_longest_key = list(key)
                final_longest_key_occurence = occurence
        final_longest_key = AlignmentUtil._get_constant_(final_longest_key, output_components)

        return final_longest_key
    
    @staticmethod
    def align_component_id(input_components: Dict[str, str], output_components: Dict[str, str]) -> Dict[str, str]:
        key_alignment = dict()
        for key, val in list(input_components.values())[0].items():
            if key and key.strip()[-1].isdigit():
                key_normalize = re.sub(r'[^a-zA-Z0-9]', ' ', key).lower().strip()
                key_normalize = re.sub(r'\d+$', '', key_normalize).strip()
                key_normalize = re.sub(r'[(\s]s[)\s]$', '', key_normalize).strip()
                key_normalize = re.sub(r'\s+', ' ', key_normalize)
                if any(key_normalize in k for k in list(output_components.values())[0].keys()):
                    for k in list(output_components.values())[0].keys():
                        if key_normalize in k and val == list(output_components.values())[0][k]:
                            key_alignment[k] = key
                            break
        return key_alignment
    
    @staticmethod
    def align_extra_split(output_keys: List[str], input_keys: List[str]) -> List[str]:
        keys_extra_split = list()
        for okey in output_keys:
            if okey and not okey.strip()[-1].isdigit():
                okey_normalize = re.sub(r'[^a-zA-Z0-9]', ' ', okey).lower().strip()
                okey_normalize = re.sub(r'\d+$', '', okey_normalize).strip()
                okey_normalize = re.sub(r'[(\s]s[)\s]$', '', okey_normalize).strip()
                okey_normalize = re.sub(r'\s+', ' ', okey_normalize)
                for ikey in input_keys:
                    if ikey and ikey.strip()[-1].isdigit():
                        ikey_normalize = re.sub(r'[^a-zA-Z0-9]', ' ', ikey).lower().strip()
                        ikey_normalize = re.sub(r'\d+$', '', ikey_normalize).strip()
                        ikey_normalize = re.sub(r'[(\s]s[)\s]$', '', ikey_normalize).strip()
                        ikey_normalize = re.sub(r'\s+', ' ', ikey_normalize)
                        if okey_normalize == ikey_normalize:
                            keys_extra_split.append(ikey)
                            break
        return keys_extra_split
    
    @staticmethod
    def process_input_more_detailed(input_components: Dict[str, Dict[str, str]], output_components: Dict[str, Dict[str, str]]) -> dict:
        key_alignment = dict()
        for idx in input_components.keys():
            curr_input_components = input_components[idx]
            curr_output_components = output_components[idx]
            for okey, oval in curr_output_components.items():
                if okey in curr_input_components.keys() and oval == curr_input_components[okey]:
                    continue
                ikeylist = list(curr_input_components.keys())
                for i in range(2, len(ikeylist) + 1):
                    for keycombo in itertools.combinations(ikeylist, i):
                        normalized_oval = RParser.key_normalize(oval)
                        normalized_oval_orig = normalized_oval
                        empty_too_early = False
                        keys_idx = dict()
                        for k in keycombo:
                            if not normalized_oval.strip():
                                empty_too_early = True
                                break
                            normalized_currival = RParser.key_normalize(curr_input_components[k])
                            if normalized_currival not in normalized_oval:
                                break
                            keys_idx[k] = normalized_oval_orig.index(normalized_currival)
                            normalized_oval = normalized_oval.replace(normalized_currival, '', 1)
                        if not empty_too_early and not normalized_oval.strip():
                            keycombo_ordered = tuple(sorted(list(keycombo), key=lambda x: keys_idx[x]))
                            if okey not in key_alignment:
                                key_alignment[okey] = dict()
                            if tuple(keycombo_ordered) not in key_alignment[okey]:
                                key_alignment[okey][tuple(keycombo_ordered)] = 0
                            key_alignment[okey][tuple(keycombo_ordered)] += 1
        final_key_alignment = dict()
        for key, keycombocountdict in key_alignment.items():
            for keycombo, count in keycombocountdict.items():
                if count > DataAccess.NUM_EXAMPLES // 2:
                    if key not in final_key_alignment:
                        final_key_alignment[key] = list()
                    final_key_alignment[key].append(keycombo)
        final_shorest_key_alignment = dict()
        for key, keycombos in final_key_alignment.items():
            currcount = 999
            if key not in final_shorest_key_alignment:
                final_shorest_key_alignment[key] = None
            for keycombo in keycombos:
                if len(keycombo) < currcount:
                    currcount = len(keycombo)
                    final_shorest_key_alignment[key] = keycombo
        return final_shorest_key_alignment

    @staticmethod
    def align_sem_with_synt(examples: List[Dict[str, str]], input_components: Dict[str, Dict[str, str]], output_components: Dict[str, Dict[str, str]], expanded_output_keys: Dict[str, Tuple[str]]) -> Dict[str, List[str]]:
        alignment = dict()
        alignment_dict = dict()
        expand_keys_all = [key for keytuple in expanded_output_keys.values() for key in keytuple] + list(expanded_output_keys.keys())
        for i in range(len(examples)):
            currinput = examples[i]['input']
            curroutput = examples[i]['output']
            currinput_components = input_components[i]
            curroutput_components = output_components[i]
            if currinput not in alignment_dict:
                alignment_dict[currinput] = dict()

            for key, val in curroutput_components.items():
                if key in expand_keys_all:
                    continue
                currval = val
                while currval:
                    found = False
                    for ikey, ival in currinput_components.items():
                        if ival and currval.startswith(ival) and ikey != key:
                            if key not in alignment_dict[currinput]:
                                alignment_dict[currinput][key] = [ikey]
                            else:
                                alignment_dict[currinput][key].append(ikey)
                            currval = currval.removeprefix(ival)
                            found = True
                            break
                    if not found:
                        conststr = currval
                        if key not in alignment_dict[currinput]:
                            alignment_dict[currinput][key] = [f"[CONST]{conststr}"]
                        else:
                            alignment_dict[currinput][key].append(f"[CONST]{conststr}")
                        currval = ""
        alignment_count = dict()
        for currinput, compdict in alignment_dict.items():
            for key, aligned_keys in compdict.items():
                if all(ikey.startswith("[CONST]") for ikey in aligned_keys):
                    continue
                key_aligned_keys = (key, tuple(aligned_keys))
                if key_aligned_keys not in alignment_count:
                    alignment_count[key_aligned_keys] = 1
                else:
                    alignment_count[key_aligned_keys] += 1
        alignment_selected = [key_aligned_keys for key_aligned_keys, count in alignment_count.items() if count > DataAccess.NUM_EXAMPLES // 2]
        for key_aligned_keys in alignment_selected:
            key, aligned_keys = key_aligned_keys
            if key not in alignment:
                alignment[key] = list()
            alignment[key].extend([keys for keys in aligned_keys if not keys.startswith("[CONST]")])
            alignment[key] = list(set(alignment[key]))
        return alignment


    @staticmethod
    def align_with_synt_str(examples: List[Dict[str, str]], input_components: Dict[str, Dict[str, str]], output_components: Dict[str, Dict[str, str]], shortest_keys: List[str], expanded_output_keys: Dict[str, Tuple[str]], aligned_keys: Dict[str, List[str]]) -> List[str]:
        alignment = list()
        for i, oc in output_components.items():
            tuned_alignment = list()
            curroutput = examples[i]['output']
            for skey in shortest_keys:
                if skey.startswith('[CONST]'):
                    continue
                dictval_to_check = None
                dictkey_to_check = None
                if any(skey in v for k, v in expanded_output_keys.items()):
                    for k, v in expanded_output_keys.items():
                        if skey in v:
                            dictval_to_check = input_components[i]
                            dictkey_to_check = skey
                            break
                elif skey in expanded_output_keys:
                    dictval_to_check_list = expanded_output_keys[skey]
                    for val in dictval_to_check_list:
                        if val in input_components[i].keys():
                            dictval_to_check = input_components[i]
                            dictkey_to_check = val
                            break
                elif any(skey in v for v in aligned_keys.values()):
                    for k, v in aligned_keys.items():
                        if skey in v:
                            dictval_to_check = input_components[i]
                            dictkey_to_check = skey
                            break
                else:
                    dictval_to_check = oc
                    dictkey_to_check = skey
                if dictkey_to_check in dictval_to_check and curroutput.startswith(dictval_to_check[dictkey_to_check]):
                    tuned_alignment.append(skey)
                    curroutput = curroutput.removeprefix(dictval_to_check[dictkey_to_check])
                else:
                    if dictkey_to_check in dictval_to_check and curroutput.find(dictval_to_check[dictkey_to_check]) != -1:
                        conststr = curroutput[:curroutput.index(dictval_to_check[dictkey_to_check])]
                        tuned_alignment.append("[CONST]" + conststr)
                        tuned_alignment.append(skey)
                        curroutput = curroutput.removeprefix(conststr)
                        curroutput = curroutput.removeprefix(dictval_to_check[dictkey_to_check])
            if curroutput:
                tuned_alignment.append("[CONST]" + curroutput)
            alignment.append(tuned_alignment)
        final_alignment = AlignmentUtil._check_tuned_alignment(alignment, examples)
        return final_alignment
        
    
    @staticmethod
    def _check_tuned_alignment(tuned_alignments: List[List[str]], examples: List[Dict[str, str]]) -> List[str]:
        currtc = None
        currtccount = 0
        variant_dict = dict()
        for tc in tuned_alignments:
            if not currtc:
                currtc = tc
                currtccount = 1
                continue
            if currtc and currtc == tc:
                currtccount += 1
                continue
            if currtc and set(currtc) >= set(tc):
                currtccount += 1
                continue
            variant_dict[tuple(tc)] = variant_dict[tuple(tc)] + 1 if tuple(tc) in variant_dict else 1
        if not variant_dict:
            return currtc
        if len(variant_dict) == 1:
            return list(list(variant_dict.keys())[0])
        elif len(variant_dict) > 1:
            maxlength = max(list(variant_dict.values()))
            valid_variants = [k for k, v in variant_dict.items() if v == maxlength]
            if len(valid_variants) == 1:
                return list(valid_variants[0])
            else:
                longest_variant_length = max([len(k) for k in valid_variants])
                longest_variants = [k for k in valid_variants if len(k) == longest_variant_length]
                if len(longest_variants) == 1:
                    return list(longest_variants[0])
                else:
                    const_count = []
                    for variant in longest_variants:
                        count = 0
                        for key in variant:
                            if key.startswith("[CONST]"):
                                count += 1
                        const_count.append(count)
                    min_const_count = min(const_count)
                    return list(longest_variants[const_count.index(min_const_count)])