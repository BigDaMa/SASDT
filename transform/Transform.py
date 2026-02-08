from typing import Dict, List, Tuple
import re

from transform.AlignmentUtil import AlignmentUtil
from LLMUtil.ResponseParser import RParser

class Transform:
    @staticmethod
    def transform(example_input_str: Dict[int, str], input_components: Dict[str, Dict[str, str]], example_output_str: Dict[int, str], output_components: Dict[str, Dict[str, str]], query_components: Dict[str, Dict[str, str]], examples: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        pass
        example_output_str = {k: v for k, v in zip(example_input_str.keys(), example_output_str.values())}
        if any(set([re.sub(r'\d+$', '', y) for y in x.keys()]) != set([y for y in x.keys()]) for x in input_components.values()) or \
       any(set([re.sub(r'\d+$', '', y) for y in x.keys()]) != set([y for y in x.keys()]) for x in output_components.values()):

            key_alignment = AlignmentUtil.align_component_id(input_components, output_components)
        else:
            key_alignment = dict()

        expand_input_key = AlignmentUtil.process_input_more_detailed(input_components, output_components)

        aligned_keys1 = AlignmentUtil.align_output_component_type_to_input(output_components, input_components)
        aligned_keys2 = AlignmentUtil.align_sem_with_synt(examples, input_components, output_components, expand_input_key)
        aligned_keys = dict()

        if not aligned_keys1 and not aligned_keys2:
            pass
        elif aligned_keys1 and not aligned_keys2:
            aligned_keys = aligned_keys1
        elif not aligned_keys1 and aligned_keys2:
            aligned_keys = aligned_keys2
        else:
            for key, val in aligned_keys1.items():
                if key not in aligned_keys:
                    aligned_keys[key] = val
                if key in aligned_keys2:
                    aligned_keys[key] = list(set(aligned_keys[key]) | set(aligned_keys2[key]))


        longest_final_key_set = list()
        key_count_dict = dict()
        for key, val in output_components.items():
            for skey in val.keys():
                if skey not in key_count_dict:
                    key_count_dict[skey] = 0
                key_count_dict[skey] += 1
        for key, val in key_count_dict.items():
            if expand_input_key and key in expand_input_key:
                for skey in expand_input_key[key]:
                    if skey in aligned_keys.keys() and aligned_keys[skey] and aligned_keys[skey][0] not in longest_final_key_set:
                        longest_final_key_set.append(aligned_keys[skey][0])
                    elif skey not in longest_final_key_set:
                        longest_final_key_set.append(skey)
            elif key in aligned_keys.keys() and aligned_keys[key] and aligned_keys[key][0] not in longest_final_key_set:
                longest_final_key_set.append(key)
            elif key not in longest_final_key_set:
                longest_final_key_set.append(key)

        longest_final_key_set = AlignmentUtil.clean_final_longest_key(longest_final_key_set, expand_input_key, aligned_keys, examples, input_components, output_components)

        longest_final_key_set = Transform.sort_keys_by_appearance_order(longest_final_key_set, example_output_str, output_components)

        longest_final_key_set = AlignmentUtil.align_with_synt_str(examples, input_components, output_components, longest_final_key_set, expand_input_key, aligned_keys)
        
        transformations = dict()
        for key, valdict in query_components.items():
            transformation = ""
            currcomponentkey = ""
            valid = valdict["id"]
            val = valdict["components"]
            i_to_o_key_alignment = AlignmentUtil.align_extra_split(longest_final_key_set, list(val.keys()))

            if isinstance(val, dict):
                for skey in longest_final_key_set:
                    if skey.startswith("[CONST]"):
                        currcomponentkey = skey
                    elif skey in key_alignment.keys() and key_alignment[skey] in val.keys() and skey not in i_to_o_key_alignment:
                        currcomponentkey = key_alignment[skey]
                    elif skey in aligned_keys.keys() and aligned_keys[skey]:
                        for curralignedkey in aligned_keys[skey]:
                            if curralignedkey in val.keys():
                                currcomponentkey = curralignedkey
                                break
                    else:
                        currcomponentkey = skey
                    currtransformation = ""
                    output_aligned_keys = list()
                    for ikey in i_to_o_key_alignment:
                        if ikey and ikey.strip()[-1].isdigit() or ikey.strip().endswith('(s)'):
                            ikey_normalize = RParser.key_normalize(ikey)
                            if ikey_normalize == currcomponentkey:
                                if currcomponentkey not in output_aligned_keys:
                                    output_aligned_keys.append(currcomponentkey)
                                output_aligned_keys.append(ikey)
                    if currcomponentkey.startswith("[CONST]"):
                        conststr = currcomponentkey.replace("[CONST]", "")
                        currtransformation += conststr
                    elif output_aligned_keys:
                        for okey in output_aligned_keys:
                            if okey in val:
                                currtransformation += f"{val[okey]}"
                    elif currcomponentkey in val:
                        currtransformation += f"{val[currcomponentkey]}"
                    if currtransformation:
                        transformation += currtransformation
                transformation = transformation.strip()
                transformation = re.sub(r'\s+', ' ', transformation)
                transformations[valid] = {
                    "transformation": transformation,
                    "orig_txt": key,
                    "original": val,
                }

        return transformations
    
    @staticmethod
    def sort_keys_by_appearance_order(key_set, output_str_dict, output_comp_dict):
        if not key_set or len(key_set) <= 1:
            return key_set
            
        key_positions = {}  # key -> list of positions across instances
        instance_orders = []  # list of (instance_id, ordered_keys) tuples
        
        for instance_id, output_str in output_str_dict.items():
            if instance_id not in output_comp_dict:
                continue
                
            components = output_comp_dict[instance_id]
            positions = {}
            
            for key in key_set:
                if key in components:
                    component_str = str(components[key])
                    if component_str.strip():  # Only consider non-empty strings
                        pos = output_str.find(component_str)
                        if pos != -1:  # Found in string
                            positions[key] = pos
                            
                            if key not in key_positions:
                                key_positions[key] = []
                            key_positions[key].append(pos)
            
            if positions:
                sorted_keys = sorted(positions.keys(), key=lambda k: positions[k])
                instance_orders.append((instance_id, sorted_keys))
        
        if not instance_orders:
            return key_set  # No valid instances found
        
        ordering_constraints = {}  # key -> set of keys that should come after
        for key in key_set:
            ordering_constraints[key] = set()
        
        constraint_counts = {}  # (key_a, key_b) -> count of instances where A < B
        
        for instance_id, ordered_keys in instance_orders:
            for i, key_a in enumerate(ordered_keys):
                for j, key_b in enumerate(ordered_keys):
                    if i < j:  # key_a appears before key_b
                        pair = (key_a, key_b)
                        if pair not in constraint_counts:
                            constraint_counts[pair] = 0
                        constraint_counts[pair] += 1
        
        for (key_a, key_b), count in constraint_counts.items():
            reverse_pair = (key_b, key_a)
            reverse_count = constraint_counts.get(reverse_pair, 0)
            
            if count > reverse_count:
                ordering_constraints[key_a].add(key_b)
            elif reverse_count > count:
                ordering_constraints[key_b].add(key_a)
        
        def topological_sort(constraints):
            in_degree = {key: 0 for key in key_set}
            for key in constraints:
                for dependent in constraints[key]:
                    if dependent in in_degree:
                        in_degree[dependent] += 1
            
            queue = [key for key in key_set if in_degree[key] == 0]
            result = []
            
            while queue:
                queue.sort()
                current = queue.pop(0)
                result.append(current)
                
                for dependent in constraints.get(current, set()):
                    if dependent in in_degree:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            queue.append(dependent)
            
            remaining = [key for key in key_set if key not in result]
            remaining.sort()  # Maintain deterministic order
            result.extend(remaining)
            
            return result
        
        sorted_keys = topological_sort(ordering_constraints)
        
        return sorted_keys