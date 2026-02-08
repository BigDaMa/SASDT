import re
from typing import Dict, List

from DataAccess.dataaccess import DataAccess


class RParser:

    def __init__(self):
        pass

    @staticmethod
    def parse_semantic_components(components: str) -> List[Dict[str, str]]:
        result = []
        for line in components.split('\n'):
            if line.startswith('{'):
                line = line[1:]
            if line.endswith('}'):
                line = line[:-1]
            pairs = line.split(',')
            component_dict = {}
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    component_dict[key.strip().strip('"')] = value.strip().strip('"')
            if component_dict:
                result.append(component_dict)
        return result
    
    @staticmethod
    def parse_component_str(component_str: List[str], escapespecchar: bool = True) -> Dict[str, Dict[str, str]]:
        components = dict()
        for i, splitstr in enumerate(component_str):
            currsplitstr = splitstr.strip()
            for line in currsplitstr.split('\n'):
                if i not in components.keys():
                    components[i] = dict()
                if line.strip():
                    try:
                        key = line.strip(' ').split(':', 1)[0].strip()
                        if escapespecchar:
                            key = re.sub(r'[^a-zA-Z0-9]', ' ', key).lower()
                        key = re.sub(r'\s+', ' ', key).strip()
                        value = line.strip(' ').split(':', 1)[1].strip() if ':' in line else ''
                        if value:
                            if escapespecchar:
                                value = re.sub(r'[^a-zA-Z0-9]', ' ', value).strip()
                            value = re.sub(r'\s+', ' ', value)
                        else:
                            value = ''
                        components[i][key] = value
                    except Exception as e:
                        print(f"Error processing LLM response component {i}: {line.strip()} with error message: {e}\nOriginal LLM response for this instance: {splitstr}")
                        return None
        return components
    
    @staticmethod
    def __process_components__(components: List[Dict[str, str]]) -> Dict[str, str]:
        for key, val in components.items():
            if isinstance(val, list):
                max_keys = 0
                best_val = None
                for v in val:
                    if isinstance(v, dict) and len(v) > max_keys:
                        max_keys = len(v)
                        best_val = v
                components[key] = best_val if best_val else {}
        return components
    
    @staticmethod
    def parse_sem_types(sem_types: List[str]) -> str:
        if len(sem_types) == 1:
            return str(sem_types[0])
        elif len(sem_types) == 2:
            return f"{sem_types[0]} and {sem_types[1]}"
        else:
            return f"{', '.join(sem_types[:-1])}, and {sem_types[-1]}"
        
    @staticmethod
    def key_normalize(k: str) -> str:
        k_normalize = re.sub(r'[^a-zA-Z0-9]', ' ', k).lower().strip()
        pat_num_end = r'^(?!\d+$).*?(\d+)$'
        match = re.search(pat_num_end, k_normalize)
        if match:
            matchstr = match.group(1)
        else:
            matchstr = None
        if matchstr and matchstr.isdigit():
            k_normalize = k_normalize.removesuffix(matchstr)
        k_normalize = re.sub(r'[(\s]s[)\s]$', '', k_normalize).strip()
        k_normalize = re.sub(r'\s+', ' ', k_normalize)
        k_normalize = k_normalize.strip()
        return k_normalize
    
    @staticmethod
    def key_normalize_with_underscore(k: str) -> str:
        processed_k = re.sub(r'\d+$', '', k)
        processed_k = processed_k.lower().strip()
        processed_k = processed_k.rstrip('_')
        return processed_k
    
    @staticmethod
    def remove_string_wrapper(resstr: str, origstr: str) -> str:
        if not ((resstr.startswith('"') and resstr.endswith('"')) or 
                (resstr.startswith("'") and resstr.endswith("'"))):
            return resstr
        
        if resstr in origstr:
            return resstr
        
        if (resstr.startswith('"') and resstr.endswith('"')) or (resstr.startswith("'") and resstr.endswith("'")):
            unwrapped = resstr[1:-1]
            if unwrapped in origstr:
                return unwrapped
        
        return resstr
    
    @staticmethod
    def get_types_from_split(response: str) -> List[str]:
        final_type_set = list()
        splitres = re.split(r'\[ITEM\]', response, flags=re.IGNORECASE)
        for splitstr in splitres:
            for line in splitstr.split('\n'):
                if line.strip():
                    # If the line contains a colon, it is a semantic type
                    if ':' in line:
                        key = line.split(':', 1)[0].strip()
                        if key not in final_type_set:
                            final_type_set.append(key)
        return final_type_set
    
    @staticmethod
    def require_further_split(input_components: Dict[str, Dict[str, str]], output_components: Dict[str, Dict[str, str]]) -> bool:
        num_require_further_split = 0
        outbreak = False
        for idx in input_components.keys():
            curr_input_components = input_components[idx]
            curr_output_components = output_components[idx]
            for okey, oval in curr_output_components.items():
                if okey in curr_input_components and oval != curr_input_components[okey]:
                    num_require_further_split += 1
                    break
                if okey not in curr_input_components:
                    if okey and okey.strip()[-1].isdigit():
                        okey_normalize = RParser.key_normalize(okey)
                        for ikey in curr_input_components.keys():
                            outbreak = False
                            if ikey and ikey.strip()[-1].isdigit():
                                ikey_normalize = RParser.key_normalize(ikey)
                            else:
                                ikey_normalize = ikey.strip()
                            if okey_normalize == ikey_normalize and oval != curr_input_components[ikey]:
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
                            if okey == ikey_normalize and oval != curr_input_components[ikey]:
                                num_require_further_split += 1
                                outbreak = True
                                break
                        if outbreak:
                            break
        return (num_require_further_split * 1.0) / (DataAccess.NUM_EXAMPLES * 1.0) > 0.5


class RParser_Prompt:

    def __init__(self):
        pass

    @staticmethod
    def parse_semantic_components(components: str) -> List[Dict[str, str]]:
        result = []
        for line in components.split('\n'):
            if line.startswith('{'):
                line = line[1:]
            if line.endswith('}'):
                line = line[:-1]
            pairs = line.split(',')
            component_dict = {}
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    component_dict[key.strip().strip('"')] = value.strip().strip('"')
            if component_dict:
                result.append(component_dict)
        return result
    
    @staticmethod
    def parse_component_str(component_str: List[str], escapespecchar: bool = True) -> Dict[str, List[str]]:
        components = dict()
        for i, splitstr in enumerate(component_str):
            currsplitstr = splitstr.strip()
            for line in currsplitstr.split('\n'):
                if i not in components.keys():
                    components[i] = list()
                if line.strip():
                    try:
                        value = line.strip()
                        if value:
                            if escapespecchar:
                                value = re.sub(r'[^a-zA-Z0-9]', ' ', value).strip()
                            value = re.sub(r'\s+', ' ', value)
                        else:
                            value = ''
                        components[i].append(value)
                    except Exception as e:
                        print(f"Error processing LLM response component {i}: {line.strip()} with error message: {e}\nOriginal LLM response for this instance: {splitstr}")
                        return None
        return components
    
    @staticmethod
    def __process_components__(components: List[Dict[str, str]]) -> Dict[str, str]:
        for key, val in components.items():
            if isinstance(val, list):
                max_keys = 0
                best_val = None
                for v in val:
                    if isinstance(v, dict) and len(v) > max_keys:
                        max_keys = len(v)
                        best_val = v
                components[key] = best_val if best_val else {}
        return components
    
    @staticmethod
    def parse_sem_types(sem_types: List[str]) -> str:
        if len(sem_types) == 1:
            return str(sem_types[0])
        elif len(sem_types) == 2:
            return f"{sem_types[0]} and {sem_types[1]}"
        else:
            return f"{', '.join(sem_types[:-1])}, and {sem_types[-1]}"
        
    @staticmethod
    def key_normalize(k: str) -> str:
        k_normalize = re.sub(r'[^a-zA-Z0-9]', ' ', k).lower().strip()
        pat_num_end = r'^(?!\d+$).*?(\d+)$'
        match = re.search(pat_num_end, k_normalize)
        if match:
            matchstr = match.group(1)
        else:
            matchstr = None
        if matchstr and matchstr.isdigit():
            k_normalize = k_normalize.removesuffix(matchstr)
        k_normalize = re.sub(r'[(\s]s[)\s]$', '', k_normalize).strip()
        k_normalize = re.sub(r'\s+', ' ', k_normalize)
        k_normalize = k_normalize.strip()
        return k_normalize
    
    @staticmethod
    def get_types_from_split(response: str) -> List[str]:
        final_type_set = list()
        splitres = re.split(r'\[ITEM\]', response, flags=re.IGNORECASE)
        for splitstr in splitres:
            for line in splitstr.split('\n'):
                if line.strip():
                    if line.strip() not in final_type_set:
                        final_type_set.append(line.strip())
        return final_type_set
    
    @staticmethod
    def require_further_split(input_components: Dict[str, Dict[str, str]], output_components: Dict[str, Dict[str, str]]) -> bool:
        num_require_further_split = 0
        for idx in input_components.keys():
            curr_input_components = input_components[idx]
            curr_output_components = output_components[idx]
            for okey, oval in curr_output_components.items():
                if okey in curr_input_components and oval != curr_input_components[okey]:
                    num_require_further_split += 1
                    break
                if okey not in curr_input_components:
                    if okey and okey.strip()[-1].isdigit():
                        okey_normalize = RParser.key_normalize(okey)
                        for ikey in curr_input_components.keys():
                            outbreak = False
                            if ikey and ikey.strip()[-1].isdigit():
                                ikey_normalize = RParser.key_normalize(ikey)
                            if okey_normalize == ikey_normalize and oval != curr_input_components[ikey]:
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
                            if okey == ikey_normalize and oval != curr_input_components[ikey]:
                                num_require_further_split += 1
                                outbreak = True
                                break
                        if outbreak:
                            break
        return (num_require_further_split * 1.0) / (DataAccess.NUM_EXAMPLES * 1.0) > 0.5