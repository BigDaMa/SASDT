from abc import ABC, abstractmethod
import random
from typing import Dict, List, Literal, Set, Tuple, Union
import openai
import re
import requests
from tqdm import tqdm
import time

from DataAccess.dataaccess import DataAccess
from LLMUtil.LLMInfo import LLMInfo
from LLMUtil.LLMPrompt import LLMPrompt
from LLMUtil.ResponseParser import RParser, RParser_Prompt
from Utils.BenchmarkCache import BenchmarkCache
from Utils.CacheUtil import cache_result, perfect_split_override, perfect_split_override_prompt
from Utils.config import Config

class LLMUtil:
    def __init__(self):
        pass

class OpenAIUtil(LLMUtil):
    def __init__(self):
        super().__init__()
        # Initialize OpenAI specific configurations here

    @staticmethod
    def get_client(key_file: str = LLMInfo.api_key) -> openai.OpenAI:
        """
        Get OpenAI client using the API key from a file.
        
        :param key_file: Path to the file containing the OpenAI API key.
        :return: OpenAI client instance.
        """
        with open(key_file, 'r') as f:
            api_key = f.read().strip()
        return openai.OpenAI(api_key=api_key)
    
    @staticmethod
    def get_semantic_components(query: str, file: str = None, model: str = None, temperature: float = None, max_tokens: int = None, seed: int = None, use_type: bool = False, type_list: List[str] = None) -> str:
        client = OpenAIUtil.get_client()
        if use_type and type_list:
            sym_types_str = RParser.parse_sem_types(type_list)
            instruction = LLMInfo.MSG_SYS_SEMEXT_WITHTYPE.format(sem_types = sym_types_str)
        else:
            instruction = LLMInfo.MSG_SYS_SEMEXT
        message = query
        request_kwargs = {
            "model": model or LLMInfo.DEFAULT_MODEL,
            "instructions": instruction,
            "input": message,
        }
        if request_kwargs.get("model") == "gpt-4.1":
            request_kwargs["temperature"] = temperature or LLMInfo.DEFAULT_TEMPERATURE
        response = client.responses.create(**request_kwargs)
        return response.output_text.strip()
    
    @staticmethod
    def get_semantic_components_wrapper(queries: List[str], file: str, model: str = None, temperature: float = None, max_tokens: int = None, seed = None) -> str:
        responses = [OpenAIUtil.get_semantic_components(query.replace("'", "").replace('"', ""), file, model, temperature, max_tokens, seed) for query in queries]
        return responses
    
    @staticmethod
    def get_further_split_query_from_components(query_components: Dict[str, Dict[str, str]]) -> str:
        query_list = []
        for key, val in query_components.items():
            single_query = list()
            if isinstance(val, dict):
                for skey, svalue in val.items():
                    if svalue:
                        single_query.append(f"{skey}: {svalue}")
            elif isinstance(val, list):
                for item in val:
                    if item:
                        single_query.append(f"{item}")
            if single_query:
                query_list.append("\n".join(single_query))
        return "\n[item]\n".join(query_list)
    
    @staticmethod
    def get_further_split(query_components: str, sem_types_str: str, model: str = None, temperature: float = None, max_tokens: int = None, seed: int = None) -> str:
        client = OpenAIUtil.get_client()
        instruction = LLMInfo.MSG_SYS_FURSP.format(sem_types = sem_types_str)
        message = query_components
        request_kwargs = {
            "model": model or LLMInfo.DEFAULT_MODEL,
            "instructions": instruction,
            "input": message,
        }
        if request_kwargs.get("model") == "gpt-4.1":
            request_kwargs["temperature"] = temperature or LLMInfo.DEFAULT_TEMPERATURE
        response = client.responses.create(**request_kwargs)
        return response.output_text.strip()
    
    @staticmethod
    def get_final_split(query: str, sem_types = List[str], model: str = None, temperature: float = None, max_tokens: int = None, seed: int = None) -> str:
        client = OpenAIUtil.get_client()
        sem_types_str = ', '.join(sem_types)
        instruction = LLMInfo.MSG_SYS_FINALSPLIT.format(sem_types = sem_types_str)
        # print(f"Instruction for final split: {instruction}")
        message = query
        request_kwargs = {
            "model": model or LLMInfo.DEFAULT_MODEL,
            "instructions": instruction,
            "input": message,
        }
        if request_kwargs.get("model") == "gpt-4.1":
            request_kwargs["temperature"] = temperature or LLMInfo.DEFAULT_TEMPERATURE
        response = client.responses.create(**request_kwargs)
        return response.output_text.strip()
    
    @staticmethod
    @perfect_split_override()
    @cache_result()
    def get_split(query_dict: Dict[int, str], file: str, input_idx: List[int], output_idx: List[int], prompt: str = "label+string", always_split_further: bool = False, use_type: bool = False, type_list: List[str] = None) -> Dict[int, Dict[str, str]]:
        if not type_list:
            use_type = False
        LLMInfo.set_default_prompt(prompt)
        query_idx = list(query_dict.keys())
        query_strs = list(query_dict.values())
        LLMQuery = PromptUtil.build_llm_query(query_strs)
        sem_components = OpenAIUtil.get_semantic_components(LLMQuery, file, use_type = use_type, type_list = type_list)
        init_sem_components_str = re.split(r'\[ITEM\]', sem_components, flags=re.IGNORECASE)
        init_output_components_str = [sem_components[i] for i in query_idx if i in output_idx]
        init_input_components_str = [sem_components[i] for i in query_idx if i in input_idx]
        init_output_components = RParser.parse_component_str(init_output_components_str)
        init_input_components = RParser.parse_component_str(init_input_components_str)
        initial_sem_types = RParser.get_types_from_split(sem_components)
        if always_split_further or RParser.require_further_split(init_input_components, init_output_components):
            initial_sem_types_str = RParser.parse_sem_types(list(initial_sem_types))
            print(f"Initial Semantic Types String: {initial_sem_types_str}")
            semantic_components_split = re.split(r'\[ITEM\]', sem_components, flags=re.IGNORECASE)
            query_further_split = OpenAIUtil.get_further_split_query_from_components(RParser.parse_component_str(semantic_components_split, escapespecchar=False))
            # print(f"Query Further Split:\n{query_further_split}")
            semantic_components_further_split = RParser.get_further_split(query_further_split, initial_sem_types_str)
            # print(f"Semantic Components Further Split:\n{semantic_components_further_split}")
            final_type_set = RParser.get_types_from_split(semantic_components_further_split)
        else:
            final_type_set = initial_sem_types
            semantic_components_further_split = sem_components
        
        llm_query = PromptUtil.build_llm_query(query_strs)
        semantic_components = OpenAIUtil.get_final_split(llm_query, list(final_type_set))
        semantic_components_split = re.split(r'\[ITEM\]', semantic_components, flags=re.IGNORECASE)
        semantic_components_split_dict = dict()
        for i in query_idx:
            curr_sem_str = semantic_components_split[query_idx.index(i)]
            for item in curr_sem_str.split('\n'):
                if item.strip():
                    try:
                        key = item.strip(' ').split(':', 1)[0].strip()
                        key = re.sub(r'[^a-zA-Z0-9]', ' ', key).lower()
                        key = re.sub(r'\s+', ' ', key).strip()
                        value = item.strip(' ').split(':', 1)[1].strip() if ':' in item else ''
                        if not value:
                            value = ''
                        if i not in semantic_components_split_dict.keys():
                            semantic_components_split_dict[i] = dict()
                        value = RParser.remove_string_wrapper(value, query_dict[i])
                        semantic_components_split_dict[i][key] = value
                    except Exception as e:
                        print(f"Error processing final LLM response component {i}: {item.strip()} with error message: {e}\nOriginal LLM response for this instance: {curr_sem_str}")
                        return None

        return semantic_components_split_dict
    

class NovitaUtil(LLMUtil):
    def __init__(self):
        super().__init__()
        # Initialize OpenAI specific configurations here

    @staticmethod
    def get_client(key_file: str = LLMInfo.novita_api_key) -> openai.OpenAI:
        """
        Get OpenAI client using the API key from a file.
        
        :param key_file: Path to the file containing the OpenAI API key.
        :return: OpenAI client instance.
        """
        with open(key_file, 'r') as f:
            api_key = f.read().strip()
        return openai.OpenAI(api_key=api_key,
                             base_url="https://api.novita.ai/openai"
                            )
    
    @staticmethod
    def get_novita_model(modelname: str) -> str:
        model_map = {
            "deepseek": "deepseek/deepseek-v3.2",
            "glm": "zai-org/glm-4.7",
        }
        return model_map.get(modelname, modelname)
    
    @staticmethod
    def get_semantic_components(query: str, file: str = None, model: str = None, temperature: float = None, max_tokens: int = None, seed: int = None, use_type: bool = False, type_list: List[str] = None) -> str:
        client = NovitaUtil.get_client()
        if use_type and type_list:
            sym_types_str = RParser.parse_sem_types(type_list)
            instruction = LLMInfo.MSG_SYS_SEMEXT_WITHTYPE.format(sem_types = sym_types_str)
        else:
            instruction = LLMInfo.MSG_SYS_SEMEXT
        message = query
        request_kwargs = {
            "model": NovitaUtil.get_novita_model(model or LLMInfo.DEFAULT_MODEL),
            "instructions": instruction,
            "input": message,
        }
        if request_kwargs.get("model") == "gpt-4.1":
            request_kwargs["temperature"] = temperature or LLMInfo.DEFAULT_TEMPERATURE
        response = client.completions.create(**request_kwargs)
        return response.output_text.strip()
    
    @staticmethod
    def get_semantic_components_wrapper(queries: List[str], file: str, model: str = None, temperature: float = None, max_tokens: int = None, seed = None) -> str:
        responses = [NovitaUtil.get_semantic_components(query.replace("'", "").replace('"', ""), file, model, temperature, max_tokens, seed) for query in queries]
        return responses
    
    @staticmethod
    def get_further_split_query_from_components(query_components: Dict[str, Dict[str, str]]) -> str:
        query_list = []
        for key, val in query_components.items():
            single_query = list()
            if isinstance(val, dict):
                for skey, svalue in val.items():
                    if svalue:
                        single_query.append(f"{skey}: {svalue}")
            elif isinstance(val, list):
                for item in val:
                    if item:
                        single_query.append(f"{item}")
            if single_query:
                query_list.append("\n".join(single_query))
        return "\n[item]\n".join(query_list)
    
    @staticmethod
    def get_further_split(query_components: str, sem_types_str: str, model: str = None, temperature: float = None, max_tokens: int = None, seed: int = None) -> str:
        client = NovitaUtil.get_client()
        instruction = LLMInfo.MSG_SYS_FURSP.format(sem_types = sem_types_str)
        message = query_components
        request_kwargs = {
            "model": NovitaUtil.get_novita_model(model or LLMInfo.DEFAULT_MODEL),
            "instructions": instruction,
            "input": message,
        }
        if request_kwargs.get("model").startswith("deepseek") or request_kwargs.get("model").startswith("zai"):
            request_kwargs["temperature"] = temperature or LLMInfo.DEFAULT_TEMPERATURE
        response = client.completions.create(**request_kwargs)
        return response.output_text.strip()
    
    @staticmethod
    def get_final_split(query: str, sem_types = List[str], model: str = None, temperature: float = None, max_tokens: int = None, seed: int = None) -> str:
        client = NovitaUtil.get_client()
        sem_types_str = ', '.join(sem_types)
        instruction = LLMInfo.MSG_SYS_FINALSPLIT.format(sem_types = sem_types_str)
        # print(f"Instruction for final split: {instruction}")
        message = query
        request_kwargs = {
            "model": NovitaUtil.get_novita_model(model or LLMInfo.DEFAULT_MODEL),
            "instructions": instruction,
            "input": message,
        }
        if request_kwargs.get("model").startswith("deepseek") or request_kwargs.get("model").startswith("zai"):
            request_kwargs["temperature"] = temperature or LLMInfo.DEFAULT_TEMPERATURE
        response = client.completions.create(**request_kwargs)
        return response.output_text.strip()
    
    @staticmethod
    @perfect_split_override()
    @cache_result()
    def get_split(query_dict: Dict[int, str], file: str, input_idx: List[int], output_idx: List[int], prompt: str = "label+string", always_split_further: bool = False, use_type: bool = False, type_list: List[str] = None) -> Dict[int, Dict[str, str]]:
        if not type_list:
            use_type = False
        LLMInfo.set_default_prompt(prompt)
        query_idx = list(query_dict.keys())
        query_strs = list(query_dict.values())
        LLMQuery = PromptUtil.build_llm_query(query_strs)
        sem_components = NovitaUtil.get_semantic_components(LLMQuery, file, use_type = use_type, type_list = type_list)
        init_sem_components_str = re.split(r'\[ITEM\]', sem_components, flags=re.IGNORECASE)
        init_output_components_str = [sem_components[i] for i in query_idx if i in output_idx]
        init_input_components_str = [sem_components[i] for i in query_idx if i in input_idx]
        init_output_components = RParser.parse_component_str(init_output_components_str)
        init_input_components = RParser.parse_component_str(init_input_components_str)
        initial_sem_types = RParser.get_types_from_split(sem_components)
        if always_split_further or RParser.require_further_split(init_input_components, init_output_components):
            initial_sem_types_str = RParser.parse_sem_types(list(initial_sem_types))
            print(f"Initial Semantic Types String: {initial_sem_types_str}")
            semantic_components_split = re.split(r'\[ITEM\]', sem_components, flags=re.IGNORECASE)
            query_further_split = NovitaUtil.get_further_split_query_from_components(RParser.parse_component_str(semantic_components_split, escapespecchar=False))
            # print(f"Query Further Split:\n{query_further_split}")
            semantic_components_further_split = RParser.get_further_split(query_further_split, initial_sem_types_str)
            # print(f"Semantic Components Further Split:\n{semantic_components_further_split}")
            final_type_set = RParser.get_types_from_split(semantic_components_further_split)
        else:
            final_type_set = initial_sem_types
            semantic_components_further_split = sem_components
        
        llm_query = PromptUtil.build_llm_query(query_strs)
        semantic_components = NovitaUtil.get_final_split(llm_query, list(final_type_set))
        semantic_components_split = re.split(r'\[ITEM\]', semantic_components, flags=re.IGNORECASE)
        semantic_components_split_dict = dict()
        for i in query_idx:
            curr_sem_str = semantic_components_split[query_idx.index(i)]
            for item in curr_sem_str.split('\n'):
                if item.strip():
                    try:
                        key = item.strip(' ').split(':', 1)[0].strip()
                        key = re.sub(r'[^a-zA-Z0-9]', ' ', key).lower()
                        key = re.sub(r'\s+', ' ', key).strip()
                        value = item.strip(' ').split(':', 1)[1].strip() if ':' in item else ''
                        if not value:
                            value = ''
                        if i not in semantic_components_split_dict.keys():
                            semantic_components_split_dict[i] = dict()
                        value = RParser.remove_string_wrapper(value, query_dict[i])
                        semantic_components_split_dict[i][key] = value
                    except Exception as e:
                        print(f"Error processing final LLM response component {i}: {item.strip()} with error message: {e}\nOriginal LLM response for this instance: {curr_sem_str}")
                        return None

        return semantic_components_split_dict
    
class OpenAIUtil_Prompt(LLMUtil):
    def __init__(self):
        super().__init__()
        # Initialize OpenAI specific configurations here

    @staticmethod
    def get_client(key_file: str = LLMInfo.api_key) -> openai.OpenAI:
        """
        Get OpenAI client using the API key from a file.
        
        :param key_file: Path to the file containing the OpenAI API key.
        :return: OpenAI client instance.
        """
        with open(key_file, 'r') as f:
            api_key = f.read().strip()
        return openai.OpenAI(api_key=api_key)
    
    @staticmethod
    def get_semantic_components(query: str, file: str = None, model: str = None, temperature: float = None, max_tokens: int = None, seed: int = None, num_retry = 3) -> str:
        if Config.money_saver and "OpenAIUtil_Prompt.get_semantic_components" in BenchmarkCache.OpenAICache and query in BenchmarkCache.OpenAICache["OpenAIUtil_Prompt.get_semantic_components"]:
            return BenchmarkCache.OpenAICache["OpenAIUtil_Prompt.get_semantic_components"][query]
        client = OpenAIUtil_Prompt.get_client()
        # instruction = LLMInfo.MSG_SYS_SEMEXT_LABEL
        instruction = LLMPrompt.BASE_SYS_SEMEXT
        message = query
        request_kwargs = {
            "model": model or LLMInfo.DEFAULT_MODEL,
            "instructions": instruction,
            "input": message,
        }
        if request_kwargs.get("model") == "gpt-4.1":
            request_kwargs["temperature"] = temperature or LLMInfo.DEFAULT_TEMPERATURE
        for i in range(num_retry):
            response = client.responses.create(**request_kwargs)
            if response.error is None or (response.error and response.error == "null"):
                break
            else:
                print(f"Retry {i+1}/{num_retry} for get_semantic_components due to error: {response}")
                time.sleep(5)  # wait for 5 seconds before retrying
        if response.error is not None or (response.error and response.error != "null"):
            raise RuntimeError(f"Failed to get semantic components after {num_retry} retries. Last response: {response}")
        outputtext = response.output_text.strip()
        if Config.money_saver:
            if "OpenAIUtil_Prompt.get_semantic_components" not in BenchmarkCache.OpenAICache:
                BenchmarkCache.OpenAICache["OpenAIUtil_Prompt.get_semantic_components"] = dict()
            BenchmarkCache.OpenAICache["OpenAIUtil_Prompt.get_semantic_components"][query] = outputtext
        return outputtext
    
    @staticmethod
    def get_semantic_components_wrapper(queries: List[str], file: str, model: str = None, temperature: float = None, max_tokens: int = None, seed = None) -> str:
        responses = [OpenAIUtil_Prompt.get_semantic_components(query.replace("'", "").replace('"', ""), file, model, temperature, max_tokens, seed) for query in queries]
        return responses
    
    @staticmethod
    def get_further_split_query_from_components(query_components: Dict[str, Dict[str, str]]) -> str:
        query_list = []
        for key, val in query_components.items():
            single_query = list()
            if isinstance(val, dict):
                for skey, svalue in val.items():
                    if svalue:
                        single_query.append(f"{skey}: {svalue}")
            elif isinstance(val, list):
                for item in val:
                    if item:
                        single_query.append(f"{item}")
            if single_query:
                query_list.append("\n".join(single_query))
        return "\n[ITEM]\n".join(query_list)
    
    @staticmethod
    def get_further_split(query_components: str, sem_types_str: str, model: str = None, temperature: float = None, max_tokens: int = None, seed: int = None, num_retry = 3) -> str:
        if Config.money_saver and "OpenAIUtil_Prompt.get_further_split" in BenchmarkCache.OpenAICache and query_components in BenchmarkCache.OpenAICache["OpenAIUtil_Prompt.get_further_split"]:
            return BenchmarkCache.OpenAICache["OpenAIUtil_Prompt.get_further_split"][query_components]
        client = OpenAIUtil_Prompt.get_client()
        # instruction = LLMInfo.MSG_SYS_FURSP_LABEL.format(sem_types = sem_types_str)
        instruction = LLMPrompt.BASE_SYS_FURSP.format(sem_types = sem_types_str)
        message = query_components
        request_kwargs = {
            "model": model or LLMInfo.DEFAULT_MODEL,
            "instructions": instruction,
            "input": message,
        }
        if request_kwargs.get("model") == "gpt-4.1":
            request_kwargs["temperature"] = temperature or LLMInfo.DEFAULT_TEMPERATURE
        for i in range(num_retry):
            response = client.responses.create(**request_kwargs)
            if response.error is None or (response.error and response.error == "null"):
                break
            else:
                print(f"Retry {i+1}/{num_retry} for get_further_split due to error: {response}")
                time.sleep(5)  # wait for 5 seconds before retrying
        if response.error is not None or (response.error and response.error != "null"):
            raise RuntimeError(f"Failed to get further split after {num_retry} retries. Last response: {response}")
        outputtext = response.output_text.strip()
        if Config.money_saver:
            if "OpenAIUtil_Prompt.get_further_split" not in BenchmarkCache.OpenAICache:
                BenchmarkCache.OpenAICache["OpenAIUtil_Prompt.get_further_split"] = dict()
            BenchmarkCache.OpenAICache["OpenAIUtil_Prompt.get_further_split"][query_components] = outputtext
        return outputtext
    
    @staticmethod
    def get_final_split(query: str, sem_types = List[str], model: str = None, temperature: float = None, max_tokens: int = None, seed: int = None, num_retry = 3) -> str:
        if Config.money_saver and "OpenAIUtil_Prompt.get_final_split" in BenchmarkCache.OpenAICache and query in BenchmarkCache.OpenAICache["OpenAIUtil_Prompt.get_final_split"]:
            return BenchmarkCache.OpenAICache["OpenAIUtil_Prompt.get_final_split"][query]
        client = OpenAIUtil_Prompt.get_client()
        sem_types_str = ', '.join(sem_types)
        instruction = LLMPrompt.BASE_SYS_FINALSP.format(sem_types = sem_types_str)
        # print(f"Instruction for final split: {instruction}")
        message = query
        request_kwargs = {
            "model": model or LLMInfo.DEFAULT_MODEL,
            "instructions": instruction,
            "input": message,
        }
        if request_kwargs.get("model") == "gpt-4.1" or request_kwargs.get("model") == "gpt-5.1":
            request_kwargs["temperature"] = temperature or LLMInfo.DEFAULT_TEMPERATURE
        for i in range(num_retry):
            response = client.responses.create(**request_kwargs)
            if response.error is None or (response.error and response.error == "null"):
                break
            else:
                print(f"Retry {i+1}/{num_retry} for get_final_split due to error: {response}")
                time.sleep(5)  # wait for 5 seconds before retrying
        if response.error is not None or (response.error and response.error != "null"):
            raise RuntimeError(f"Failed to get final split after {num_retry} retries. Last response: {response}")
        outputtext = response.output_text.strip()
        if Config.money_saver:
            if "OpenAIUtil_Prompt.get_final_split" not in BenchmarkCache.OpenAICache:
                BenchmarkCache.OpenAICache["OpenAIUtil_Prompt.get_final_split"] = dict()
            BenchmarkCache.OpenAICache["OpenAIUtil_Prompt.get_final_split"][query] = outputtext
        return outputtext
    
    @staticmethod
    @perfect_split_override_prompt()
    @cache_result()
    def get_split_prompt(query_dict: Dict[int, str], file: str, input_idx: List[int], output_idx: List[int], prompt: str = "label", always_split_further: bool = False) -> Dict[int, List[str]]:
        LLMInfo.set_default_prompt(prompt)
        query_idx = list(query_dict.keys())
        query_strs = list(query_dict.values())
        LLMQuery = PromptUtil.build_llm_query(query_strs)
        print(f"LLM Query for initial split:\n{LLMQuery}")
        init_sem_components_str = None
        retry_count = 0
        sem_components = ""
        while not init_sem_components_str or len(init_sem_components_str) < len(query_strs):
            if retry_count > Config.num_retry:
                if init_sem_components_str and len(init_sem_components_str) >= len(query_strs) // 2:
                    last_sem_comp = init_sem_components_str[-1]
                    for i in range(len(query_strs) - len(init_sem_components_str)):
                        sem_components += f"\n[ITEM]\n{last_sem_comp}"
                        init_sem_components_str = re.split(r'\[ITEM\]', sem_components, flags=re.IGNORECASE)
                    break
                else:
                    print(f"Exceeded maximum retry attempts ({Config.num_retry}) for getting initial semantic components. Semantic components for output is uncertain. Aborting.")
                    print(f"Last LLM Response String:\n{sem_components}")
                    print(f"Last LLM Response:\n{init_sem_components_str}")
                    raise RuntimeError("Failed to get initial semantic components from LLM.")
            sem_components = OpenAIUtil_Prompt.get_semantic_components(LLMQuery, file)
            init_sem_components_str = re.split(r'\[ITEM\]', sem_components, flags=re.IGNORECASE)
            retry_count += 1
        print(f"Semantic Components from LLM:\n{sem_components}")
        init_output_components_str = [init_sem_components_str[DataAccess.NUM_EXAMPLES + i] for i in range(DataAccess.NUM_EXAMPLES)]
        init_input_components_str = [init_sem_components_str[i] for i in range(DataAccess.NUM_EXAMPLES)]
        init_output_components = RParser_Prompt.parse_component_str(init_output_components_str)
        init_input_components = RParser_Prompt.parse_component_str(init_input_components_str)
        initial_sem_types = RParser_Prompt.get_types_from_split(sem_components)
        if always_split_further:
            initial_sem_types_str = RParser_Prompt.parse_sem_types(list(initial_sem_types))
            print(f"Initial Semantic Types String: {initial_sem_types_str}")
            query_further_split = "\n[ITEM]\n".join(init_sem_components_str)
            # print(f"Query Further Split:\n{query_further_split}")
            semantic_components_further_split = OpenAIUtil_Prompt.get_further_split(query_further_split, initial_sem_types_str)
            # print(f"Semantic Components Further Split:\n{semantic_components_further_split}")
            # final_type_set = RParser.get_types_from_split(semantic_components_further_split)
            final_type_set = semantic_components_further_split
        else:
            final_type_set = initial_sem_types
            semantic_components_further_split = sem_components
        
        llm_query = PromptUtil.build_llm_query(query_strs)
        semantic_components_split = None
        retry_count = 0
        while not semantic_components_split or len(semantic_components_split) < len(query_strs):
            if retry_count > Config.num_retry:
                if semantic_components_split and len(semantic_components_split) >= len(query_strs) // 2:
                    last_sem_comp = semantic_components_split[-1]
                    for i in range(len(query_strs) - len(semantic_components_split)):
                        sem_components += f"\n[ITEM]\n{last_sem_comp}"
                        semantic_components_split = re.split(r'\[ITEM\]', sem_components, flags=re.IGNORECASE)
                    break
                else:
                    print(f"Exceeded maximum retry attempts ({Config.num_retry}) for getting initial semantic components. Semantic components for output is uncertain. Aborting.")
                    print(f"Last LLM Response String:\n{sem_components}")
                    print(f"Last LLM Response:\n{init_sem_components_str}")
                    raise RuntimeError("Failed to get initial semantic components from LLM.")
            semantic_components = OpenAIUtil_Prompt.get_final_split(llm_query, list(final_type_set))
            semantic_components_split = re.split(r'\[ITEM\]', semantic_components, flags=re.IGNORECASE)
            retry_count += 1
        print("Original Text of Final Semantic Components from LLM:")
        print(semantic_components)
        semantic_components_split_dict = dict()
        print(f"Final Semantic Components from LLM:\n{semantic_components_split}")
        for i in query_idx:
            curr_sem_str = semantic_components_split[query_idx.index(i)]
            if "][" in curr_sem_str:
                curr_sem_str = curr_sem_str.replace("][", "]\n[")
            for item in curr_sem_str.split('\n'):
                if item == "[UNKNOWN]":
                    continue
                if item.strip():
                    try:
                        key = item.strip(' ').split(':', 1)[0].strip()
                        key = re.sub(r'[^a-zA-Z0-9]', ' ', key).lower()
                        key = re.sub(r'\s+', ' ', key).strip()
                        if prompt == "label+string":
                            value = item.strip(' ').split(':', 1)[1].strip() if ':' in item else ''
                        else:
                            value = ""
                        # if value:
                        #     value = re.sub(r'[^a-zA-Z0-9]', ' ', value).strip()
                        #     value = re.sub(r'\s+', ' ', value)
                        # else:
                        #     value = ''
                        if not value:
                            value = ''
                        if i not in semantic_components_split_dict.keys():
                            if prompt == "label+string":
                                semantic_components_split_dict[i] = dict()
                            elif prompt == "label":
                                semantic_components_split_dict[i] = list()
                        if prompt == "label+string":
                            semantic_components_split_dict[i][f"[{key}]"] = value
                        elif prompt == "label":
                            semantic_components_split_dict[i].append(f"[{key}]")
                    except Exception as e:
                        print(f"Error processing final LLM response component {i}: {item.strip()} with error message: {e}\nOriginal LLM response for this instance: {curr_sem_str}")
                        return None
            # if prompt == "label":
            #     semantic_components_split_dict[i] = "".join(semantic_components_split_dict[i]) \
            #     if type(semantic_components_split_dict[i]) == list \
            #         and all(e.startswith("[") and e.endswith("]") for e in semantic_components_split_dict[i]) \
            #     else semantic_components_split_dict[i]

        return semantic_components_split_dict
    
    @staticmethod
    def get_sem_component_labels(prompt: List[str], model: str = None, temperature: float = None, num_retry = 3) -> List[str]:
        label_responses = []
        model = model or LLMInfo.DEFAULT_MODEL
        client = OpenAIUtil_Prompt.get_client()
        for p in prompt: 
            message = p
            request_kwargs = {
                "model": model or LLMInfo.DEFAULT_MODEL,
                "input": message,
            }
            if request_kwargs.get("model") == "gpt-4.1":
                request_kwargs["temperature"] = temperature or LLMInfo.DEFAULT_TEMPERATURE
            for i in range(num_retry):
                response = client.responses.create(**request_kwargs)
                if response.error is None or (response.error and response.error == "null"):
                    break
                else:
                    print(f"Retry {i+1}/{num_retry} for get_sem_component_labels due to error: {response}")
                    time.sleep(5)  # wait for 5 seconds before retrying
            if response.error is not None or (response.error and response.error != "null"):
                raise RuntimeError(f"Failed to get semantic component labels after {num_retry} retries. Last response: {response}")
            outputtext = response.output_text.strip()
            label_responses.append(outputtext)
        return label_responses
    
class OLLAMAUtil(LLMUtil):

    __BASE_URL__ = "http://localhost:11434"
    __MODE__ = "gen"
    __MODEL__ = "llama3.2-3b"

    STOP_WORDS = {
        "llama": ["<|end_of_text|>", "<|eot_id|>", "\n"],
        "qwen": ["<|im_end|>"],
        "granite": ["<|end_of_text|>"]
    }

    def __init__(self):
        super().__init__()
        # Initialize Ollama specific configurations here
    
    @staticmethod
    def set_model(model: str):
        OLLAMAUtil.__MODEL__ = model

    @staticmethod
    def set_mode(mode: str):
        OLLAMAUtil.__MODE__ = mode

    @staticmethod
    def make_url() -> str:
        endpoint = "generate" if OLLAMAUtil.__MODE__ == "gen" or OLLAMAUtil.__MODE__ == "generate" else "chat"
        return f"{OLLAMAUtil.__BASE_URL__}/api/{endpoint}"

    @staticmethod
    def get_stop_words() -> List[str]:
        if OLLAMAUtil.__MODEL__.startswith("qwen"):
            return OLLAMAUtil.STOP_WORDS["qwen"]
        if OLLAMAUtil.__MODEL__.startswith("llama"):
            return OLLAMAUtil.STOP_WORDS["llama"]
        if OLLAMAUtil.__MODEL__.startswith("granite"):
            return OLLAMAUtil.STOP_WORDS["granite"]
        return ["<|end_of_text|>", "\n"]
    
    @staticmethod
    def get_ollama_model_from_str():
        if OLLAMAUtil.__MODEL__.startswith("qwen3-0.6"):
            return "qwen3-0.6b-custom"
        elif OLLAMAUtil.__MODEL__.startswith("qwen3-1.7"):
            return "qwen3-1.7b-custom"
        elif OLLAMAUtil.__MODEL__.startswith("qwen3-4b"):
            return "qwen3-4b"
        elif OLLAMAUtil.__MODEL__.startswith("llama3.2-1b"):
            return "llama3.2-1b-custom"
        elif OLLAMAUtil.__MODEL__.startswith("llama3.2-3b-orig-cpu"):
            return "llama3.2-3b-cpu"
        elif OLLAMAUtil.__MODEL__.startswith("llama3.2-3b-orig"):
            return "llama3.2-3b"
        elif OLLAMAUtil.__MODEL__.startswith("llama3.2-3b"):
            return "llama3.2-3b-custom"
        elif OLLAMAUtil.__MODEL__.startswith("granite-1b"):
            return "granite-1b-custom"
        elif OLLAMAUtil.__MODEL__.startswith("phi4"):
            return "phi4"
        elif OLLAMAUtil.__MODEL__.startswith("granite-4b") or OLLAMAUtil.__MODEL__.startswith("granite-micro"):
            return "granite-4b-custom"
        print("Unrecognized model, defaulting to llama3.2-1b-custom")
        return "llama3.2-1b-custom"
    
    @staticmethod
    def lode_model():
        url = OLLAMAUtil.make_url()
        params = {"model": OLLAMAUtil.get_ollama_model_from_str()}
        r = requests.get(url, params=params)

    @staticmethod
    def unload_model():
        url = OLLAMAUtil.make_url()
        params = {"model": OLLAMAUtil.get_ollama_model_from_str(), "keep_alive": 0}
        r = requests.get(url, params=params)

    @staticmethod
    def get_response_batch(prompts: List[str], max_tokens: int = 512, temperature: float = 0.05) -> List[str]:
        responses = []
        url = OLLAMAUtil.make_url()
        headers = {"Content-Type": "application/json"}

        for p in tqdm(prompts):
            if OLLAMAUtil.__MODE__ == "gen" or OLLAMAUtil.__MODE__ == "generate":
                payload = {
                    "model": OLLAMAUtil.get_ollama_model_from_str(),
                    "prompt": p,
                    "max_tokens": max_tokens,
                    # "keep_alive": 0,
                    "temperature": temperature,
                    "stream": False,
                    "options": {
                        "stop": OLLAMAUtil.get_stop_words(),
                        "min_p": 0.1,
                        "top_k": 1,
                        "top_p": 0.05
                    }
                }
            elif OLLAMAUtil.__MODE__ == "chat":
                payload = {
                    "model": OLLAMAUtil.get_ollama_model_from_str(),
                    "messages": [
                        {"role": "user", "content": p}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    # "keep_alive": 0,
                    "stream": False,
                    "options": {
                        "stop": OLLAMAUtil.get_stop_words(),
                        "min_p": 0.1,
                        "top_k": 1,
                        "top_p": 0.05
                    }
                }
            res = requests.post(url, json=payload, headers=headers)
            if res.status_code == 200:
                res = res.json()
                if OLLAMAUtil.__MODE__ == "gen" or OLLAMAUtil.__MODE__ == "generate":
                    responses.append(res.get("response", "").strip())
                elif OLLAMAUtil.__MODE__ == "chat":
                    responses.append(res.get("message", {}).get("content", "").strip())
            else:
                # print(f"Error: {res.status_code}, {res.get("message", {})}")
                print(f"Error:\n{str(res)}")
                responses.append("")
        return responses
    
    # @staticmethod
    # def post_process(responses: List[str]) -> List[str]:
    #     processed_responses = []
    #     for res in responses:
    #         if type(res) == list and all(e.startswith("[") and e.endswith("]") for e in res):
    #             processed_responses.append(res)
    #         else:
    #             pass



class PromptUtil:
    def __init__(self):
        pass

    @staticmethod
    def build_llm_query(query: List[str]) -> str:
        """
        Build a query for the LLM based on the provided examples and the input query.
        
        :param examples: A dictionary of example outputs. keys: example input, values: example output.
        :param query: The input text to be transformed.
        :return: A formatted string ready for the LLM.
        """
        # finallist = list()
        # if "outputstr" in option:
        #     example_output = list(str(d['output']) for d in examples)
        #     example_output_text = "\n[item]\n".join(example_output)
        #     finallist.append(example_output_text)
        # if "inputstr" in option:    
        #     example_input = list(str(d['input']) for d in examples)
        #     example_input_text = "\n[item]\n".join(example_input)
        #     finallist.append(example_input_text)
        # if "querystr" in option:
        #     query_text = "\n[item]\n".join(query)
        #     finallist.append(query_text)
        return "\n[ITEM]\n".join(query)
    
    # @staticmethod
    # def build_ollama_query(examples: List[Tuple[str, str]], query: str, numexamples: int = 2) -> str:
    #     prompt_base = LLMPrompt.OLLAMA_BASE
    #     replaces = {}
    #     types = []
    #     pattern = r'\[([^\]]*)\]'
    #     for ex in examples:
    #         for match in re.finditer(pattern, ex[1]):
    #             types.append(match.group(1))
    #     types = list(set(types))
    #     if len(types) == 1:
    #         # prompt_base = prompt_base.format(sem_types = f"[{types[0]}]")
    #         replaces['sem_types'] = f"[{types[0]}]"
    #     # elif len(types) == 2:
    #     #     prompt_base = prompt_base.format(f"[{types[0]}] and [{types[1]}]")
    #     else:
    #         # prompt_base = prompt_base.format(", ".join([f"[{t}]" for t in types[:-1]]) + f", and [{types[-1]}]")
    #         # prompt_base = prompt_base.format(sem_types = ", ".join([f"[{t}]" for t in types]))
    #         replaces['sem_types'] = ", ".join([f"[{t}]" for t in types])
    #     # print(f"Prompt Base: {prompt_base}")
    #     prompt = ""
    #     if numexamples > len(examples):
    #         numexamples = len(examples)
    #     for i in range(numexamples):
    #         input_example = examples[i][0]
    #         output_example = examples[i][1]
    #         # prompt_base = prompt_base.format(**{f'input{i+1}': input_example, f'output{i+1}': output_example})
    #         replaces[f'input{i+1}'] = input_example
    #         replaces[f'output{i+1}'] = output_example
    #     replaces['query'] = query
    #     prompt = prompt_base.format(**replaces)
    #     # prompt_base = prompt_base.format(query = query)
    #     # for i in range(numexamples):
    #     #     idx = random.randint(0, len(examples) - 1)
    #     #     prompt += f"Q: {examples[idx][0]} A: {examples[idx][1]}\n"
    #     # prompt += f"Q: {query} A: "
    #     # prompt = prompt_base + prompt
    #     return prompt

    @staticmethod
    def build_ollama_query(examples: List[Tuple[str, str]], query: str, numexamples: int = 2) -> str:
        prompt_base = LLMPrompt.OLLAMA_BASE
        replaces = {}
        types = []
        pattern = r'\[([^\]]*)\]'
        for ex in examples:
            for match in re.finditer(pattern, ex[1]):
                types.append(match.group(1))
        types = list(set(types))
        if len(types) == 1:
            replaces['sem_types'] = f"[{types[0]}]"
        else:
            replaces['sem_types'] = ", ".join([f"[{t}]" for t in types])
        prompt = ""
        if numexamples > len(examples):
            numexamples = len(examples)
        compositions_set = set(ex[1] for ex in examples)
        if len(compositions_set) < numexamples:
            remaining = numexamples - len(compositions_set)
        else:
            remaining = 0
        selected = []
        selected_id = []
        for comp in compositions_set:
            for i, ex in enumerate(examples):
                if ex[1] == comp and ex not in selected:
                    selected.append(ex)
                    selected_id.append(i)
                    break
        if remaining:
            if len(examples) - len(selected_id) <= remaining:
                selected_id += [i for i in range(len(examples)) if i not in selected_id]
            else:
                for i, ex in enumerate(examples):
                    if i not in selected_id:
                        selected.append(ex)
                        selected_id.append(i)
                    if len(selected) >= numexamples:
                        break
        examples_str = ""
        for i in selected_id:
            examples_str += f"\tInput: {examples[i][0]}\n\tExpected output: {examples[i][1]}\n"
        replaces['examples'] = examples_str
        replaces['query'] = query
        prompt = prompt_base.format(**replaces)
        # prompt_base = prompt_base.format(query = query)
        # for i in range(numexamples):
        #     idx = random.randint(0, len(examples) - 1)
        #     prompt += f"Q: {examples[idx][0]} A: {examples[idx][1]}\n"
        # prompt += f"Q: {query} A: "
        # prompt = prompt_base + prompt
        return prompt