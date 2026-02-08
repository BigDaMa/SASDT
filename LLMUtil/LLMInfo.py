from pathlib import Path

class LLMInfo:
    _project_root = Path(__file__).parent.parent
    api_key = str(_project_root / 'openaiapi.txt')
    DEFAULT_MODEL = 'gpt-5.1'
    DEFAULT_TEMPERATURE = 0
    DEFAULT_MAX_TOKENS = 32767
    DEFAULT_SEED = 42
    ENABLE_SEED = True
    DEFAULT_PROMPT = "label+string"

    MSG_SYS_SEMEXT = "The input contains a series of strings split by '[ITEM]' in a single line. Split the strings into the smallest semantic components. The response only contains 'semantic component type': 'corresponding string' pairs, each in a single line with no additional explanation. Use '[ITEM]' in a single line to separate them. If multiple components within the same string have the same semantic component type, append a number with a single preceding space to differentiate them."

    MSG_SYS_FURSP = "A series of initial splits are provided in the format 'semantic component type': 'corresponding string', with '[ITEM]' in a single line to as item delimiter. Split the given semantic types {sem_types} into smallest semantic components. The new type names do not need to contain the original type name. If the given type cannot be further split, return that exact string. The output should be exactly in the format 'semantic component type': 'corresponding string', each in a single line with no additional token. Use '[ITEM]' in a single line to separate the further splits from multiple input items. If multiple components within the same string have the same semantic component type, append a number with a single preceding space to differentiate them."

    MSG_SYS_FINALSPLIT = "The input contains a series of strings split by '[ITEM]' in a single line. Split the strings into the smallest semantic components. At least the following semantic types and their corresponding strings must be identified for each string: {sem_types}. The response only contains 'semantic component type': 'corresponding string' pairs, each in a single line with no additional explanation. Use '[ITEM]' in a single line to separate them. If multiple components within the same string have the same semantic component type, append a number with a single preceding space to differentiate them."

    MSG_SYS_SEMEXT_LABEL = "The input contains a series of strings split by '[ITEM]' in a single line. Identify the order of occurrence of the smallest semantic components in each string. The response only contains '[semantic component type]' in the order of occurrence, each in a single line with no additional explanation. Splitting on syllable or character level is prohibited for non-numerical tokens. Punctuation in the input should be ignored. Use '[ITEM]' in a single line to separate the splits from each original string. If multiple components within the same string have the same semantic component type, append a number with a single preceding space to differentiate them."

    MSG_SYS_FURSP_LABEL = "The following semantic types are identified from the given strings in the format '[semantic component type]': {sem_types}, with '[ITEM]' in a single line to as item delimiter. Check if the given semantic types can be further split. The new type names do not need to contain the original type name. If the given type cannot be split to a more detailed semantic type, do not change the name and the order of the original types. Splitting on syllable or character level is prohibited for non-numerical tokens. Punctuation in the input should be ignored. The output should be exactly in the format '[semantic component type]' in the order of occurrence, each in a single line with no additional token. Use '[ITEM]' in a single line to separate the further splits from each original string. If multiple components within the same string have the same semantic component type, append a number with a single preceding space to differentiate them."

    MSG_SYS_FINALSPLIT_LABEL = "The input contains a series of strings split by '[ITEM]' in a single line. Identify the order of occurrence of at least the following semantic components in each string: {sem_types}. Splitting on syllable or character level is prohibited for non-numerical tokens. The response only contains '[semantic component type]' in the order of occurrence, each in a single line with no additional explanation. Punctuation in the input should be ignored. Use '[ITEM]' in a single line to separate the split result from each original string. If multiple components within the same string have the same semantic component type, append a number with a single preceding space to differentiate them."

    MSG_SYS_SEMEXT_WITHTYPE = "The input contains a series of strings split by '[ITEM]' in a single line. Split the strings into the smallest semantic components. The response only contains 'semantic component type': 'corresponding string' pairs, each in a single line with no additional explanation. At least the following semantic types and their corresponding strings must be identified for each string: {sem_types}. The corresponding strings for the semantic component types may not overlap. If one given semantic type does not exist, skip it. Use '[ITEM]' in a single line to separate the splits from multiple input items. If multiple components within the same string have the same semantic component type, append a number with a single preceding space to differentiate them."

    MSG_SYS_FURSP = "A series of initial splits are provided in the format 'semantic component type': 'corresponding string', with '[ITEM]' in a single line to as item delimiter. Split the given semantic types {sem_types} into smallest semantic components. The new type names do not need to contain the original type name. If the given type cannot be further split, return that exact string. Splitting on syllable or character level is prohibited for non-numerical tokens. The output should be exactly in the format 'semantic component type': 'corresponding string', each in a single line with no additional token. Use '[ITEM]' in a single line to separate the further splits from multiple input items. If multiple components within the same string have the same semantic component type, append a number with a single preceding space to differentiate them."

    MSG_SYS_FINALSPLIT = "The input contains a series of strings split by '[ITEM]' in a single line. Split the strings into the smallest semantic components. At least the following semantic types and their corresponding strings must be identified for each string: {sem_types}. The response only contains 'semantic component type': 'corresponding string' pairs, each in a single line with no additional explanation. Use '[ITEM]' in a single line to separate them. If multiple components within the same string have the same semantic component type, append a number with a single preceding space to differentiate them."

    @classmethod
    def set_api_key(cls, key_file: str = str(Path(__file__).parent.parent / 'openaiapi.txt')):
        cls.api_key = key_file

    @classmethod
    def set_default_model(cls, model: str = 'gpt-5.1'):
        cls.DEFAULT_MODEL = model

    @classmethod
    def set_default_temperature(cls, temperature: float = 0.05):
        cls.DEFAULT_TEMPERATURE = temperature

    @classmethod
    def set_default_max_tokens(cls, max_tokens: int = 32767):
        cls.DEFAULT_MAX_TOKENS = max_tokens

    @classmethod
    def set_default_seed(cls, seed: int = 42):
        cls.DEFAULT_SEED = seed

    @classmethod
    def set_default_prompt(cls, prompt: str = "label+string"):
        cls.DEFAULT_PROMPT = prompt