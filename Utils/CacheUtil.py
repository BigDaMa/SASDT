import functools
import pickle
import os
import hashlib

from Utils.config import Config
from Propagate.position import AbsPosPosition, RegExPosPosition

RegExEscapeDict = {
    "[ESCAPESPACE]": r"\s",
    "[ESCAPEWORD]": r"\b[A-Za-z]+\b",
    "[ESCAPENUMBER]": r"\d+",
    "[ESCAPECOMMA]": r",",
    "[ESCAPEPERIOD]": r"\.",
    "[ESCAPEAT]": r"@",
    "[ESCAPEDOUBLECOLON]": r":",
    "[ESCAPEBACKSLASH]": r"\\",
    "[ESCAPESLASH]": r"/",
    "[ESCAPEIN]": r"In\s",
    "[ESCAPEDOUBLEQUOTE]": r'\"',
    "[ESCAPELEFTPAREN]": r"\(",
    "[ESCAPERIGHTPAREN]": r"\)",
    "[ESCAPEBY]": r"by\s",
}


def cache_result(cache_dir=Config.cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{hashlib.md5(str(args + tuple(kwargs.items())).encode()).hexdigest()}"
            cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            if Config.use_cache and os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                print(f"Loaded from cache: {cache_path}")
                return result
            
            result = func(*args, **kwargs)
            if Config.save_cache and not Config.perfect_cluster and not Config.perfect_split and not Config.perfect_operator:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
                print(f"Saved to cache: {cache_path}")
            return result
        
        return wrapper
    return decorator

def perfect_cluster_override():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if Config.perfect_cluster and not Config.use_cache:
                if args:
                    file = args[0]
                    filename = file.replace('.csv', '_cluster.txt')
                    perfect_data_path = os.path.join(Config.perfect_directory, filename)
                    
                    try:
                        with open(perfect_data_path, 'r') as f:
                            result = list()
                            for line in f:
                                line_split = set([int(x) for x in line.strip().split(',')])
                                result.append(line_split)
                            print(f"Loaded perfect cluster data from: {perfect_data_path}")
                            return result
                    except FileNotFoundError:
                        print(f"Perfect cluster file not found: {perfect_data_path}")
                        print("Falling back to normal function execution")
                    except Exception as e:
                        print(f"Error reading perfect cluster file: {e}")
                        print("Falling back to normal function execution")
            
            # Fall back to normal function execution
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def perfect_split_override():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if Config.perfect_split and not Config.use_cache:
                if args:
                    file = args[1]
                    filename = file.replace('.csv', '_split.txt')
                    query_dict = args[0]
                    perfect_data_path = os.path.join(Config.perfect_directory, filename)
                    
                    try:
                        with open(perfect_data_path, 'r') as f:
                            result = dict()
                            idx = 0
                            for line in f:
                                if line.strip() and line.strip() == '[ITEM]':
                                    idx += 1
                                    continue
                                if line.strip():
                                    try:
                                        key = line.strip(' ').split(':', 1)[0].strip()
                                        value = line.strip(' ').split(':', 1)[1].strip() if ':' in line else ''
                                        if not value:
                                            value = ''
                                        if idx in query_dict.keys():
                                            if idx not in result.keys():
                                                result[idx] = dict()
                                            result[idx][key] = value
                                    except Exception as e:
                                        print(f"Error processing perfect split component {idx}: {line.strip()} with error message: {e}\nOriginal perfect split response for this instance: {line.strip()}")
                                        return None
                            print(f"Loaded perfect split data from: {perfect_data_path}")
                            return result
                    except FileNotFoundError:
                        print(f"Perfect split file not found: {perfect_data_path}")
                        print("Falling back to normal function execution")
                    except Exception as e:
                        print(f"Error reading perfect split file: {e}")
                        print("Falling back to normal function execution")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def perfect_split_override_prompt():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if Config.perfect_split and not Config.use_cache:
                if args:
                    file = args[1]
                    filename = file.replace('.csv', '_split_prompt.txt')
                    query_dict = args[0]
                    perfect_data_path = os.path.join(Config.perfect_directory, filename)
                    
                    try:
                        with open(perfect_data_path, 'r') as f:
                            result = dict()
                            idx = 0
                            for line in f:
                                if line.strip() and line.strip() == '[ITEM]':
                                    idx += 1
                                    continue
                                if line.strip():
                                    try:
                                        key = line.strip(' ').split(':', 1)[0].strip()
                                        value = line.strip(' ').split(':', 1)[1].strip() if ':' in line else ''
                                        if not value:
                                            value = ''
                                        if idx in query_dict.keys():
                                            if idx not in result.keys():
                                                result[idx] = dict()
                                            result[idx][key] = value
                                    except Exception as e:
                                        print(f"Error processing perfect split component {idx}: {line.strip()} with error message: {e}\nOriginal perfect split response for this instance: {line.strip()}")
                                        return None
                            print(f"Loaded perfect split data from: {perfect_data_path}")
                            return result
                    except FileNotFoundError:
                        print(f"Perfect split file not found: {perfect_data_path}")
                        print("Falling back to normal function execution")
                    except Exception as e:
                        print(f"Error reading perfect split file: {e}")
                        print("Falling back to normal function execution")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def perfect_operator_override():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if Config.perfect_operator and not Config.use_cache:
                if args:
                    file = args[-1]
                    filename = file.replace('.csv', '_op.txt')
                    perfect_data_path = os.path.join(Config.perfect_directory, filename)
                    
                    try:
                        with open(perfect_data_path, 'r') as f:
                            result = dict()
                            for line in f:
                                if line.strip().isnumeric():
                                    idx = int(line.strip())
                                    result[idx] = dict()
                                elif ':' in line:
                                    key = line.strip(' ').split(':', 1)[0].strip()
                                    value = line.strip(' ').split(':', 1)[1].strip() if ':' in line else ''
                                    if not value:
                                        value = ''
                                    pos = value.split('[PARAMSPLIT]', 1)
                                    pos1 = pos[0].strip()
                                    pos2 = pos[1].strip()
                                    if pos1.startswith('AbsPos'):
                                        param = int(pos1[pos1.find('(')+1:pos1.rfind(')')].strip())
                                        pos1 = AbsPosPosition(param)
                                    elif pos1.startswith('RegPos'):
                                        param = str(pos1[pos1.find('(')+1:pos1.rfind(')')].strip())
                                        patternend = len(param)
                                        for _ in range(2):
                                            patternend = param.rfind(',', 0, patternend)
                                        pattern = RegExEscapeDict[param[:patternend].strip()]
                                        occstart = len(param)
                                        # for i in range(2):
                                        occend = param.rfind(',', 0, occstart)
                                        occstart = param.rfind(',', 0, occend)
                                        occ = int(param[occstart+1:occend].strip())
                                        direction = param[occend+1:].strip()
                                        pos1 = RegExPosPosition(pattern, occ, direction)
                                    if pos2.startswith('AbsPos'):
                                        param = int(pos2[pos2.find('(')+1:pos2.rfind(')')].strip())
                                        pos2 = AbsPosPosition(param)
                                    elif pos2.startswith('RegPos'):
                                        param = str(pos2[pos2.find('(')+1:pos2.rfind(')')].strip())
                                        patternend = len(param)
                                        for _ in range(2):
                                            patternend = param.rfind(',', 0, patternend)
                                        pattern = RegExEscapeDict[param[:patternend].strip()]
                                        occstart = len(param)
                                        occend = param.rfind(',', 0, occstart)
                                        occstart = param.rfind(',', 0, occend)
                                        occ = int(param[occstart+1:occend].strip())
                                        direction = param[occend+1:].strip()
                                        pos2 = RegExPosPosition(pattern, occ, direction)
                                    result[idx][key] = [(pos1, pos2)]
                                elif line.strip() and line.strip() == '[ITEM]':
                                    continue
                            print(f"Loaded perfect operator data from: {perfect_data_path}")
                            return result
                    except FileNotFoundError:
                        print(f"Perfect operator file not found: {perfect_data_path}")
                        print("Falling back to normal function execution")
                    except Exception as e:
                        print(f"Error reading perfect operator file: {e}")
                        print("Falling back to normal function execution")
            
            # Fall back to normal function execution
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def clear_cache(cache_dir="cache"):
    """Clear all cache files"""
    if os.path.exists(cache_dir):
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleared cache directory: {cache_dir}")

def clear_specific_cache(pattern, cache_dir="cache"):
    """Clear cache files matching a pattern"""
    if os.path.exists(cache_dir):
        for filename in os.listdir(cache_dir):
            if pattern in filename:
                file_path = os.path.join(cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed cache file: {file_path}")