import itertools
from typing import Dict, List
from DataAccess.dataaccess import DataAccess
from Clustering.clustermeta import ClusterMeta
from LLMUtil.LLMUtil import OLLAMAUtil, OpenAIUtil
from LLMUtil.ResponseParser import RParser
from Propagate.witness import get_pos_dict, validate_pos_dict
from Propagate.propagate import Propagate
from Utils.IterPrintUtil import IterPrintUtil
from transform.Transform import Transform
from transform.Evaluation import Evaluation
from Utils.config import Config
from DataAccess.datafeature import DataFeature

import os
import pandas as pd
import argparse
import time
import traceback
import signal
import sys


def parse_arguments():
    """Parse command line arguments and update Config class"""
    parser = argparse.ArgumentParser(description='GDT-Full Processing Tool')
    
    parser.add_argument('-c', '--use_cache', 
                       action='store_true', 
                       default=False,
                       help='Enable caching (default: False)')
    
    parser.add_argument('-s', '--save_cache',
                       action='store_true',
                       default=False,
                       help='Save cache after processing (default: False)')
    
    parser.add_argument('-d', '--cache_dir', 
                       type=str, 
                       default='cache',
                       help='Specify cache directory (default: cache)')
    
    parser.add_argument('-pc', '--perfect_cluster', 
                       action='store_true', 
                       default=False,
                       help='Enable perfect clustering (default: False)')
    
    parser.add_argument('-ps', '--perfect_split', 
                       action='store_true', 
                       default=False,
                       help='Enable perfect splitting (default: False)')
    
    parser.add_argument('-po', '--perfect_operator', 
                       action='store_true', 
                       default=False,
                       help='Enable perfect operator (requires -pc/--perfect_cluster, default: False)')

    parser.add_argument('-pd', '--perfect_directory', 
                       type=str, 
                       default='data_perfect',
                       help='Specify directory for perfect data (default: data_perfect)')

    parser.add_argument('-v', '--verbose', 
                       action='store_true', 
                       default=False,
                       help='Enable verbose output (default: False)')
    
    parser.add_argument('-ms', '--moneysaver', 
                       action='store_true', 
                       default=False,
                       help='Enable money saver mode for LLM calls (default: False)')

    args = parser.parse_args()

    if args.use_cache and (args.perfect_cluster or args.perfect_split or args.perfect_operator):
        parser.error("Error: --use_cache/-c cannot be used with --perfect_cluster/-pc, --perfect_split/-ps, or --perfect_operator/-po")

    Config.set_use_cache(args.use_cache)
    Config.set_save_cache(args.save_cache)
    Config.set_cache_dir(args.cache_dir)
    Config.set_perfect_cluster(args.perfect_cluster)
    Config.set_perfect_split(args.perfect_split)
    if args.perfect_operator and not args.perfect_cluster:
        print("Error: --perfect_operator/-po requires --perfect_cluster/-pc to be set. Disabling perfect operator")
        args.perfect_operator = False
    Config.set_perfect_operator(args.perfect_operator)
    Config.set_perfect_directory(args.perfect_directory)
    Config.set_verbose(args.verbose)
    Config.set_money_saver(args.moneysaver)

    print(f"Configuration:")
    print(f"  Use Cache: {Config.use_cache}")
    print(f"  Save Cache: {Config.save_cache}")
    print(f"  Cache Directory: {Config.cache_dir}")
    print(f"  Perfect Cluster: {Config.perfect_cluster}")
    print(f"  Perfect Split: {Config.perfect_split}")
    print(f"  Perfect Operator: {Config.perfect_operator}")
    print(f"  Directory for perfect data: {Config.perfect_directory}")
    print(f"  Verbose: {Config.verbose}")
    print(f"  Money Saver Mode: {Config.money_saver}")
    print("-" * 10)
    print("=" * 10)
    print("-" * 10)

    return args

def signal_handler_timeout(sig, frame):
    print("Time out occurred!")
    raise Exception("Time out occurred!")

if sys.platform.startswith('linux'):
    signal.signal(signal.SIGALRM, signal_handler_timeout)

def run_single(file: str, model: str = None, numexamples: int = 5) -> Dict[str, float]:
    time_before_processing_start = time.time()
    DataAccess.NUM_EXAMPLES = numexamples
    print(f"Processing file: {file}")
    if sys.platform.startswith('linux'):
        signal.alarm(600)  # Set timeout to 600 seconds (10 minutes)
    time_before_data_access_end = time.time()
    print(f"\t\tTime taken for data access: {time_before_data_access_end - time_before_processing_start:.2f} seconds")
    time_data_access_start = time.time()
    data = DataAccess.get_data_from_file(DataAccess.DATA_FOLDER, file)
    examples = DataAccess.get_examples(data, numexamples=numexamples)
    example_input = {i: examples[i]['input'] for i in range(len(examples))}
    example_output = {(i + len(data.index)): examples[i]['output'] for i in range(len(examples))}
    ground_truth = {i: data.iloc[i, 1] for i in range(5, len(data.index))}
    example_strs = {**example_input, **example_output}
    data_cluster = DataAccess.get_data_for_clustering(data)
    time_before_cluster_end = time.time()
    print(f"\t\tTime taken for feature extraction: {time_before_cluster_end - time_data_access_start:.2f} seconds")
    time_cluster_start = time.time()
    if not model or not model.startswith("gpt"):
        split_clusters = ClusterMeta.run_cluster(file, "collapse", "DBSCAN", config = "collapse+prompt_label")
    else:
        split_clusters = ClusterMeta.run_cluster(file, "collapse", "DBSCAN", config = "collapse+prompt_label_gpt")
    mapped_clusters = DataAccess.map_clustered_idx_to_str(split_clusters, data_cluster)
    mapped_clusters = [dict(sorted(cluster.items())) for cluster in mapped_clusters]
    if Config.verbose:
        IterPrintUtil.print(mapped_clusters)
        print("-----END MAP CLUSTER-----")
    time_before_sem_split_end = time.time()
    print(f"\t\tTime taken for mapping clusters: {time_before_sem_split_end - time_cluster_start:.2f} seconds")
    time_post_cluster_start = time.time()
    query_strs = []
    query_str_idx = []
    output_idx_set = split_clusters[-1]
    for cluster in mapped_clusters:
        curr_strs = [cluster[i] for i in cluster.keys() if i < DataAccess.NUM_EXAMPLES]
        curr_idx = [i for i in cluster.keys() if i < DataAccess.NUM_EXAMPLES]
        if not curr_strs and not curr_idx:
            curr_strs = [cluster[list(cluster.keys())[0]]]
            curr_idx = [list(cluster.keys())[0]]
        query_strs.extend(curr_strs)
        query_str_idx.extend(curr_idx)
    for idx in output_idx_set:
        query_strs.append(data_cluster[idx])
        query_str_idx.append(idx)
    query_dict = {query_str_idx[i]: query_strs[i] for i in range(len(query_strs))}
    test_strs = {}
    for cluster in mapped_clusters:
        for idx in cluster.keys():
            if idx not in example_strs.keys() and idx not in test_strs.keys():
                test_strs[idx] = cluster[idx]
    test_strs = dict(sorted(test_strs.items()))

    type_list = DataFeature.get_all_sem_types(file)
    type_list = list(set([RParser.key_normalize_with_underscore(t) for t in type_list]))
    time_post_cluster_end = time.time()
    print(f"\t\tTime taken after clustering: {time_post_cluster_end - time_post_cluster_start:.2f} seconds")
    time_get_split_start = time.time()
    sem_components_selected = OpenAIUtil.get_split(query_dict, file, list(example_input.keys()), list(example_output.keys()), use_type = True, type_list = type_list)
    # print(sem_components_selected)
    if Config.verbose:
        IterPrintUtil.print(sem_components_selected)
        print(len(sem_components_selected))
        print("-----END GET SPLIT-----")
    time_get_split_end = time.time()
    print(f"\t\tTime taken for getting split: {time_get_split_end - time_get_split_start:.2f} seconds")
    time_get_pos_dict_start = time.time()
    comp_pos_dict = get_pos_dict(sem_components_selected, query_strs, query_str_idx, file)
    time_get_pos_dict_end = time.time()
    print(f"\t\tTime taken for getting position dictionary: {time_get_pos_dict_end - time_get_pos_dict_start:.2f} seconds")
    time_validate_pos_dict_start = time.time()
    valid_comp_pos_dict = validate_pos_dict(mapped_clusters, comp_pos_dict, sem_components_selected)
    time_validate_pos_dict_end = time.time()
    print(f"\t\tTime taken for validating position dictionary: {time_validate_pos_dict_end - time_validate_pos_dict_start:.2f} seconds")
    time_propagate_start = time.time()
    propagationDict = Propagate.propagate(mapped_clusters, valid_comp_pos_dict, sem_components_selected)
    time_propagate_end = time.time()
    print(f"\t\tTime taken for propagating: {time_propagate_end - time_propagate_start:.2f} seconds")
    time_prepare_transform_start = time.time()
    example_input_components = {i: propagationDict[i] for i in example_input.keys() if i in propagationDict}
    # input_components = list(dict(sorted(example_input_components.items())).values())
    example_output_components = {i - len(data.index): propagationDict[i] for i in example_output.keys() if i in propagationDict}
    # output_components = list(dict(sorted(example_output_components.items())).values())
    test_components = {test_strs[i]: {"id": i, "components": propagationDict[i]} for i in propagationDict.keys() if i not in example_input.keys() and i not in example_output.keys() and ((i < len(data.index) and i >= 5) or (i >= len(data.index) and i - len(data.index) >= 5))}
    time_prepare_transform_end = time.time()
    print(f"\t\tTime taken for preparing transform: {time_prepare_transform_end - time_prepare_transform_start:.2f} seconds")
    time_transform_start = time.time()
    time_transform_end = time.time()
    print(f"\t\tTime taken for transforming: {time_transform_end - time_transform_start:.2f} seconds")
    time_evaluate_start = time.time()
    transformations = Transform.transform(example_input, example_input_components, example_output, example_output_components, test_components, examples)
    time_evaluate_end = time.time()
    print(f"\t\tTime taken for evaluating: {time_evaluate_end - time_evaluate_start:.2f} seconds")
    time_output_start = time.time()
    evaluation_results = Evaluation.evaluate(transformations, ground_truth)
    outputfilename = f"eval_result_{numexamples}.txt"
    with open(outputfilename, "a") as f:
        f.write(f"File: {file}; Model: {OLLAMAUtil.__MODEL__}; Mode: {OLLAMAUtil.__MODE__}\n")
        f.write(f"\tAccuracy: {evaluation_results['accuracy']}\n")
        f.write(f"\tCorrect: {evaluation_results['correct']}\n")
        f.write(f"\tIncorrect: {evaluation_results['incorrect']}\n")
        f.write("----------\n")
    Evaluation.print_eval_result(evaluation_results)
    print("==========")
    time_output_end = time.time()
    print(f"\t\tTime taken for output: {time_output_end - time_output_start:.2f} seconds")
    time_full_end = time.time()
    print(f"\tTime taken for full processing: {time_full_end - time_before_processing_start:.2f} seconds")
    return evaluation_results

def main():
    time_full_start = time.time()
    args = parse_arguments()

    OLLAMAUtil.set_model("qwen3-4b")
    OLLAMAUtil.set_mode("gen")
    for file in os.listdir(DataAccess.DATA_FOLDER):
        if file.endswith('.csv'):
            DataFeature.data_features = {}
            time_file_start = time.time()
            try:
                run_single(file, numexamples=5)
            except Exception as e:
                time_file_end = time.time()
                print(f"Time taken for file {file}: {time_file_end - time_file_start:.2f} seconds")
                print(f"Error processing file {file}: {e}")
                print(traceback.format_exc())
            print("==========")
            time_file_end = time.time()
            print(f"\tTime taken for file {file}: {time_file_end - time_file_start:.2f} seconds")
    time_full_end = time.time()
    print(f"Time taken for full processing: {time_full_end - time_full_start:.2f} seconds")

def init_eval_df(index: List[str], columns: List[str], columns_attr: List[str] = None) -> pd.DataFrame:
    final_columns = None
    if columns_attr:
        final_columns = []
        for col in columns:
            for attr in columns_attr:
                final_columns.append(f"{col}-{attr}")
    else:
        final_columns = columns
    eval_df = pd.DataFrame(index=index, columns=final_columns)
    return eval_df

def benchmark():
    args = parse_arguments()
    
    output_df_base = "eval_table"
    output_df_suffix = None
    if args.perfect_cluster and args.perfect_split and args.perfect_operator:
        DataAccess.DATA_FOLDER = "data"
        output_df_suffix = "_pcpspo"
    elif args.perfect_cluster and args.perfect_split:
        DataAccess.DATA_FOLDER = "data"
        output_df_suffix = "_pcps"
    elif args.perfect_cluster:
        DataAccess.DATA_FOLDER = "data"
        output_df_suffix = "_pc"
    elif args.perfect_split:
        DataAccess.DATA_FOLDER = "data"
        output_df_suffix = "_ps"
    
    
    model_list = ["qwen3-4b", "llama3.2-3b", "phi4", "gpt"]
    model_mode = ["gen"]
    OLLAMAUtil.set_model("qwen3-4b")
    OLLAMAUtil.set_mode("gen")
    for i in range(1, 6):
        outputfilename = "eval_result.txt"
        with open(outputfilename, "w") as f:
            pass
        result_df = init_eval_df(
            index = [file for file in os.listdir(DataAccess.DATA_FOLDER) if file.endswith('.csv')],
            columns = model_list,
            columns_attr = model_mode
        )
        runtime_df = init_eval_df(
            index = [file for file in os.listdir(DataAccess.DATA_FOLDER) if file.endswith('.csv')],
            columns = model_list,
            columns_attr = model_mode
        )
        print("=x-x==x-x==x-x==x-x==x-x==x-x=")
        print(f"Iteration {i} of benchmark")
        print("=x-x==x-x==x-x==x-x==x-x==x-x=")
        with open(outputfilename, "a") as f:
            f.write(f"===== Iteration {i} of benchmark =====\n")
        for file in os.listdir(DataAccess.DATA_FOLDER):
            if file.endswith('.csv'):
                print("===============================")
                print(f"Starting processing for file: {file}")
                print("===============================")
                for model, mode in itertools.product(model_list, model_mode):
                    print("---------------------------")
                    print(f"Model: {model}")
                    print(f"Mode: {mode}")
                    print("---------------------------")
                    if model.startswith("gpt"):
                        OLLAMAUtil.set_model(model)
                        OLLAMAUtil.set_mode(mode)
                        try:
                            starttime = time.time()
                            result_dict = run_single(file, 'gpt', numexamples=i)
                            endtime = time.time()
                            duration = endtime - starttime
                        except Exception as e:
                            print(f"Error processing file {file}: {e}")
                            print(traceback.format_exc())
                            result_dict = {"accuracy": 0.0, "correct": 0, "incorrect": 0}
                            duration = -1
                    else:
                        OLLAMAUtil.set_model(model)
                        OLLAMAUtil.set_mode(mode)
                        OLLAMAUtil.unload_model()
                        DataFeature.data_features = {}
                        try:
                            starttime = time.time()
                            result_dict = run_single(file, numexamples=i)
                            endtime = time.time()
                            duration = endtime - starttime
                        except Exception as e:
                            print(f"Error processing file {file}: {e}")
                            print(traceback.format_exc())
                            result_dict = {"accuracy": 0.0, "correct": 0, "incorrect": 0}
                            duration = -1
                    runtime_df.at[file, f"{model}-{mode}"] = duration
                    result_df.at[file, f"{model}-{mode}"] = result_dict['accuracy']
                    print("==========\n")
                    print("==========")
                print("xxxxxxxxxxxxxxxxxxx")
                print(f"Finished processing for file: {file}")
                print("xxxxxxxxxxxxxxxxxxx")
        result_df.to_csv(f"{output_df_base}_iter{i}{output_df_suffix}.csv")
        runtime_df.to_csv(f"{output_df_base}_runtime_iter{i}{output_df_suffix}.csv")
        with open(outputfilename, "a") as f:
            f.write(f"===== End of Iteration {i} =====\n\n")


if __name__ == "__main__":
    benchmark()
