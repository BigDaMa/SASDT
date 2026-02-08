from time import sleep
import time
from typing import List
import pandas as pd
from LLMUtil.LLMUtil import OpenAIUtil, NovitaUtil
import os
from unidecode import unidecode
import random
import string

def benchmark_gpt(file: str, temperature: float = 0.1, numexamples: int = 5):
    time_start = time.time()
    client = OpenAIUtil.get_client()
    with open("gpt_benchmark_results.txt", "a") as f:
        f.write(f"Benchmarking file: {file}\n")
    df = pd.read_csv(file, header=None)
    examples = {"input": [], "output": []}
    for i in range(numexamples):
        question = df.iloc[i, 0]
        answer = df.iloc[i, 1]
        examples["input"].append(question)
        examples["output"].append(answer)
    
    queries = []
    ground_truths = []
    for i in range(5, len(df)):
        query = df.iloc[i, 0]
        ground_truth = df.iloc[i, 1]
        queries.append(query)
        ground_truths.append(ground_truth)
    
    predictions = []
    instruction = "Answer the following questions, each in a single line with no additional information. Do not repeat the question."
    prompt = ""
    for q, a in zip(examples["input"], examples["output"]):
        prompt += f"Q: {q} A: {a}\n"
    for query in queries:
        prompt += f"Q: {query} A:\n"
    prompt = prompt.removesuffix("\n")
    request_kwargs = {
        "model": "gpt-5.1",
        "instructions": instruction,
        "input": prompt,
    }
    time_preprocess_end = time.time()
    print(f"\t\tTime taken for preprocessing: {time_preprocess_end - time_start:.2f} seconds")
    time_response_start = time.time()
    response = client.responses.create(**request_kwargs)
    time_response_end = time.time()
    # print(response)
    print(f"\t\tTime taken for response: {time_response_end - time_response_start:.2f} seconds")
    time_postprocess_start = time.time()
    result = response.output_text.strip()
    predictions = result.split("\n")
    time_postprocess_end = time.time()
    print(f"\t\tTime taken for postprocessing: {time_postprocess_end - time_postprocess_start:.2f} seconds")
    for i in range(len(predictions)):
        if "A: " in predictions[i]:
            predictions[i] = unidecode(predictions[i].split("A: ")[1].strip(), errors='replace', replace_str='')
        else:
            predictions[i] = unidecode(predictions[i].strip(), errors='replace', replace_str='')
    # Calculate accuracy
    time_accuracy_start = time.time()
    correct = 0
    total = len(ground_truths)
    if len(predictions) != len(ground_truths):
        predictions = predictions[:min(len(predictions), len(ground_truths))]
        ground_truths = ground_truths[:min(len(predictions), len(ground_truths))]
    with open("gpt_benchmark_results.txt", "a") as f:
        for pred, gt in zip(predictions, ground_truths):
            if pred.strip() == gt.strip():
                correct += 1
            else:
                f.write(f"Prediction: {pred.strip()}\nGround Truth: {gt.strip()}\n")
    time_accuracy_end = time.time()
    print(f"\t\tTime taken for accuracy: {time_accuracy_end - time_accuracy_start:.2f} seconds")
    accuracy = correct / total if total > 0 else 0
    time_write_start = time.time()
    with open("gpt_benchmark_results_runtime.txt", "a") as f:
        f.write(f"File: {file}, Accuracy: {accuracy:.2%}\n==========\n")
    time_write_end = time.time()
    print(f"\t\tTime taken for writing: {time_write_end - time_write_start:.2f} seconds")
    print(f"File: {file}, Accuracy: {accuracy:.2%}")
    time_full_end = time.time()
    print(f"\tTime taken for full processing: {time_full_end - time_start:.2f} seconds")
    return accuracy

def benchmark():
    accuracy_dict = dict()
    runtime_dict = dict()
    mean_acc = dict()
    mean_runtime = dict()
    models = ['gpt']
    for model in models:
        with open(f"{model}_benchmark_results_runtime.txt", "w") as f:
            pass
        for file in os.listdir('data_shuffled'):
            if not file.endswith('.csv'):
                continue
            accuracy_dict[file] = []
            runtime_dict[file] = []
            mean_acc[file] = []
            mean_runtime[file] = []
            file_path = os.path.join('data_shuffled', file)
            for i in range(5):
                time_gpt_start = time.time()
                accuracy = benchmark_gpt(file_path, numexamples=i)
                time_gpt_end = time.time()
                print(f"\tTime taken for GPT using {i} examples: {time_gpt_end - time_gpt_start:.2f} seconds")
                mean_acc[file].append(accuracy)
                mean_runtime[file].append(time_gpt_end - time_gpt_start)
                accuracy_dict[file].append(accuracy)
                runtime_dict[file].append(time_gpt_end - time_gpt_start)
        mean_acc['mean'] = []
        mean_runtime['mean'] = []
        for i in range(5):
            accs = [mean_acc[file][i] for file in mean_acc if file != 'mean']
            runtimes = [mean_runtime[file][i] for file in mean_runtime if file != 'mean']
            mean_runtime_value = sum(runtimes) / len(runtimes) if len(runtimes) > 0 else 0
            mean_accuracy = sum(accs) / len(accs) if len(accs) > 0 else 0
            mean_acc['mean'].append(mean_accuracy)
            mean_runtime['mean'].append(mean_runtime_value)
        
        accuracy_dict['mean'] = mean_acc['mean']
        runtime_dict['mean'] = mean_runtime['mean']
        final_list = []
        final_runtime = []
        for k, vlist in accuracy_dict.items():
            final_list.append([k] + vlist)
        for k, vlist in runtime_dict.items():
            final_runtime.append([k] + vlist)
        accuracy_df = pd.DataFrame(final_list, columns=['Dataset'] + [f'Accuracy_{i+1}' for i in range(5)])
        runtime_df = pd.DataFrame(final_runtime, columns=['Dataset'] + [f'Runtime_{i+1}' for i in range(5)])
        accuracy_df.to_csv(f"{model}_benchmark_summary.csv", index=False)
        runtime_df.to_csv(f"{model}_benchmark_runtime_summary.csv", index=False)


def init_eval_df(index: List[str], columns: List[str]) -> pd.DataFrame:
    final_columns = columns
    eval_df = pd.DataFrame(index=index, columns=final_columns)
    return eval_df

if __name__ == "__main__":
    benchmark()