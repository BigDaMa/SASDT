# SASDT

Experiments for SASDT.

## Setup

Use Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
```

API keys are stored as plaintext in the project root. Each file contains only its key. 

- `openaiapi.txt` for OpenAI



## Dataset

Every dataset in `data_shuffled` is a headerless two-column CSV:

1. source representation
2. target representation

Ground-truth files live in `data_perfect` and use the base dataset stem (without
the shuffle suffix `_0`, `_1`, or `_2`) plus a suffix:

- `_cluster.txt`: one string per line, with `[ITEM]` between clusters
- `_types.txt`: one semantic type per line
- `_split.txt`: original string followed by `type: value` lines; `[ITEM]` between blocks
- `_op.txt`: cluster representative followed by serialized boundary-operator pairs;
  `[PARAMSPLIT]` separates the boundaries and `[ITEM]` separates blocks

## Running experiments

### Full benchmark: GPT-5.4 discovery, three extraction models, 1–5 examples

Put the headerless dataset CSVs in `data_shuffled_final`, put the OpenAI API key
in `openaiapi.txt`, and start your Ollama server on `http://localhost:11434`.
Run `ollama list` to check that the existing model names below are available.
The repository preserves the names from the earlier code base:

| CSV column | CLI extraction model | API/Ollama model name |
| --- | --- | --- |
| Qwen3.5-4B | `qwen3.5-4b` | Ollama `qwen3.5-4b` |
| Granite4.2-3B | `granite4.2-3b` | Ollama `granite4.2-3b` |
| Ministral3 | `ministral3-3b` | Ollama `ministral3-3b` |


Run the full benchmark with this single-line command:

```bash
python main.py --experiment benchmark --data-dir data_shuffled --output-dir results -slm gpt-5.4 -slm qwen3-4b -slm granite4.2-3b -slm ministral3-3b --example-counts 1,2,3,4,5 --money-saver
```

Omit `--money-saver` to query GPT independently for each run. Add
`--dataset Names.csv` to restrict the benchmark; repeat `--dataset` for more files.
Each dataset needs at least six rows to have held-out inputs at any example
count. Rows are used in their existing order; this command does not shuffle.

`-llm` selects the OpenAI model that discovers semantic compositions for the example inputs and outputs. `-slm` selects the exact-substring extraction model.
When it is an SLM, that same model also predicts compositions for the remaining inputs. It uses two Ollama requests disable thinking with `think: false`. 

Models used by the experiments are automatically parsed and case-insensitive. For example, GPT5.4, gpt-5.4, gpt 5.4 point to the same OpenAI model GPT-5.4.

Regenerate tables from saved results without calling any LLM:

```bash
python -m gdt.export_results results/my-run/summary.csv --output-dir results/my-run --example-counts 1,2,3,4,5
```

### Smaller runs and alternative experiments

```bash
python main.py --experiment benchmark --data-dir data_shuffled --dataset Names.csv -slm qwen3.5-4b -n 2
python main.py --experiment direct-llm --provider openai -llm GPT5.4 -n 1,2,3,4,5
```

### Ablation Studies

Use these short flags in addition to provide ground truths:

```bash
python main.py --experiment benchmark --data-dir data_shuffled -slm gpt-5.4 -n 1,2,3,4,5 **-pg**
python main.py --experiment benchmark --data-dir data_shuffled -slm gpt-5.4 -n 1,2,3,4,5 **-ps**
python main.py --experiment benchmark --data-dir data_shuffled -slm gpt-5.4 -n 1,2,3,4,5 **-pg -ps**
python main.py --experiment benchmark --data-dir data_shuffled -slm gpt-5.4 -n 1,2,3,4,5 **-pg -ps -pp**
```

| Flags | Provided ground truth | Remaining inference |
| --- | --- | --- |
| `-pg` | Grouping | Provide ground truth for grouping from `ds_types.txt` |
| `-ps` | Splits | Provide ground truth for semantic splits from `ds_split.txt` |
| `-pg -ps` | Grouping and splits | |
| `-pp` (also enables `-pg -ps`, omitting these has no effect) | Grouping, splits, and operators | Ground truth for operators are provided from `ds_op.txt` |