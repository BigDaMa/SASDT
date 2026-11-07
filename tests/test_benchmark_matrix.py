from __future__ import annotations

from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from gdt.config import Config, ExperimentConfig
from gdt.experiments import model_column, run_benchmark, write_benchmark_tables, write_summary
from gdt.export_results import main as export_results
from gdt.results import EvaluationResult, RunResult, append_jsonl
from gdt.pipeline import PipelineRunner
from LLMUtil.ollama import OLLAMAUtil
from LLMUtil.providers import OpenAIUtil
from LLMUtil.semantic import OpenAIUtil_Prompt
from Utils.BenchmarkCache import BenchmarkCache, cache_gpt_splits


class BenchmarkMatrixTests(unittest.TestCase):
    def setUp(self):
        BenchmarkCache.clear()
        self.addCleanup(BenchmarkCache.clear)

    def test_tables_average_datasets_not_repetitions_and_include_failures(self):
        def result(dataset, accuracy, runtime, error=None):
            return RunResult(dataset, "gpt-5.1", "gen", 2, runtime,
                             evaluation=EvaluationResult(accuracy, 1, 0), error=error)
        results = [result("A.csv", .5, 10), result("A.csv", 1, 20),
                   result("B.csv", .25, 5), result("C.csv", 0, 7, "failed")]
        with tempfile.TemporaryDirectory() as directory:
            write_benchmark_tables(results, directory, [1, 2])
            with Path(directory, "Accuracy-2.csv").open() as handle:
                accuracy = list(csv.reader(handle))
            with Path(directory, "Runtime-2.csv").open() as handle:
                runtime = list(csv.reader(handle))
            self.assertEqual(accuracy[0], ["Dataset", "GPT-5.1"])
            self.assertEqual(accuracy[1], ["A.csv", "0.75"])
            self.assertAlmostEqual(float(accuracy[-1][1]), 1 / 3)
            self.assertEqual(float(runtime[-1][1]), 9.0)
            self.assertEqual(float(accuracy[3][1]), 0)
            self.assertTrue(Path(directory, "Accuracy-1.csv").is_file())

    def test_tables_include_only_recorded_datasets_for_each_count(self):
        results = [
            RunResult("Custom.csv", "gpt-5.1", "gen", 1, 10,
                      evaluation=EvaluationResult(.5, 1, 1)),
            RunResult("Failed.csv", "gpt-5.1", "gen", 1, 7, error="failed"),
            RunResult("Other.csv", "gpt-5.1", "gen", 2, 20,
                      evaluation=EvaluationResult(1, 2, 0)),
        ]
        # A supplied dataset order must neither add unprocessed rows nor omit results.
        for datasets in (None, ["Unprocessed.csv", "Other.csv", "Custom.csv"]):
            with self.subTest(datasets=datasets), tempfile.TemporaryDirectory() as directory:
                write_benchmark_tables(results, directory, [1, 2, 3], datasets)
                for count, expected in [
                    (1, ["Custom.csv", "Failed.csv"]), (2, ["Other.csv"]), (3, []),
                ]:
                    for metric in ("Accuracy", "Runtime"):
                        with Path(directory, f"{metric}-{count}.csv").open() as handle:
                            rows = list(csv.DictReader(handle))
                        self.assertEqual([row["Dataset"] for row in rows], [*expected, "mean"])
                        if count == 1:
                            self.assertEqual(float(rows[1]["GPT-5.1"]),
                                             0 if metric == "Accuracy" else 7)
                            self.assertEqual(float(rows[-1]["GPT-5.1"]),
                                             .25 if metric == "Accuracy" else 8.5)

    def test_fail_fast_tables_omit_datasets_and_counts_not_reached(self):
        failed = RunResult("A.csv", "gpt-5.1", "gen", 1, 7, error="failed")
        with tempfile.TemporaryDirectory() as directory, \
             patch("gdt.experiments.PipelineRunner") as runner, redirect_stdout(io.StringIO()):
            for name in ("A.csv", "B.csv"):
                Path(directory, name).write_text("a,b\n")
            runner.return_value.run.return_value = failed
            output = Path(directory, "reports")
            results = run_benchmark(
                ExperimentConfig(data_directory=Path(directory), output_directory=output),
                example_counts=[1, 2], fail_fast=True,
            )
            self.assertEqual(results, [failed])
            runner.return_value.run.assert_called_once_with(Path(directory, "A.csv"))
            for count, expected in [(1, ["A.csv", "mean"]), (2, ["mean"])]:
                for metric in ("Accuracy", "Runtime"):
                    with (output / f"{metric}-{count}.csv").open() as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual([row["Dataset"] for row in rows], expected)
                    if count == 1:
                        self.assertEqual(float(rows[0]["GPT-5.1"]),
                                         0 if metric == "Accuracy" else 7)

    def test_exported_tables_preserve_summary_values_even_with_errors(self):
        results = [
            RunResult("Custom.csv", "gpt-5.1", "gen", 1, 10,
                      evaluation=EvaluationResult(.5, 1, 1)),
            RunResult("Failed.csv", "gpt-5.1", "gen", 1, 7, error="failed"),
            RunResult("Partial.csv", "gpt-5.1", "gen", 1, 9, error="failed after evaluation",
                      evaluation=EvaluationResult(.25, 1, 3)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_summary(results, root)
            for result in results:
                append_jsonl(root / "runs.jsonl", result)
            with (root / "summary.csv").open() as handle:
                summary = list(csv.DictReader(handle))
            for source in (root / "summary.csv", root / "runs.jsonl"):
                with self.subTest(source=source.name):
                    output = root / source.suffix.lstrip(".")
                    export_results([str(source), "--output-dir", str(output)])
                    for metric, field in (("Accuracy", "accuracy"), ("Runtime", "duration_seconds")):
                        with (output / f"{metric}-1.csv").open() as handle:
                            rows = list(csv.DictReader(handle))
                        self.assertEqual([row["Dataset"] for row in rows[:-1]],
                                         [row["dataset"] for row in summary])
                        self.assertEqual([float(row["GPT-5.1"]) for row in rows[:-1]],
                                         [float(row[field]) for row in summary])

    def test_model_headers_keep_versions_and_omit_parameter_counts(self):
        for slm, llm, expected in [
            ("qwen3.5-4b", "GPT5.4", "GPT-5.4+Qwen3.5"),
            ("qwen3.5:4B", "gpt-5.4", "GPT-5.4+Qwen3.5"),
            ("GPT_5.4", "gpt-5.4", "GPT-5.4"),
            ("gpt", "gpt-5.1", "GPT-5.1"),
            ("granite4.2-3b", "gpt-5.4", "GPT-5.4+Granite4.2"),
            ("ministral3-3b", "gpt-5.4", "GPT-5.4+Ministral3"),
            ("llama3.2-3b", "gpt-5.4", "GPT-5.4+LLaMA3.2"),
            ("qwen3-0.6b", "gpt-5.4", "GPT-5.4+Qwen3"),
            ("phi4-mini-3.8b", "gpt-5.4", "GPT-5.4+Phi4-mini"),
            ("CustomModel-4b", "gpt-5.4", "GPT-5.4+CustomModel"),
        ]:
            with self.subTest(slm=slm, llm=llm):
                self.assertEqual(model_column(slm, llm), expected)

    def test_tables_separate_llm_pairs_and_example_counts(self):
        results = [
            RunResult("A.csv", model, "gen", count, runtime, llm_model=llm,
                      evaluation=EvaluationResult(accuracy, 1, 0))
            for model, llm, count, runtime, accuracy in [
                ("qwen3.5-4b", "gpt-5.4", 1, 10, .25),
                ("qwen3.5:4b", "gpt-5.4", 2, 20, .75),
                ("qwen3.5-4b", "gpt-5.1", 1, 30, .5),
                ("gpt-5.4", "gpt-5.4", 2, 40, 1),
            ]
        ]
        header = ["Dataset", "GPT-5.4+Qwen3.5", "GPT-5.1+Qwen3.5", "GPT-5.4"]
        with tempfile.TemporaryDirectory() as directory:
            write_benchmark_tables(results, directory)
            for filename, expected in [
                ("Accuracy-1.csv", ["A.csv", "0.25", "0.5", ""]),
                ("Runtime-1.csv", ["A.csv", "10", "30", ""]),
                ("Accuracy-2.csv", ["A.csv", "0.75", "", "1"]),
                ("Runtime-2.csv", ["A.csv", "20", "", "40"]),
            ]:
                with self.subTest(filename=filename), Path(directory, filename).open() as handle:
                    rows = list(csv.reader(handle))
                    self.assertEqual(rows[0], header)
                    self.assertEqual(rows[1], expected)

    def test_export_summary_includes_all_counts_even_when_tables_are_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "runs.jsonl")
            output = Path(directory, "reports")
            for count in range(1, 6):
                append_jsonl(source, RunResult(
                    f"Custom-{count}.csv", "qwen3.5-4b", "gen", count, count * 10,
                    llm_model="gpt-5.4", evaluation=EvaluationResult(count / 5, count, 5 - count),
                ))
            export_results([str(source), "--output-dir", str(output), "--example-counts", "2"])
            with (output / "summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["num_examples"]) for row in rows], [1, 2, 3, 4, 5])
            self.assertTrue((output / "Accuracy-2.csv").is_file())
            self.assertFalse((output / "Accuracy-1.csv").exists())
            export_results([str(output / "summary.csv"), "--output-dir", str(output)])
            for count in range(1, 6):
                for metric in ("Accuracy", "Runtime"):
                    with (output / f"{metric}-{count}.csv").open() as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual([row["Dataset"] for row in rows],
                                     [f"Custom-{count}.csv", "mean"])

    def test_money_saver_runs_max_examples_and_gpt_first(self):
        calls = []
        class Runner:
            def __init__(self, config):
                self.config = config
            def run(self, path):
                calls.append((self.config.num_examples, self.config.model))
                return RunResult(path.name, self.config.model, "gen", self.config.num_examples,
                                 1, evaluation=EvaluationResult(1, 1, 0))
        with tempfile.TemporaryDirectory() as directory, \
             patch("gdt.experiments.PipelineRunner", Runner), redirect_stdout(io.StringIO()):
            Path(directory, "A.csv").write_text("a,b\nc,d\ne,f\nx,y\n")
            run_benchmark(ExperimentConfig(data_directory=Path(directory),
                                           output_directory=Path(directory, "results"),
                                           money_saver=True),
                          models=["phi4", "gpt-5.1"], example_counts=[1, 2, 3])
            with Path(directory, "results", "summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertEqual({int(row["num_examples"]) for row in rows}, {1, 2, 3})
            for count in (1, 2, 3):
                self.assertTrue(Path(directory, "results", f"Accuracy-{count}.csv").is_file())
                self.assertTrue(Path(directory, "results", f"Runtime-{count}.csv").is_file())
        self.assertEqual(calls, [(3, "gpt-5.1"), (3, "phi4"), (2, "gpt-5.1"),
                                (2, "phi4"), (1, "gpt-5.1"), (1, "phi4")])

    def test_string_cache_reuses_subsets_and_queries_only_missing_strings(self):
        calls = []
        @cache_gpt_splits("values")
        def extract(query_dict, file, input_idx, output_idx, use_type=True, type_list=None):
            calls.append(dict(query_dict))
            return {i: {name: text for name in type_list} for i, text in query_dict.items()}
        with patch.object(Config, "money_saver", True), patch.object(Config, "model", "gpt-5.1"):
            extract({0: "A", 1: "B", 2: "C"}, "file.csv", [], [], type_list=["name", "city"])
            smaller = extract({8: "A", 9: "D"}, "file.csv", [], [], type_list=["name"])
            extract({7: "D"}, "file.csv", [], [], type_list=["name"])
            self.assertEqual(calls, [{0: "A", 1: "B", 2: "C"}, {9: "D"}])
            self.assertEqual(smaller, {8: {"name": "A"}, 9: {"name": "D"}})
            extract({8: "A"}, "file.csv", [], [], type_list=["new_type"])
            self.assertEqual(calls[-1], {8: "A"})
            extract({8: "A"}, "other.csv", [], [], type_list=["name"])
            self.assertEqual(len(calls), 4)

    def test_cached_discovery_also_serves_source_label_requests(self):
        with patch.object(Config, "money_saver", True), \
             patch.object(Config, "use_cache", False), patch.object(Config, "save_cache", False), \
             patch.object(Config, "perfect_split", False), \
             patch.object(Config, "query_individual_factor", 2.5), \
             patch.object(OpenAIUtil_Prompt, "get_semantic_components",
                          return_value='ORIGINAL_STRING: Alice\n[name]\n[ITEM]\n'
                                       'ORIGINAL_STRING: Alice output\n[name]') as split, \
             patch.object(OpenAIUtil, "create_response") as transport:
            OpenAIUtil_Prompt.get_split_prompt({0: "Alice", 6: "Alice output"}, "names.csv", [0], [6])
            labels = OpenAIUtil_Prompt.get_labels_for_types({5: "Alice"}, "names.csv", ["[name]"])
            self.assertEqual(labels, {5: ["[name]"]})
            self.assertEqual(split.call_count, 1)
            transport.assert_not_called()

    def test_ollama_extracts_exact_values_using_historical_model_name(self):
        reply = Mock()
        reply.json.return_value = {"response": 'ORIGINAL_STRING: Alice\nname: Alice'}
        requests = Mock()
        requests.post.return_value = reply
        with patch.object(Config, "use_cache", False), patch.object(Config, "save_cache", False), \
             patch.object(Config, "perfect_split", False), \
             patch.object(Config, "query_individual_factor", 2.5), \
             patch.object(OLLAMAUtil, "__MODEL__", "llama3.2-3b-orig-cpu"), \
             patch.object(OLLAMAUtil, "__MODE__", "gen"), \
             patch.dict("sys.modules", {"requests": requests}):
            result = OLLAMAUtil.get_split({0: "Alice"}, "names.csv", [0], [],
                                          use_type=True, type_list=["name"])
        self.assertEqual(result, {0: {"name": "Alice"}})
        payload = requests.post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "llama3.2-3b-cpu")
        self.assertIn("exact substring", payload["prompt"])
        self.assertNotIn("ORIGINAL_STRING", payload["prompt"])

    def test_money_saver_does_not_cache_ollama_extraction(self):
        reply = Mock()
        reply.json.return_value = {"response": 'ORIGINAL_STRING: Alice\nname: Alice'}
        requests = Mock()
        requests.post.return_value = reply
        with patch.object(Config, "money_saver", True), patch.object(Config, "use_cache", False), \
             patch.object(Config, "save_cache", False), patch.object(Config, "perfect_split", False), \
             patch.object(Config, "query_individual_factor", 2.5), \
             patch.dict("sys.modules", {"requests": requests}):
            for _ in range(2):
                OLLAMAUtil.get_split({0: "Alice"}, "names.csv", [0], [],
                                     use_type=True, type_list=["name"])
        self.assertEqual(requests.post.call_count, 2)

    def test_pipeline_routes_local_models_to_ollama_with_split_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "Names.csv")
            path.write_text("Alice Smith,Smith Alice\n" + "UNUSED,UNUSED\n" * 4
                            + "Bob Brown,Brown Bob\n")
            components = {0: {"first_name": "Alice", "last_name": "Smith"},
                          6: {"last_name": "Smith", "first_name": "Alice"}}
            for model in ["qwen3-4b", "llama3.2-3b-orig-cpu", "phi4"]:
                with self.subTest(model=model), redirect_stdout(io.StringIO()), \
                     patch.object(OLLAMAUtil, "get_split", return_value=components) as local, \
                     patch.object(OpenAIUtil, "get_split") as gpt:
                    result = PipelineRunner(ExperimentConfig(
                        data_directory=path.parent, num_examples=1, model=model, slm_split=True,
                        cluster_config="collapse",
                    )).run(path)
                    self.assertTrue(result.succeeded, result.error)
                    self.assertEqual(result.evaluation.accuracy, 1.0)
                    local.assert_called_once()
                    gpt.assert_not_called()

    def test_openai_exact_extraction_caches_successful_individual_results(self):
        with patch.object(Config, "money_saver", True), patch.object(Config, "model", "gpt-5.1"), \
             patch.object(Config, "use_cache", False), patch.object(Config, "save_cache", False), \
             patch.object(Config, "perfect_split", False), \
             patch.object(Config, "query_individual_factor", 2.5), \
             patch.object(OpenAIUtil, "get_semantic_components", side_effect=[
                 'ORIGINAL_STRING: Alice\nname: Alice',
                 'ORIGINAL_STRING: Bob\nname: Bob',
             ]) as query:
            OpenAIUtil.get_split({0: "Alice"}, "names.csv", [0], [],
                                 use_type=True, type_list=["name"])
            result = OpenAIUtil.get_split({5: "Alice", 6: "Bob"}, "names.csv", [5, 6], [],
                                          use_type=True, type_list=["name"])
            self.assertEqual(result, {5: {"name": "Alice"}, 6: {"name": "Bob"}})
            self.assertEqual(query.call_count, 2)
            self.assertNotIn("Alice", query.call_args.args[0])
            self.assertIn("Bob", query.call_args.args[0])
