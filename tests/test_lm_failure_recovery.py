from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from gdt.config import Config, ExperimentConfig
from gdt.diagnostics import dataset_diagnostics
from gdt.experiments import run_benchmark
from gdt.export_results import load_results
from gdt.pipeline import PipelineRunner
from LLMUtil.ollama import OLLAMAUtil
from LLMUtil.providers import (
    OpenAIUtil, SplitQueryBudget, _exact_component_item_is_valid, request_split_batch,
)
from LLMUtil.ResponseParser import RParser_Prompt
from LLMUtil.semantic import OpenAIUtil_Prompt
from tests.test_split_budget import echoed, originals_in
from transform.Transform import Transform
from Utils.BenchmarkCache import BenchmarkCache
from Utils.CacheUtil import cache_result


class LMFailureRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(Config.apply, ExperimentConfig())
        BenchmarkCache.clear()
        self.addCleanup(BenchmarkCache.clear)

    def test_partial_results_keep_alignment_across_retry_modes(self):
        for budgeted in (False, True):
            for individual in (False, True):
                with self.subTest(budgeted=budgeted, individual=individual), \
                     patch.object(Config, "query_individual_factor", 2 if budgeted else None), \
                     patch.object(Config, "num_retry", 1), tempfile.TemporaryDirectory() as directory:
                    if individual:
                        responses = ["name: A", "name: invalid", "name: C", "name: invalid"]
                    elif budgeted:
                        responses = [echoed("C", "name: C") + "\n[ITEM]\n" + echoed("A", "name: A"),
                                     "bad", "bad", "bad"]
                    else:
                        responses = ["name: A\n[ITEM]\nname: invalid\n[ITEM]\nname: invalid",
                                     "name: invalid\n[ITEM]\nname: invalid\n[ITEM]\nname: C"]
                    with dataset_diagnostics(directory, "test.csv", ExperimentConfig()) as diagnostics:
                        _, items = request_split_batch(
                            Mock(side_effect=responses), ["A", "B", "C"], instructions="extract",
                            operation="test", one_at_a_time=individual,
                            item_is_valid=_exact_component_item_is_valid(["name"]),
                        )
                    self.assertEqual(items, ["name: A", "", "name: C"])
                    self.assertEqual(diagnostics.result_fields()["lm_empty_results"], 1)
                    self.assertIn("EMPTY RESULT for 1", (Path(directory) / "errors.txt").read_text())

    def test_exhausted_shared_budget_preserves_count_without_sending_queries(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(Config, "query_individual_factor", 1.5):
            send = Mock()
            with dataset_diagnostics(directory, "test.csv", ExperimentConfig()) as diagnostics:
                _, items = request_split_batch(
                    send, ["A", "B"], instructions="labels", operation="refinement",
                    budget=SplitQueryBudget(limit=2, used=2), item_is_valid=RParser_Prompt.is_composition,
                )
            send.assert_not_called()
            self.assertEqual(items, ["", ""])
            self.assertTrue(diagnostics.result_fields()["has_lm_failures"])
            self.assertEqual(diagnostics.result_fields()["lm_failed_responses"], 0)
            self.assertEqual(diagnostics.result_fields()["lm_empty_results"], 2)

    def test_nontext_responses_retry_then_return_empty_results(self):
        for response in (None, {}, ["[name]"]):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory, \
                 patch.object(Config, "query_individual_factor", 2):
                send = Mock(return_value=response)
                with dataset_diagnostics(directory, "test.csv", ExperimentConfig()) as diagnostics:
                    _, items = request_split_batch(
                        send, ["A"], instructions="labels", operation="test",
                        item_is_valid=RParser_Prompt.is_composition,
                    )
                self.assertEqual(items, [""])
                self.assertEqual(send.call_count, 2)
                self.assertEqual(diagnostics.result_fields()["lm_failed_responses"], 2)
                self.assertIn("Response is not text", (Path(directory) / "errors.txt").read_text())

    def test_valid_retry_keeps_failure_marker_without_empty_result(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(Config, "query_individual_factor", 2):
            with dataset_diagnostics(directory, "test.csv", ExperimentConfig()) as diagnostics:
                _, items = request_split_batch(
                    Mock(side_effect=["bad", echoed("A", "[name]")]), ["A"],
                    instructions="labels", operation="test", item_is_valid=RParser_Prompt.is_composition,
                )
            self.assertEqual(items, ["[name]"])
            self.assertEqual(diagnostics.result_fields(), {
                "has_lm_failures": True, "lm_failed_responses": 1,
                "lm_empty_results": 0, "lm_tolerated_splits": 0,
            })

    def test_pipeline_keeps_successful_clusters_and_ignores_failed_example_pairs(self):
        for batched in (False, True):
            with self.subTest(batched=batched), tempfile.TemporaryDirectory() as directory:
                path = Path(directory, "Names.csv")
                path.write_text("Alice Smith,Smith Alice\nBob Jones,Jones Bob\n"
                                + "UNUSED,UNUSED\n" * 3
                                + "Carol White,White Carol\nMr. David Green,Green David\n"
                                "Eve Brown,Brown Eve\n", encoding="utf-8")
                config = ExperimentConfig(
                    model="phi4", slm_split=True, num_examples=2, query_individual_factor=1,
                    language_model_batch=batched, data_directory=Path(directory),
                    output_directory=Path(directory, "results"),
                )

                def discover(query, **_kwargs):
                    return "\n[ITEM]\n".join(echoed(value, "[last_name][first_name]"
                        if value in ("Smith Alice", "Jones Bob") else "[first_name][last_name]")
                        for value in originals_in(query))

                def respond(prompts, **_kwargs):
                    prompt = prompts[0]
                    originals = originals_in(prompt) if batched else [prompt.rsplit("Input:\n", 1)[1].strip()]
                    blocks = []
                    for original in originals:
                        if "Examples:" in prompt:
                            body = "[first_name][last_name]"
                        elif original in ("Mr. David Green", "Smith Alice"):
                            body = "first_name: not present"
                        else:
                            first, last = original.split()
                            body = (f"last_name: {first}\nfirst_name: {last}" if original == "Jones Bob"
                                    else f"first_name: {first}\nlast_name: {last}")
                        blocks.append(echoed(original, body) if batched else body)
                    return ["\n[ITEM]\n".join(blocks)]

                with patch.object(OpenAIUtil, "create_response", side_effect=discover), \
                     patch.object(OLLAMAUtil, "get_response_batch", side_effect=respond), \
                     redirect_stdout(io.StringIO()):
                    result = PipelineRunner(config).run(path)
                self.assertTrue(result.succeeded, result.error)
                self.assertTrue(result.has_lm_failures)
                self.assertEqual(result.lm_empty_results, 2)
                self.assertEqual(result.evaluation.correct, 2)
                self.assertEqual(result.evaluation.incorrect, 1)
                rows = result.evaluation.output_items
                self.assertEqual([row["transformation"] for row in rows], ["White Carol", "", "Brown Eve"])
                self.assertEqual(rows[1]["sem. components"], [])

    def test_benchmark_finishes_all_datasets_when_every_model_request_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory, "data")
            data.mkdir()
            for name in ("A.csv", "B.csv"):
                (data / name).write_text("Alice Smith,Smith Alice\n" + "UNUSED,UNUSED\n" * 4
                                         + "Bob Jones,Jones Bob\n", encoding="utf-8")
            config = ExperimentConfig(
                model="phi4", num_examples=1, query_individual_factor=1,
                data_directory=data, output_directory=Path(directory, "results"),
            )
            with patch.object(OpenAIUtil, "create_response", side_effect=RuntimeError("offline")), \
                 patch.object(OLLAMAUtil, "get_response_batch", side_effect=RuntimeError("offline")), \
                 redirect_stdout(io.StringIO()) as console:
                results = run_benchmark(config, models=["phi4"], example_counts=[1], fail_fast=True)
            self.assertEqual(len(results), 2)
            for result in results:
                self.assertTrue(result.succeeded, result.error)
                self.assertTrue(result.has_lm_failures)
                self.assertGreater(result.lm_empty_results, 0)
                self.assertEqual(result.evaluation.accuracy, 0)
                self.assertEqual(len(result.evaluation.output_items), 1)
                self.assertEqual(result.evaluation.output_items[0]["transformation"], "")
            self.assertIn("LM FAILURES:", console.getvalue())
            self.assertIn("success_with_lm_failures", (config.output_directory / "progress.log").read_text())
            for filename in ("summary.csv", "runs.jsonl"):
                reloaded = load_results(config.output_directory / filename)
                self.assertEqual([r.to_record() for r in reloaded], [r.to_record() for r in results])
            with (config.output_directory / "Accuracy-1.csv").open(newline="") as handle:
                table = list(csv.DictReader(handle))
            self.assertEqual(len(table), 3)
            self.assertTrue(any(value == "0.0" for value in table[0].values()))

    def test_no_usable_examples_gives_empty_function(self):
        self.assertEqual(Transform.find_transformation_function({}, {}, {}), [])
        self.assertEqual(Transform.find_transformation_function({0: {}}, {0: {}}, {0: "output"}), [])

    def test_failed_splits_are_not_reused_as_successful_money_saver_results(self):
        with tempfile.TemporaryDirectory() as directory:
            Config.apply(ExperimentConfig(money_saver=True, query_individual_factor=1))
            with patch.object(OpenAIUtil, "create_response", side_effect=[
                "bad", echoed("A", "[name]") + "\n[ITEM]\n" + echoed("B", "[name]"),
            ]) as send:
                with dataset_diagnostics(directory, "test.csv", ExperimentConfig()) as first:
                    split = OpenAIUtil_Prompt.get_split_prompt({0: "A", 1: "B"}, "test.csv", [0], [1])
                self.assertEqual(split, {0: [], 1: []})
                self.assertTrue(first.result_fields()["has_lm_failures"])
                with dataset_diagnostics(directory, "test.csv", ExperimentConfig()) as second:
                    split = OpenAIUtil_Prompt.get_split_prompt({0: "A", 1: "B"}, "test.csv", [0], [1])
                self.assertEqual(split, {0: ["[name]"], 1: ["[name]"]})
                self.assertFalse(second.result_fields()["has_lm_failures"])
                self.assertEqual(send.call_count, 2)

    def test_failed_dataset_does_not_save_derived_disk_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            Config.apply(ExperimentConfig(save_cache=True, cache_directory=Path(directory, "cache")))

            @cache_result()
            def derive():
                return {"derived": "from empty split"}

            with dataset_diagnostics(directory, "test.csv", ExperimentConfig()):
                request_split_batch(Mock(return_value="bad"), ["A"], instructions="labels",
                                    operation="test", item_is_valid=RParser_Prompt.is_composition)
                derive()
            self.assertFalse(Path(directory, "cache").exists())
