from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from gdt.config import Config, ExperimentConfig
from gdt.diagnostics import clear_errors, dataset_diagnostics, split_phase
from gdt.experiments import run_pipeline_experiment
from gdt.pipeline import PipelineRunner
from gdt.results import EvaluationResult
from gdt.timing import PipelineTimer, output_time, stop_pipeline_timer
from LLMUtil.providers import request_split_batch, request_split_with_retry
from LLMUtil.ResponseParser import RParser_Prompt
from Utils.CacheUtil import cache_result
from Utils.IterPrintUtil import IterPrintUtil


class OutputTests(unittest.TestCase):
    def test_failed_phase_logs_only_rejected_responses_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(output_directory=Path(directory), num_retry=1)
            def run(_path):
                with split_phase("discovery"):
                    request_split_with_retry(lambda: "old", prompt="old query",
                                             expected_count=1, operation="initial")
                with split_phase("extraction"):
                    request_split_with_retry(lambda: "good", prompt="successful query",
                                             expected_count=1, operation="initial")
                    request_split_with_retry(Mock(side_effect=["bad one", "bad two"]),
                                             prompt="failing query", expected_count=2,
                                             operation="final", num_retry=1)
                return EvaluationResult(accuracy=0, correct=0, incorrect=2)
            clear_errors(directory)
            with patch.object(PipelineRunner, "_run", side_effect=run), redirect_stdout(io.StringIO()):
                runner = PipelineRunner(config)
                first = runner.run("first.csv")
                second = runner.run("second.csv")
                for result in (first, second):
                    self.assertTrue(result.succeeded)
                    self.assertTrue(result.has_lm_failures)
                    self.assertEqual(result.lm_failed_responses, 2)
                    self.assertEqual(result.lm_empty_results, 2)
            log = (Path(directory) / "errors.txt").read_text(encoding="utf-8")
            self.assertNotIn("old query", log)
            self.assertNotIn("successful query", log)
            self.assertNotIn("good", log)
            for text in ["bad one", "bad two", "first.csv",
                         "second.csv", "PHASE: extraction", "-" * 96, "=" * 96]:
                self.assertIn(text, log)
            self.assertEqual(log.count("QUERY "), 4)
            self.assertEqual(log.count("ERROR: Number of split mismatch. 1/2 splits obtained."), 4)

    def test_recovered_attempts_are_logged_but_successful_attempts_are_not(self):
        for budgeted in (False, True):
            for individual in (False, True):
                with self.subTest(budgeted=budgeted, individual=individual), \
                     tempfile.TemporaryDirectory() as directory:
                    config = ExperimentConfig(output_directory=Path(directory))
                    valid = "[name]" if individual or not budgeted else "ORIGINAL_STRING: Alice\n[name]"
                    with dataset_diagnostics(directory, "test.csv", config), \
                         split_phase("compositions"), \
                         patch.object(Config, "query_individual_factor", 2 if budgeted else None), \
                         patch.object(Config, "num_retry", 1):
                        request_split_batch(
                            Mock(side_effect=["malformed", valid]), ["Alice"],
                            instructions="labels", operation="test", one_at_a_time=individual,
                            item_is_valid=RParser_Prompt.is_composition,
                        )
                    log = (Path(directory) / "errors.txt").read_text(encoding="utf-8")
                    self.assertEqual(log.count("QUERY "), 1)
                    self.assertIn("malformed", log)
                    self.assertNotIn("[name]", log)
                    self.assertIn("ERROR: Output format", log)

    def test_request_errors_explain_missing_headers_counts_and_invalid_contents(self):
        cases = [
            (["A"], "[name]", "Output format error. No ORIGINAL_STRING found."),
            (["A", "B"], "ORIGINAL_STRING: A\n[name]",
             "Number of split mismatch. 1/2 splits obtained."),
            (["A"], "ORIGINAL_STRING: A\nnot labels", "Invalid split contents."),
            (["A"], "ORIGINAL_STRING: unknown\n[name]", "Unknown ORIGINAL_STRING."),
            (["A"], RuntimeError("offline"), "Request failed. RuntimeError."),
        ]
        for originals, response, reason in cases:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                request = (Mock(side_effect=response) if isinstance(response, Exception)
                           else Mock(return_value=response))
                with patch.object(Config, "query_individual_factor", 1), \
                     dataset_diagnostics(directory, "test.csv", ExperimentConfig()) as diagnostics:
                    request_split_batch(request, originals, instructions="labels", operation="test",
                                        item_is_valid=RParser_Prompt.is_composition)
                self.assertTrue(diagnostics.result_fields()["has_lm_failures"])
                log = (Path(directory) / "errors.txt").read_text(encoding="utf-8")
                self.assertIn(reason, log)
                self.assertEqual(log.count("QUERY "), 1)

    def test_new_experiment_clears_stale_errors_even_without_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "errors.txt"
            path.write_text("stale failure", encoding="utf-8")
            with patch("gdt.experiments.discover_csv_files", return_value=[]):
                run_pipeline_experiment(ExperimentConfig(output_directory=Path(directory)))
            self.assertEqual(path.read_text(encoding="utf-8"), "")

    def test_budget_history_includes_partial_results_and_transport_errors(self):
        request = Mock(side_effect=['ORIGINAL_STRING: A\n[name]',
                                    RuntimeError("offline"), "malformed"])
        with tempfile.TemporaryDirectory() as directory, \
             dataset_diagnostics(directory, "test.csv", ExperimentConfig()) as diagnostics, \
             patch.object(Config, "query_individual_factor", 2):
            _, items = request_split_batch(request, ["A", "B"], instructions="labels", operation="test")
        self.assertEqual(items, ["[name]", ""])
        exchanges = diagnostics.exchanges
        self.assertEqual(len(exchanges), 3)
        self.assertIn('ORIGINAL_STRING: A', exchanges[0].prompt)
        self.assertNotIn('ORIGINAL_STRING: A', exchanges[1].prompt)
        self.assertEqual(exchanges[1].response, "RuntimeError: offline")
        self.assertEqual(exchanges[2].response, "malformed")


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        patcher = patch("gdt.timing.time.perf_counter", side_effect=lambda: self.now)
        patcher.start()
        self.addCleanup(patcher.stop)

    def advance(self, seconds):
        self.now += seconds

    def test_nested_output_and_post_transformation_evaluation_are_excluded(self):
        with PipelineTimer() as timer:
            self.advance(3)  # LLM request and response wait.
            with output_time():
                self.advance(10)
                with output_time():
                    self.advance(20)
            self.advance(2)  # Transformation processing.
            stop_pipeline_timer()
            self.advance(100)  # Evaluation after final transformations.
            with output_time():
                self.advance(50)
        self.assertEqual(timer.elapsed, 5)

    def test_slow_console_and_debug_formatting_are_excluded(self):
        class SlowStream(io.StringIO):
            def write(stream, text):
                self.advance(20)
                return super().write(text)
        def format_value(*_args):
            self.advance(40)
            return "formatted"
        stream = SlowStream()
        with redirect_stdout(stream), PipelineTimer() as timer:
            self.advance(5)
            print("library diagnostic")
            with patch.object(IterPrintUtil, "format", side_effect=format_value):
                IterPrintUtil.print({"debug": "value"})
        self.assertEqual(timer.elapsed, 5)
        self.assertIn("formatted", stream.getvalue())

    def test_cache_serialization_and_writes_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            @cache_result(directory)
            def transform():
                self.advance(7)
                return {"result": "value"}
            with patch.object(Config, "use_cache", False), \
                 patch.object(Config, "save_cache", True), \
                 patch.object(Config, "perfect_cluster", False), \
                 patch.object(Config, "perfect_split", False), \
                 patch.object(Config, "perfect_operator", False), \
                 patch("Utils.CacheUtil.pickle.dump", side_effect=lambda *a, **k: self.advance(80)), \
                 redirect_stdout(io.StringIO()), PipelineTimer() as timer:
                self.assertEqual(transform(), {"result": "value"})
            self.assertEqual(timer.elapsed, 7)

    def test_failure_freezes_timer_and_restores_output_context(self):
        with self.assertRaises(ValueError):
            with PipelineTimer() as timer:
                self.advance(4)
                with output_time():
                    self.advance(60)
                    raise ValueError("failure")
        with output_time():
            self.advance(100)
        self.assertEqual(timer.elapsed, 4)
