from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gdt.config import ExperimentConfig
from gdt.diagnostics import split_phase
from gdt.experiments import run_benchmark
from gdt.progress import ProgressLog, progress_phase
from gdt.results import EvaluationResult, RunResult


class ProgressTests(unittest.TestCase):
    def test_events_are_visible_during_run_and_interruption_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            log = ProgressLog(directory, "pipeline", 1)
            with self.assertRaises(KeyboardInterrupt):
                with log.run("Names.csv", ExperimentConfig()):
                    with split_phase("semantic discovery"):
                        content = log.path.read_text(encoding="utf-8")
                        self.assertIn("START", content)
                        self.assertIn('phase="semantic discovery"', content)
                        self.assertNotIn("END", content)
                        raise KeyboardInterrupt()
            content = log.path.read_text(encoding="utf-8")
            self.assertIn('status="interrupted"', content)
            progress_phase("outside run")
            self.assertEqual(content, log.path.read_text(encoding="utf-8"))

    def test_benchmark_records_config_failure_and_fail_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("Names.csv").write_text("Alice,A\nBob,B\n", encoding="utf-8")
            config = ExperimentConfig(
                data_directory=root, output_directory=root / "output",
                num_examples=1, llm_model="gpt-4.1",
            )
            failure = RunResult("Names.csv", "phi4", "gen", 1, 2.5, error="bad response")
            with patch("gdt.experiments.PipelineRunner.run", return_value=failure), \
                 redirect_stdout(io.StringIO()):
                run_benchmark(config, models=["phi4", "gpt-5.1"], fail_fast=True)
            content = (config.output_directory / "progress.log").read_text(encoding="utf-8")
            self.assertIn('run="1/2"', content)
            self.assertIn('"llm": "gpt-4.1"', content)
            self.assertIn('"slm": "phi4"', content)
            self.assertIn('status="failed"', content)
            self.assertIn('error="bad response"', content)
            self.assertEqual(content.count(" START |"), 1)
            self.assertEqual(content.count(" END |"), 1)

    def test_new_invocation_resets_log_and_success_includes_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            log = ProgressLog(directory, "pipeline", 1)
            log.write("OLD")
            log = ProgressLog(directory, "pipeline", 1)
            with log.run("Names.csv", ExperimentConfig()) as current:
                current.finish(RunResult(
                    "Names.csv", "gpt-5.1", "gen", 1, 1.25,
                    evaluation=EvaluationResult(1.0, 1, 0),
                ))
            content = log.path.read_text(encoding="utf-8")
            self.assertNotIn("OLD", content)
            self.assertIn('status="success"', content)
            self.assertIn("accuracy=1.0", content)
            self.assertIn("runtime_seconds=1.25", content)
