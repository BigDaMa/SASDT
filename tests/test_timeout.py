from contextlib import contextmanager, redirect_stdout
import csv
import io
import json
from pathlib import Path
import signal
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gdt.cli import main, parse_arguments
from gdt.config import Config, ExperimentConfig
from gdt.diagnostics import split_phase
from gdt.direct_benchmark import run_direct_benchmark
from gdt.results import EvaluationResult
from gdt.timeout import DatasetTimeoutError, dataset_time_limit
from LLMUtil.providers import _retry, request_split_batch


@contextmanager
def simulated_linux_alarm():
    """Deliver the installed handler synchronously, including on Windows."""
    handlers = {14: signal.SIG_DFL}

    def install(number, handler):
        previous = handlers[number]
        handlers[number] = handler
        return previous

    api = SimpleNamespace(SIGALRM=14, signal=Mock(side_effect=install), alarm=Mock())
    with patch("gdt.timeout.sys", SimpleNamespace(platform="linux")), \
            patch("gdt.timeout.signal", api):
        yield api, lambda: handlers[14](14, None)


class TimeoutTests(unittest.TestCase):
    def tearDown(self):
        Config.apply(ExperimentConfig())

    def test_default_and_override(self):
        self.assertEqual(parse_arguments([]).config.timeout_seconds, 600)
        self.assertEqual(parse_arguments(["--timeout", "12"]).config.timeout_seconds, 12)

    def test_deadline_bypasses_all_retry_modes_and_restores_alarm(self):
        for factor in (None, 2.5):
            for individual in (False, True):
                with self.subTest(factor=factor, individual=individual), \
                        simulated_linux_alarm() as (api, expire), \
                        patch.object(Config, "query_individual_factor", factor), \
                        patch.object(Config, "num_retry", 3):
                    request = Mock(side_effect=expire)
                    with self.assertRaisesRegex(DatasetTimeoutError, "600 seconds") as caught:
                        with dataset_time_limit(600), split_phase("test LM wait"):
                            request_split_batch(
                                lambda _query: _retry(request, num_retry=3),
                                ["Alice", "Bob"], instructions="labels", operation="test",
                                one_at_a_time=individual,
                            )
                    request.assert_called_once()
                    self.assertEqual(caught.exception.phase, "test LM wait")
                    self.assertEqual([call.args[0] for call in api.alarm.call_args_list], [600, 0])
                    self.assertEqual(api.signal.call_args.args, (14, signal.SIG_DFL))

    def test_success_and_ordinary_failure_cancel_alarm(self):
        for failure in (None, ValueError("bad dataset"), KeyboardInterrupt()):
            with self.subTest(failure=failure), simulated_linux_alarm() as (api, _expire):
                try:
                    with dataset_time_limit(600):
                        if failure is not None:
                            raise failure
                except BaseException as error:
                    self.assertIs(error, failure)
                self.assertEqual([call.args[0] for call in api.alarm.call_args_list], [600, 0])
                self.assertEqual(api.signal.call_args.args, (14, signal.SIG_DFL))

    def test_non_linux_does_not_install_alarm(self):
        with patch("gdt.timeout.sys", SimpleNamespace(platform="win32")), \
                patch("gdt.timeout.signal") as api, dataset_time_limit(600):
            pass
        api.signal.assert_not_called()
        api.alarm.assert_not_called()

    def test_all_experiments_report_timeout_and_continue(self):
        for experiment in ("benchmark", "pipeline", "direct-llm", "cluster"):
            with self.subTest(experiment=experiment):
                self._check_reports(experiment, fail_fast=False)

    def test_fail_fast_still_exports_timeout(self):
        for experiment in ("benchmark", "direct-llm"):
            with self.subTest(experiment=experiment):
                self._check_reports(experiment, fail_fast=True)

    def _check_reports(self, experiment, *, fail_fast):
        with tempfile.TemporaryDirectory() as directory, \
                simulated_linux_alarm() as (api, expire):
            root = Path(directory)
            for name in ("01_slow.csv", "02_fast.csv"):
                (root / name).write_text("Alice,A\nBob,B\n", encoding="utf-8")
            visited = []

            def run(path):
                name = Path(path).name
                visited.append(name)
                if name == "01_slow.csv":
                    with split_phase("waiting for LM"):
                        _retry(expire, num_retry=3)
                return EvaluationResult(accuracy=1.0, correct=1, incorrect=0)

            def predict(dataset, _config, _provider):
                run(dataset.path)
                return ["B"]

            def cluster(path, **_kwargs):
                run(path)
                return [{0, 1}, {2}]

            output = root / "reports"
            args = ["--experiment", experiment, "--data-dir", str(root),
                    "--output-dir", str(output), "--num-examples", "1"]
            if fail_fast:
                args.append("--fail-fast")
            console = io.StringIO()
            with patch("gdt.pipeline.PipelineRunner._run", side_effect=run), \
                    patch("gdt.direct_benchmark._predict", side_effect=predict), \
                    patch("gdt.direct_benchmark.check_models"), \
                    patch("Clustering.clustermeta.ClusterMeta.run_cluster", side_effect=cluster), \
                    redirect_stdout(console):
                if experiment == "direct-llm":
                    # The public direct CLI intentionally has no path/fail-fast
                    # overrides; retain timeout coverage through the Python API.
                    results = run_direct_benchmark(ExperimentConfig(
                        data_directory=root, output_directory=output, num_examples=1,
                    ), fail_fast=fail_fast)
                    status = 0 if all(result.succeeded for result in results) else 1
                else:
                    status = main(args)

            self.assertEqual(status, 1)
            self.assertEqual(visited, ["01_slow.csv"] if fail_fast else
                             ["01_slow.csv", "02_fast.csv"])
            error = "DatasetTimeoutError: dataset exceeded 600 seconds"
            self.assertIn(error, console.getvalue())
            diagnostics = (output / "errors.txt").read_text(encoding="utf-8")
            self.assertIn("DATASET: 01_slow.csv", diagnostics)
            self.assertIn("PHASE: waiting for LM", diagnostics)
            self.assertIn(error, diagnostics)
            self.assertNotIn("02_fast.csv", diagnostics)
            progress = (output / "progress.log").read_text(encoding="utf-8")
            self.assertIn('status="failed"', progress)
            self.assertIn(error, progress)
            if not fail_fast:
                self.assertIn('status="success"', progress)
            summary = output / {"cluster": "cluster", "direct-llm": "direct_llm"}.get(
                experiment, ""
            ) / "summary.csv"
            with summary.open(encoding="utf-8", newline="") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), len(visited))
            self.assertEqual(records[0]["error"], error)
            if not fail_fast:
                self.assertEqual(records[1]["error"], "")
            if experiment in ("benchmark", "direct-llm"):
                name = "runs.jsonl" if experiment == "benchmark" else "direct_llm_runs.jsonl"
                records = [json.loads(line) for line in (output / name).read_text().splitlines()]
                self.assertEqual(records[0]["error"], error)
            self.assertEqual([call.args[0] for call in api.alarm.call_args_list],
                             [600, 0] * len(visited))


@unittest.skipUnless(sys.platform.startswith("linux"), "requires real Linux SIGALRM")
class LinuxTimeoutTests(unittest.TestCase):
    def test_blocked_socket_wait_is_interrupted_without_retry(self):
        previous = signal.getsignal(signal.SIGALRM)
        reader, writer = socket.socketpair()
        with reader, writer:
            reader.settimeout(10)
            request = Mock(side_effect=lambda: reader.recv(1))
            started = time.monotonic()
            with self.assertRaises(DatasetTimeoutError):
                with dataset_time_limit(1):
                    _retry(request, num_retry=3)
            elapsed = time.monotonic() - started
        request.assert_called_once()
        self.assertGreaterEqual(elapsed, 0.9)
        self.assertLess(elapsed, 3)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.alarm(0), 0)

    def test_multiple_model_waits_share_one_wall_clock_budget(self):
        completed = []
        started = time.monotonic()
        with self.assertRaises(DatasetTimeoutError):
            with dataset_time_limit(1):
                for index in range(10):
                    time.sleep(0.3)
                    completed.append(index)
        self.assertGreater(len(completed), 0)
        self.assertLess(len(completed), 10)
        self.assertLess(time.monotonic() - started, 3)


if __name__ == "__main__":
    unittest.main()
