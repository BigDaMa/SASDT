from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import ollamatest
from gdt.cluster_experiment import run_cluster_experiment
from gdt.config import Config, ExperimentConfig
from gdt.pipeline import PipelineRunner
from gdt.results import EvaluationResult
from gdt.timing import _active_timer
from LLMUtil.ollama import OLLAMAUtil
from tests.test_timeout import simulated_linux_alarm


class OllamaLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(Config.apply, ExperimentConfig())
        self.addCleanup(OLLAMAUtil.set_model, OLLAMAUtil.__MODEL__)
        self.addCleanup(OLLAMAUtil.set_mode, OLLAMAUtil.__MODE__)
        OLLAMAUtil.set_model("qwen3.5-4b")
        OLLAMAUtil.set_mode("generate")
        self.now = 0.0
        self.reply = Mock()
        self.reply.json.return_value = {
            "response": "[name]", "thinking": "THINKING SHOULD NOT APPEAR",
            "message": {"content": "[name]", "thinking": "THINKING SHOULD NOT APPEAR"},
        }

    def test_thinking_disabled_and_excluded_from_generate_and_chat_answers(self):
        for mode in ("generate", "chat"):
            with self.subTest(mode=mode):
                OLLAMAUtil.set_model("qwen3.5-4b")
                OLLAMAUtil.set_mode(mode)
                post = Mock(return_value=self.reply)
                with patch.dict("sys.modules", {"requests": SimpleNamespace(post=post)}):
                    with OLLAMAUtil.dataset_session():
                        OLLAMAUtil.load_model()
                        self.assertEqual(
                            OLLAMAUtil.get_response_batch(["Alice", "Bob"]),
                            ["[name]", "[name]"],
                        )
                        # Cleanup must target the model actually used, even if
                        # provider configuration changes before the scope ends.
                        OLLAMAUtil.set_model("unused-model")
                self.assertEqual(post.call_count, 4)
                for call in post.call_args_list:
                    self.assertIs(call.kwargs["json"]["think"], False)
                    self.assertIs(call.kwargs["json"]["stream"], False)
                    self.assertEqual(call.kwargs["json"]["model"], "qwen3.5:4b")
                for call in post.call_args_list[:-1]:
                    self.assertNotIn("keep_alive", call.kwargs["json"])
                self.assertTrue(post.call_args_list[1].args[0].endswith(f"/api/{mode}"))
                self.assertTrue(post.call_args.args[0].endswith("/api/generate"))
                self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], 0)

    def test_thinking_is_not_a_fallback_for_a_missing_or_empty_answer(self):
        for mode in ("generate", "chat"):
            for include_empty_answer in (False, True):
                with self.subTest(mode=mode, empty_answer=include_empty_answer):
                    OLLAMAUtil.set_mode(mode)
                    body = {"thinking": "[name]", "message": {"thinking": "[name]"}}
                    if include_empty_answer:
                        body["response"] = ""
                        body["message"]["content"] = ""
                    self.reply.json.return_value = body
                    post = Mock(return_value=self.reply)
                    with patch.dict("sys.modules", {"requests": SimpleNamespace(post=post)}):
                        self.assertEqual(OLLAMAUtil.get_response_batch(["Alice"]), [""])

    def test_run_without_ollama_requests_does_not_offload_any_model(self):
        with patch.object(OLLAMAUtil, "unload_model") as unload:
            with OLLAMAUtil.dataset_session():
                pass
        unload.assert_not_called()

    def test_pipeline_offload_follows_timer_and_deadline_on_all_exit_paths(self):
        for outcome in ("success", "failure", "timeout", "interrupt"):
            for unload_fails in (False, True):
                with self.subTest(outcome=outcome, unload_fails=unload_fails), \
                        tempfile.TemporaryDirectory() as directory, \
                        simulated_linux_alarm() as (alarm, expire):
                    self.now = 0.0
                    events = []
                    warning = io.StringIO()

                    def post(_url, *, json, timeout):
                        self.assertIs(json["think"], False)
                        if json.get("keep_alive") == 0:
                            events.append("offload")
                            self.assertIsNone(_active_timer.get())
                            self.assertEqual(alarm.alarm.call_args.args, (0,))
                            self.now += 900  # Longer than the dataset allowance.
                            if unload_fails:
                                raise RuntimeError("offload failed")
                        else:
                            events.append("query")
                            self.now += 3
                            if outcome == "failure":
                                raise ValueError("request failed")
                            if outcome == "timeout":
                                expire()
                            if outcome == "interrupt":
                                raise KeyboardInterrupt()
                        return self.reply

                    def run(_path):
                        OLLAMAUtil.get_response_batch(["Alice"])
                        return EvaluationResult(1.0, 1, 0)

                    config = ExperimentConfig(model="qwen3.5-4b", output_directory=Path(directory))
                    with patch.dict("sys.modules", {"requests": SimpleNamespace(post=post)}), \
                            patch("gdt.timing.time.perf_counter", side_effect=lambda: self.now), \
                            patch.object(PipelineRunner, "_run", side_effect=run), \
                            redirect_stderr(warning):
                        if outcome == "interrupt":
                            with self.assertRaises(KeyboardInterrupt):
                                PipelineRunner(config).run("names.csv")
                        else:
                            result = PipelineRunner(config).run("names.csv")
                            self.assertEqual(result.duration_seconds, 3)
                            self.assertEqual(result.succeeded, outcome == "success")
                            if outcome == "failure":
                                self.assertIn("ValueError: request failed", result.error)
                            elif outcome == "timeout":
                                self.assertIn("DatasetTimeoutError", result.error)
                    self.assertEqual(events, ["query", "offload"])
                    self.assertEqual([c.args[0] for c in alarm.alarm.call_args_list], [600, 0])
                    self.assertEqual("offload failed" in warning.getvalue(), unload_fails)

    def test_cluster_experiment_offloads_after_each_dataset_outside_both_timers(self):
        with tempfile.TemporaryDirectory() as directory, \
                simulated_linux_alarm() as (alarm, _expire):
            root = Path(directory)
            for name in ("one.csv", "two.csv"):
                (root / name).write_text("Alice,A\nBob,B\n", encoding="utf-8")
            events = []

            def post(_url, *, json, timeout):
                if json.get("keep_alive") == 0:
                    self.assertIsNone(_active_timer.get())
                    self.assertEqual(alarm.alarm.call_args.args, (0,))
                    self.now += 900
                    events.append("offload")
                else:
                    self.now += 2
                    events.append("query")
                return self.reply

            def cluster(*_args, **_kwargs):
                OLLAMAUtil.get_response_batch(["Alice"])
                return [{0, 1}, {2}]

            config = ExperimentConfig(
                data_directory=root, output_directory=root / "reports",
                perfect_directory=root / "truth", num_examples=1, model="qwen3.5-4b",
            )
            with patch.dict("sys.modules", {"requests": SimpleNamespace(post=post)}), \
                    patch("gdt.timing.time.perf_counter", side_effect=lambda: self.now), \
                    patch("Clustering.clustermeta.ClusterMeta.run_cluster", side_effect=cluster), \
                    redirect_stdout(io.StringIO()):
                results = run_cluster_experiment(config)
            self.assertEqual([r.duration_seconds for r in results], [2, 2])
            self.assertTrue(all(r.succeeded for r in results))
            self.assertEqual(events, ["query", "offload", "query", "offload"])

    def test_standalone_report_excludes_offload_and_keeps_console_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "ollamatest.txt"
            events = []
            stdout, stderr = io.StringIO(), io.StringIO()

            def post(_url, *, json, timeout):
                self.assertIs(json["think"], False)
                if json.get("keep_alive") == 0:
                    events.append((json["model"], "offload"))
                    self.now += 900
                    raise RuntimeError("offload failed")
                events.append((json["model"], "query"))
                self.now += 2
                return self.reply

            selected_models = ["qwen3.5:4b", "granite4.2-3b"]
            with patch.dict("sys.modules", {"requests": SimpleNamespace(post=post)}), \
                    patch.object(ollamatest, "RESULTS_PATH", report), \
                    patch.object(ollamatest, "input_text", ["Alice", "Bob"]), \
                    patch.object(ollamatest, "models", selected_models), \
                    patch("ollamatest.time.perf_counter", side_effect=lambda: self.now), \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(ollamatest.main(), 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            content = report.read_text(encoding="utf-8")
            self.assertNotIn("THINKING SHOULD NOT APPEAR", content)
            self.assertEqual(content.count("Total runtime: 4.000000 seconds"), 2)
            self.assertEqual(content.count("Mean runtime: 2.000000 seconds"), 2)
            self.assertEqual(content.count("offload failed"), 2)
            self.assertEqual(events, [
                (model, action)
                for model in selected_models
                for action in ("query", "query", "offload")
            ])


if __name__ == "__main__":
    unittest.main()
