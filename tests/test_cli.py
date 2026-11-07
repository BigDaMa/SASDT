from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gdt.cli import build_parser, main, parse_arguments
from gdt.config import Config, ExperimentConfig


class CliTests(unittest.TestCase):
    def tearDown(self) -> None:
        Config.apply(ExperimentConfig())

    def test_model_defaults(self) -> None:
        args = parse_arguments([])
        self.assertEqual(args.config.llm_model, "gpt-5.1")
        self.assertEqual(args.config.model, "gpt-5.1")
        self.assertFalse(args.config.language_model_batch)
        self.assertFalse(Config.language_model_batch)
        self.assertFalse(args.config.slm_split)
        self.assertFalse(Config.slm_split)

    def test_slm_split_is_hidden_and_presence_only(self) -> None:
        help_text = build_parser().format_help()
        self.assertNotIn("--slm-split", help_text)
        self.assertNotRegex(help_text, r"(?<!\S)-sp(?=[\s,]|$)")
        for alias in ("-sp", "--slm-split"):
            with self.subTest(alias=alias):
                args = parse_arguments([alias, "-slm", "phi4"])
                self.assertTrue(args.config.slm_split)
                self.assertTrue(Config.slm_split)
                self.assertFalse(args.config.language_model_batch)
                self.assertNotIn("slm_split", args.config.as_display_dict())
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments([alias, "true"])

    def test_slm_batch_aliases_are_presence_only_flags(self) -> None:
        for alias in ("-lb", "--language-model-batch"):
            with self.subTest(alias=alias):
                args = parse_arguments([alias])
                self.assertTrue(args.config.language_model_batch)
                self.assertTrue(Config.language_model_batch)
            for value in ("true", "false"):
                with self.subTest(alias=alias, value=value), \
                     redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_arguments([alias, value])

    def test_independent_models_reach_benchmark(self) -> None:
        with patch("gdt.experiments.run_benchmark", return_value=[]) as run:
            with redirect_stdout(io.StringIO()):
                status = main(["-llm", "gpt-4.1", "-slm", "qwen3-4b", "-slm", "phi4"])
        self.assertEqual(status, 0)
        config = run.call_args.args[0]
        self.assertEqual(config.llm_model, "gpt-4.1")
        self.assertEqual(config.model, "qwen3-4b")
        self.assertEqual(run.call_args.kwargs["models"], ["qwen3-4b", "phi4"])
        self.assertEqual(Config.llm_model, "gpt-4.1")
        self.assertEqual(Config.model, "qwen3-4b")

    def test_clustering_only_experiment_runs_without_llm_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("Sample.csv").write_text(
                "Alice,A\nBob,B\n123,1\n", encoding="utf-8"
            )
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        "--experiment",
                        "cluster",
                        "--cluster-config",
                        "collapse",
                        "--data-dir",
                        str(root),
                        "--dataset",
                        "Sample.csv",
                        "--num-examples",
                        "1",
                        "--output-dir",
                        str(root / "results"),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertTrue((root / "results" / "cluster" / "summary.csv").is_file())
            self.assertIn("### DATASET START: Sample.csv ###", output.getvalue())
            self.assertIn("### DATASET END: Sample.csv ###", output.getvalue())

    def test_perfect_operator_implies_clusters_and_splits(self) -> None:
        args = parse_arguments(["-pp"])
        self.assertTrue(args.config.perfect_cluster)
        self.assertTrue(args.config.perfect_split)
        self.assertTrue(Config.perfect_cluster)
        self.assertTrue(Config.perfect_split)
        self.assertEqual(args.config.data_directory, Path("data_shuffled"))


if __name__ == "__main__":
    unittest.main()
