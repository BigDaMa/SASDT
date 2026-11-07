"""Regressions for the fixed test set, complete ablations and reusable caches."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Clustering.collapsecluster import CollapseCluster
from DataAccess.dataaccess import DataAccess
from DataAccess.datafeature import DataFeature
from LLMUtil.ollama import OLLAMAUtil
from LLMUtil.providers import OpenAIUtil
from Utils.BenchmarkCache import BenchmarkCache, cache_gpt_splits
from gdt.cli import parse_arguments
from gdt.config import Config, ExperimentConfig
from gdt.data import PairDataset
from gdt.direct_benchmark import _run_one
from gdt.pipeline import PipelineRunner
from tests.test_split_budget import echoed, originals_in


class ExperimentContractTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(Config.apply, ExperimentConfig())
        self.addCleanup(DataFeature.clear)
        BenchmarkCache.clear()
        self.addCleanup(BenchmarkCache.clear)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "Names.csv"
        self.path.write_text("Alice Smith,Smith Alice\n" + "UNUSED,UNUSED\n" * 4
                             + "Bob Brown,Brown Bob\n", encoding="utf-8")

    def config(self, **kwargs):
        return ExperimentConfig(data_directory=self.root, output_directory=self.root / "results",
                                perfect_directory=self.root, num_examples=1, **kwargs)

    def test_count_above_five_is_rejected_by_both_cli_forms(self):
        for args in (["-n", "6"], ["--example-counts", "1,6"]):
            with self.subTest(args=args), redirect_stdout(io.StringIO()), \
                 patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(args)

    def test_direct_baseline_uses_same_test_rows_and_original_ids(self):
        for count in range(6):
            with self.subTest(count=count), \
                 patch("gdt.direct_benchmark._predict", return_value=["wrong"]):
                config = ExperimentConfig(data_directory=self.root, num_examples=count,
                                          output_directory=self.root / "results")
                result = _run_one(config, self.path, "openai")
                self.assertTrue(result.succeeded, result.error)
                self.assertEqual(result.evaluation.incorrect_items,
                                 [{"index": 5, "prediction": "wrong", "ground_truth": "Brown Bob"}])

    def test_perfect_splits_replace_all_composition_and_extraction_requests(self):
        sources = ["first_name: Alice\nlast_name: Smith"] + ["unused: UNUSED"] * 4 + [
            "first_name: Bob\nlast_name: Brown"]
        targets = ["last_name: Smith\nfirst_name: Alice"] + ["unused: UNUSED"] * 4 + [
            "last_name: Brown\nfirst_name: Bob"]
        dataset = PairDataset.from_csv(self.path)
        originals = [row.source for row in dataset.records()] + [
            row.target for row in dataset.records()]
        splits = [original + "\n" + split for original, split in zip(originals, sources + targets)]
        self.root.joinpath("Names_split.txt").write_text("\n[ITEM]\n".join(splits), encoding="utf-8")
        for model in ("gpt-5.1", "phi4"):
            with self.subTest(model=model), redirect_stdout(io.StringIO()), \
                 patch.object(OpenAIUtil, "create_response") as llm, \
                 patch.object(OLLAMAUtil, "_split_response") as slm:
                result = PipelineRunner(self.config(model=model, perfect_split=True)).run(self.path)
                self.assertTrue(result.succeeded, result.error)
                self.assertEqual(result.evaluation.accuracy, 1)
                self.assertEqual(result.evaluation.correct, 1)
                self.assertFalse(result.has_lm_failures)
                llm.assert_not_called()
                slm.assert_not_called()
                self.assertEqual(set(DataFeature.data_features), {0, 5, 6})

    def test_cached_clusters_restore_compositions_for_extraction(self):
        config = self.config(save_cache=True, cache_directory=self.root / "cache")
        Config.apply(config)

        def labels(query, **kwargs):
            return "\n[ITEM]\n".join(echoed(value, "[last_name][first_name]"
                if value == "Smith Alice" else "[first_name][last_name]")
                for value in originals_in(query))

        with patch.object(DataAccess, "DATA_FOLDER", str(self.root)), \
             patch.object(DataAccess, "NUM_EXAMPLES", 1), \
             patch.object(OpenAIUtil, "create_response", side_effect=labels) as send:
            clusters = CollapseCluster.run_cluster_collapse_promptsplit_gpt(self.path.name)
            expected = dict(DataFeature.data_features)
            calls = send.call_count
            self.assertEqual(calls, 2)
            DataFeature.clear()
            Config.use_cache = True
            self.assertEqual(CollapseCluster.run_cluster_collapse_promptsplit_gpt(self.path.name), clusters)
            self.assertEqual(send.call_count, calls)
            self.assertEqual(DataFeature.data_features, expected)
            self.assertEqual(DataFeature.get_all_sem_types(self.path.name), ["first_name", "last_name"])

    def test_money_saver_does_not_reuse_larger_example_context(self):
        calls = []

        @cache_gpt_splits("labels")
        def discover(query_dict, file):
            calls.append(Config.num_examples)
            return {i: ["[name]"] for i in query_dict}

        Config.apply(self.config(money_saver=True))
        for count in (5, 1, 1, 5):
            Config.num_examples = count
            discover({0: "Alice"}, "Names.csv")
        self.assertEqual(calls, [5, 1])
