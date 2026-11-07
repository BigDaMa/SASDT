"""String-based ablations, shuffled IDs, and recoverable truth diagnostics."""

from contextlib import ExitStack, redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from DataAccess.datafeature import DataFeature
from LLMUtil.ollama import OLLAMAUtil
from LLMUtil.providers import OpenAIUtil
from Propagate.position import AbsPosPosition, RegExPosPosition
from Propagate.positionop import get_str_from_pos
from Propagate.tokenutil import is_valid_extraction
from Propagate.witness import WitnessPosition
from gdt.config import Config, ExperimentConfig
from gdt.cluster_experiment import run_cluster_experiment
from gdt.data import PairDataset
from gdt.diagnostics import dataset_diagnostics
from gdt.ground_truth import (
    GroundTruth, ground_truth_session, parse_position, perfect_path,
    provide_clusters, provide_operators, provide_splits,
)
from gdt.pipeline import PipelineRunner


class GroundTruthTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(Config.apply, ExperimentConfig())
        self.addCleanup(DataFeature.clear)
        self.rows = [
            ("Alice Smith", "Smith Alice"), ("Carol White", "White Carol"),
            ("David Black", "Black David"), ("Ellen Green", "Green Ellen"),
            ("Frank Grey", "Grey Frank"), ("Bob Brown", "Brown Bob"),
            ("Grace Blue", "Blue Grace"), ("Helen Gold", "Gold Helen"),
        ]
        self.path = self.root / "Names_0.csv"
        self.write_csv(self.path, self.rows)
        self.write_truth()

    def write_csv(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)

    def write(self, suffix, content):
        (self.root / ("Names" + suffix)).write_text(content, encoding="utf-8")

    def write_truth(self):
        self.write("_cluster.txt", "\n".join(row[0] for row in self.rows) + "\n[ITEM]\n"
                   + "\n".join(row[1] for row in self.rows))
        self.write("_split.txt", "\n[ITEM]\n".join(
            f"{original}\nfirst_name:{original.split()[0]}\nlast_name:{original.split()[1]}"
            for original, _target in self.rows) + "\n[ITEM]\n" + "\n[ITEM]\n".join(
            f"{target}\nlast_name:{target.split()[0]}\nfirst_name:{target.split()[1]}"
            for _original, target in self.rows) + "\n[ITEM]\n")
        # Carol is an inactive reserved example when n=1. Her rules still apply
        # to the cluster containing Alice and the held-out source strings.
        self.ops = (
            "first_name:AbsPos(0)[PARAMSPLIT]RegPos([ESCAPESPACE],0,L)\n"
            "last_name:RegPos([ESCAPESPACE],0,R)[PARAMSPLIT]AbsPos(None)"
        )
        self.write("_op.txt", "Carol White\n" + self.ops)
        self.write("_types.txt", "first_name\nlast_name\n")

    def config(self, **kwargs):
        return ExperimentConfig(data_directory=self.root, perfect_directory=self.root,
                                output_directory=self.root / "results", num_examples=1, **kwargs)

    def load(self):
        return GroundTruth(PairDataset.from_csv(self.path), self.root, 1)

    def test_same_truth_files_follow_three_different_row_orders(self):
        for number, rows in enumerate((self.rows, self.rows[::-1], self.rows[3:] + self.rows[:3])):
            path = self.root / f"Names_{number}.csv"
            self.write_csv(path, rows)
            truth = GroundTruth(PairDataset.from_csv(path), self.root, 1)
            self.assertEqual(truth.clusters, [{0, 5, 6, 7}, {8}])
            self.assertEqual(truth.split_for({42: rows[0][0]})[42],
                             {"first_name": rows[0][0].split()[0],
                              "last_name": rows[0][0].split()[1]})
            self.assertEqual(set(truth.operators), {0})
            self.assertEqual(perfect_path(path, "_types.txt", self.root).name, "Names_types.txt")

    def test_absent_representative_resolves_after_inactive_clusters_are_removed(self):
        representative = "External Representative"
        self.write("_cluster.txt", "Unused\n[ITEM]\n"
                   + "\n".join(target for _original, target in self.rows)
                   + "\n[ITEM]\n" + representative + "\n"
                   + "\n".join(original for original, _target in self.rows)
                   + "\n[ITEM]\nUnused")
        self.write("_op.txt", "Unused\nmalformed unused operator\n[ITEM]\n"
                   + representative + "\n" + self.ops)
        with (self.root / "Names_split.txt").open("a", encoding="utf-8") as handle:
            handle.write(representative + "\nfirst_name:External\nlast_name:Representative")
        config = self.config(perfect_operator=True)
        with dataset_diagnostics(config.output_directory, self.path.name, config) as diagnostics:
            truth = self.load()
            self.assertEqual(truth.clusters, [{8}, {0, 5, 6, 7}])
            self.assertEqual(set(truth.operators), {1})
            self.assertEqual(get_str_from_pos("Helen Gold", truth.operators[1]["last_name"][0]),
                             "Gold")
            self.assertNotIn(representative, truth.known)
            self.assertNotIn(representative, truth.active_strings)
            self.assertNotIn(representative, truth.splits)
            self.assertNotIn("Unused", truth.cluster_by_string)
            self.assertEqual(set(truth.splits), truth.active_strings)
            self.assertEqual(diagnostics.ground_truth_errors, [])

    def test_disjoint_datasets_share_operator_representatives(self):
        other_rows = [
            ("Iris Rose", "Rose Iris"), ("Jack Fox", "Fox Jack"),
            ("Kate Wolf", "Wolf Kate"), ("Liam Wood", "Wood Liam"),
            ("Mia Lake", "Lake Mia"), ("Noah Hill", "Hill Noah"),
            ("Olivia Reed", "Reed Olivia"), ("Paul Stone", "Stone Paul"),
        ]
        combined = self.rows + other_rows
        self.write("_cluster.txt", "\n".join(original for original, _target in combined)
                   + "\n[ITEM]\n" + "\n".join(target for _original, target in combined))
        self.write("_split.txt", "\n[ITEM]\n".join(
            f"{original}\nfirst_name:{original.split()[0]}\nlast_name:{original.split()[1]}"
            for original, _target in combined) + "\n[ITEM]\n" + "\n[ITEM]\n".join(
            f"{target}\nlast_name:{target.split()[0]}\nfirst_name:{target.split()[1]}"
            for _original, target in combined))
        # The existing operator representative, Carol White, appears only in variant 0.
        for number, rows in enumerate((self.rows, other_rows)):
            path = self.root / f"Names_{number}.csv"
            self.write_csv(path, rows)
            with self.subTest(variant=number), redirect_stdout(io.StringIO()), \
                 patch.object(OpenAIUtil, "create_response") as send:
                result = PipelineRunner(self.config(perfect_operator=True)).run(path)
            self.assertTrue(result.succeeded, result.error)
            self.assertEqual(result.evaluation.correct, 3)
            self.assertEqual(result.evaluation.accuracy, 1.0)
            send.assert_not_called()
        self.assertFalse((self.root / "results" / "errors.txt").exists())

    def test_absent_representatives_keep_first_conflicting_operator_set(self):
        representatives = ["First Representative", "Equal Representative", "Conflicting Representative"]
        cluster_path = self.root / "Names_cluster.txt"
        cluster_path.write_text("\n".join(representatives) + "\n"
                                + cluster_path.read_text(encoding="utf-8"), encoding="utf-8")
        equivalent = self.ops.replace("AbsPos(0)", "AbsPosPosition(0)")
        conflict = self.ops.replace("AbsPos(0)", "AbsPos(1)")
        self.write("_op.txt", "\n[ITEM]\n".join(
            representative + "\n" + operations
            for representative, operations in zip(representatives, (self.ops, equivalent, conflict))))
        config = self.config(perfect_operator=True)
        with dataset_diagnostics(config.output_directory, self.path.name, config) as diagnostics:
            truth = self.load()
            self.assertEqual(get_str_from_pos("Bob Brown", truth.operators[0]["first_name"][0]),
                             "Bob")
            self.assertEqual(len(diagnostics.ground_truth_errors), 1)
            self.assertIn("keeping first set from 'First Representative'",
                          diagnostics.ground_truth_errors[0][1])

    def test_missing_strings_are_logged_and_outside_dataset_strings_are_silent(self):
        self.write("_cluster.txt", "Alice Smith\nBob Brown\nGhost\n[ITEM]\nSmith Alice")
        self.write("_split.txt", "Alice Smith\nfirst_name:Alice\n[ITEM]\nGhost\nname:Ghost")
        config = self.config(perfect_cluster=True, perfect_split=True)
        with dataset_diagnostics(config.output_directory, self.path.name, config) as diagnostics:
            truth = self.load()
            self.assertEqual(truth.clusters, [{0, 5}, {8}, {6}, {7}])
            self.assertEqual(truth.split_for({8: "Smith Alice", 5: "Bob Brown"}), {8: {}, 5: {}})
            self.assertEqual(truth.split_for({5: "Bob Brown"}, prompt_only=True), {5: []})
            self.assertFalse(diagnostics.result_fields()["has_lm_failures"])
        log = (config.output_directory / "errors.txt").read_text(encoding="utf-8")
        self.assertNotIn("unknown string", log)
        self.assertNotIn("Ghost", log)
        self.assertIn("missing string; using a separate cluster", log)
        self.assertIn("missing string; using an empty split", log)
        self.assertNotIn("White Carol", log)  # An inactive target is not missing evidence.
        self.assertNotIn("QUERY", log)

    def test_operations_use_cluster_representatives_and_first_conflicting_set(self):
        equivalent = self.ops.replace("AbsPos(0)", "AbsPosPosition(0)")
        conflict = self.ops.replace("AbsPos(0)", "AbsPos(1)")
        self.write("_op.txt", "Carol White\n" + self.ops + "\n[ITEM]\nBob Brown\n"
                   + equivalent + "\n[ITEM]\nGrace Blue\n" + conflict
                   + "\n[ITEM]\nGhost\n" + self.ops)
        config = self.config(perfect_operator=True)
        with dataset_diagnostics(config.output_directory, self.path.name, config) as diagnostics:
            truth = self.load()
            self.assertEqual(get_str_from_pos("Helen Gold", truth.operators[0]["first_name"][0]),
                             "Helen")
            self.assertEqual(get_str_from_pos("Helen Gold", truth.operators[0]["last_name"][0]),
                             "Gold")
            self.assertEqual(len(diagnostics.ground_truth_errors), 1)
        log = (config.output_directory / "errors.txt").read_text(encoding="utf-8")
        self.assertIn("conflicting operations", log)
        self.assertIn("keeping first set from 'Carol White'", log)
        self.assertNotIn("unknown string", log)
        self.assertNotIn("Ghost", log)

    def test_malformed_operations_are_recoverable(self):
        self.write("_op.txt", "Alice Smith\nname:AbsPos(0)\n[ITEM]\nCarol White\n" + self.ops)
        config = self.config(perfect_operator=True)
        with dataset_diagnostics(config.output_directory, self.path.name, config) as diagnostics:
            truth = self.load()
            self.assertEqual(set(truth.operators), {0})
            self.assertEqual(len(diagnostics.ground_truth_errors), 1)
            self.assertIn("[PARAMSPLIT]", diagnostics.ground_truth_errors[0][1])

    def test_literal_matching_keeps_duplicate_ids_spaces_commas_and_colons(self):
        original = '  Straße, https://a:b/"c"  '
        rows = [(original, "out")] * 6
        self.write_csv(self.path, rows)
        self.write("_cluster.txt", original + "\n[ITEM]\nout")
        self.write("_split.txt", original + "\nurl:https://a:b/\n[ITEM]\nout\nname:out")
        truth = self.load()
        self.assertEqual(truth.clusters, [{0, 5}, {6}])
        self.assertEqual(truth.split_for({0: original, 5: original}),
                         {0: {"url": "https://a:b/"}, 5: {"url": "https://a:b/"}})

    def test_position_objects_keep_existing_boundary_semantics(self):
        self.assertEqual(parse_position("AbsPos(-1)"), AbsPosPosition(-1))
        self.assertEqual(parse_position("AbsPos(None)"), AbsPosPosition(None))
        self.assertEqual(parse_position("AbsPosPosition(None)"), AbsPosPosition(None))
        self.assertEqual(parse_position("RegPos([ESCAPENUMBER],-2,R)"),
                         RegExPosPosition(r"\d+", -2, "R"))
        self.assertEqual(parse_position("RegPos([ESCAPENUMBER],-2,R)").get_index("A12-B34-C56"), 7)
        self.assertEqual(parse_position("RegPos([ESCAPECOMMA],0,L)").get_index("a,b"), 1)
        self.assertEqual(parse_position("RegPos([ESCAPEIN][ESCAPELEFTPAREN],0,L)").regex, r"In\s\(")

    def test_serialized_exclusive_endpoints_match_witnesses_and_application(self):
        cases = (
            # The hour bounds are correct, but the separate token-cut rule
            # rejects propagation between the alphanumeric characters T and 1.
            ("04/03/2023 T13:25", "13", "AbsPos(-5)", "AbsPos(-3)", False),
            ("04/03/2023 T13:25", "25", "AbsPos(-2)", "AbsPos(None)", True),
            ("+86 138 199 2992", "138", "AbsPos(-12)", "AbsPos(-9)", True),
            ("082-666-1954", "082", "AbsPos(0)", "AbsPos(3)", True),
            ("ACM SIGMOD Record.", "ACM SIGMOD Record", "AbsPos(0)",
             "RegPos([ESCAPEPERIOD],-1,L)", True),
        )
        for original, expected, left, right, valid in cases:
            with self.subTest(original=original, component=expected):
                pair = (parse_position(left), parse_position(right))
                self.assertIn(pair, WitnessPosition(original, expected))
                self.assertEqual(get_str_from_pos(original, pair), expected)
                self.assertEqual(is_valid_extraction(
                    original, expected, pair[0].get_index(original), pair[1].get_index(original),
                ), valid)
        truncated = (parse_position("AbsPos(-2)"), parse_position("AbsPos(-1)"))
        self.assertEqual(get_str_from_pos("13:25", truncated), "2")
        self.assertFalse(is_valid_extraction("13:25", "2", -2, -1))

    def test_unordered_truth_splits_produce_text_ordered_compositions_and_transformations(self):
        path = self.root / "Names_split.txt"
        blocks = path.read_text(encoding="utf-8").split("[ITEM]")
        path.write_text("[ITEM]".join(
            "\n".join([lines[0], *reversed(lines[1:])]) + "\n"
            for block in blocks if (lines := block.strip().splitlines())
        ), encoding="utf-8")
        truth = self.load()
        self.assertEqual(truth.split_for({0: "Alice Smith", 8: "Smith Alice"}, prompt_only=True),
                         {0: ["[first_name]", "[last_name]"],
                          8: ["[last_name]", "[first_name]"]})
        self.assertEqual(truth.split_for({8: "Smith Alice"}),
                         {8: {"first_name": "Alice", "last_name": "Smith"}})
        for flags in ({"perfect_split": True}, {"perfect_cluster": True, "perfect_split": True},
                      {"perfect_operator": True}):
            with self.subTest(flags=flags), redirect_stdout(io.StringIO()), \
                 patch.object(OpenAIUtil, "create_response") as send:
                result = PipelineRunner(self.config(**flags)).run(self.path)
                self.assertTrue(result.succeeded, result.error)
                self.assertEqual(result.evaluation.accuracy, 1.0)
                send.assert_not_called()

    def test_all_four_scenarios_for_gpt_and_slm(self):
        scenarios = ({"perfect_cluster": True}, {"perfect_split": True},
                     {"perfect_cluster": True, "perfect_split": True},
                     {"perfect_operator": True})
        for model, slm_split in (("gpt-5.1", False), ("phi4", False),
                                 ("gpt-5.1", True), ("phi4", True)):
            for flags in scenarios:
                with self.subTest(model=model, slm_split=slm_split, flags=flags), \
                     redirect_stdout(io.StringIO()):
                    config = self.config(model=model, slm_split=slm_split, **flags)
                    truth = self.load()
                    extractor = OLLAMAUtil if slm_split and model == "phi4" else OpenAIUtil

                    def extract(query, *args, **kwargs):
                        self.assertTrue(kwargs["use_type"])
                        self.assertEqual(kwargs["type_list"], ["first_name", "last_name"])
                        return truth.split_for(query)

                    with patch.object(OpenAIUtil, "create_response") as llm, \
                         patch.object(OLLAMAUtil, "_split_response") as slm:
                        if not config.perfect_split:
                            with patch.object(extractor, "get_split", side_effect=extract) as split:
                                result = PipelineRunner(config).run(self.path)
                                split.assert_called_once()
                        else:
                            result = PipelineRunner(config).run(self.path)
                        llm.assert_not_called()
                        slm.assert_not_called()
                    self.assertTrue(result.succeeded, result.error)
                    self.assertEqual(result.evaluation.accuracy, 1.0)
                    self.assertEqual(result.evaluation.correct, 3)
                    self.assertFalse(result.has_lm_failures)

    def test_pipeline_reports_truth_errors_and_evaluates_every_held_out_row(self):
        self.write("_cluster.txt", "Alice Smith\nBob Brown\nGrace Blue\nUnknown")
        self.write("_split.txt", "Alice Smith\nfirst_name:Alice\nlast_name:Smith")
        config = self.config(perfect_operator=True)
        with redirect_stdout(io.StringIO()), patch.object(OpenAIUtil, "create_response") as send:
            result = PipelineRunner(config).run(self.path)
        send.assert_not_called()
        self.assertTrue(result.succeeded, result.error)
        self.assertEqual(result.evaluation.correct + result.evaluation.incorrect, 3)
        self.assertFalse(result.has_lm_failures)
        self.assertIn("Smith Alice", (config.output_directory / "errors.txt").read_text())

    def test_supplied_operators_replace_the_operator_stage_without_inference_or_validation(self):
        from Propagate import witness
        from Propagate.propagate import Propagate

        with redirect_stdout(io.StringIO()), \
             patch.object(witness, "get_cluster_operators", wraps=witness.get_cluster_operators) as stage, \
             patch.object(witness, "get_pos_dict") as infer, \
             patch.object(witness, "validate_pos_dict") as validate, \
             patch.object(Propagate, "propagate", wraps=Propagate.propagate) as propagate:
            result = PipelineRunner(self.config(perfect_operator=True)).run(self.path)
        self.assertTrue(result.succeeded, result.error)
        self.assertEqual(result.evaluation.accuracy, 1.0)
        stage.assert_called_once()
        infer.assert_not_called()
        validate.assert_not_called()
        # The provided result has the same cluster-indexed shape as inferred operators.
        operators = propagate.call_args.args[1]
        self.assertEqual(set(operators), {0, 1})
        self.assertEqual(set(operators[0]), {"first_name", "last_name"})
        self.assertEqual(operators[1], {})

    def test_cluster_experiment_evaluates_against_shared_string_truth(self):
        with redirect_stdout(io.StringIO()), patch.object(OpenAIUtil, "create_response") as send:
            results = run_cluster_experiment(self.config(perfect_cluster=True),
                                             datasets=[self.path.name])
        send.assert_not_called()
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].succeeded, results[0].error)
        self.assertEqual(results[0].evaluation.accuracy, 1.0)

    def test_supplied_festival_type_filename_is_supported_without_renaming(self):
        (self.root / "Festival_types.txt").write_text("festival\nyear\n", encoding="utf-8")
        Config.apply(self.config(perfect_cluster=True))
        self.assertEqual(DataFeature.get_all_sem_types("Festivals_2.csv"), ["festival", "year"])
        self.assertFalse((self.root / "Festivals_types.txt").exists())

    def test_ground_truth_errors_flush_on_interruption(self):
        self.write("_cluster.txt", "Ghost")
        config = self.config(perfect_cluster=True)
        with self.assertRaises(KeyboardInterrupt):
            with dataset_diagnostics(config.output_directory, self.path.name, config):
                self.load().clusters
                raise KeyboardInterrupt()
        log = (config.output_directory / "errors.txt").read_text(encoding="utf-8")
        self.assertIn("GROUND TRUTH ERROR: missing string; using a separate cluster", log)
        self.assertNotIn("Ghost", log)
        self.assertIn("DATASET: Names_0.csv", log)

    def test_three_providers_load_only_the_corresponding_truth_source(self):
        Config.apply(self.config(perfect_operator=True))
        config = self.config(perfect_operator=True)
        # Invalid later-stage files must not be parsed by an earlier provider.
        self.write("_split.txt", "Ghost\nname:Ghost")
        self.write("_op.txt", "Alice Smith\nname:AbsPos(0)")
        with dataset_diagnostics(config.output_directory, self.path.name, config) as diagnostics, \
             ground_truth_session():
            self.assertEqual(provide_clusters(self.path.name), [{0, 5, 6, 7}, {8}])
            self.assertEqual(diagnostics.ground_truth_errors, [])
            self.assertEqual(provide_splits(self.path.name, {5: "Bob Brown"}), {5: {}})
            self.assertTrue(diagnostics.ground_truth_errors)
            self.assertTrue(all(path.endswith("_split.txt")
                                for path, _reason, _original in diagnostics.ground_truth_errors))
            diagnostics.ground_truth_errors.clear()
            self.assertEqual(provide_operators(self.path.name), {0: {}, 1: {}})
            self.assertEqual(len(diagnostics.ground_truth_errors), 1)
            self.assertTrue(diagnostics.ground_truth_errors[0][0].endswith("_op.txt"))

    def test_operator_provider_returns_independent_cluster_results(self):
        Config.apply(self.config(perfect_operator=True))
        with ground_truth_session():
            operators = provide_operators(self.path.name)
            self.assertEqual(list(operators), [0, 1])
            self.assertEqual(get_str_from_pos("Bob Brown", operators[0]["first_name"][0]), "Bob")
            # Returned results can be consumed without changing parser state.
            operators[0]["first_name"].clear()
            self.assertTrue(provide_operators(self.path.name)[0]["first_name"])

    def test_supplied_operators_are_not_validated_against_split_evidence(self):
        from Propagate import witness

        # This operator disagrees with the split; the truth provider still delivers it.
        pair = (AbsPosPosition(0), AbsPosPosition(1))
        self.write("_op.txt", "Carol White\nfirst_name:AbsPos(0)[PARAMSPLIT]AbsPos(1)")
        Config.apply(self.config(perfect_operator=True))
        truth = self.load()
        with ground_truth_session(), \
             patch.object(witness, "get_pos_dict") as infer, \
             patch.object(witness, "validate_pos_dict") as validate:
            result = witness.get_cluster_operators(
                [{i: truth.active[i] for i in cluster} for cluster in truth.clusters],
                {0: {"first_name": "Alice"}}, ["Alice Smith"], [0], self.path.name,
            )
        self.assertEqual(result, {0: {"first_name": [pair]}, 1: {}})
        infer.assert_not_called()
        validate.assert_not_called()

    def test_all_flag_combinations_use_the_same_pipeline_step_order(self):
        from Clustering.clustermeta import ClusterMeta
        from Propagate import witness
        from Propagate.propagate import Propagate
        from transform.Evaluation import Evaluation
        from transform.Transform import Transform

        expected = ["cluster", "select", "split", "operators",
                    "propagate", "transform", "evaluate"]
        for flags in ({}, {"perfect_cluster": True}, {"perfect_split": True},
                      {"perfect_cluster": True, "perfect_split": True},
                      {"perfect_operator": True}):
            with self.subTest(flags=flags), ExitStack() as stack:
                stack.enter_context(redirect_stdout(io.StringIO()))
                config = self.config(cluster_config="collapse", **flags)
                events = []

                def traced(name, function):
                    def call(*args, **kwargs):
                        events.append(name)
                        return function(*args, **kwargs)
                    return call

                for owner, method, event in (
                    (ClusterMeta, "run_cluster", "cluster"),
                    (PipelineRunner, "_select_semantic_split_ids", "select"),
                    (witness, "get_cluster_operators", "operators"),
                    (Propagate, "propagate", "propagate"),
                    (Transform, "transform_with_function", "transform"),
                    (Evaluation, "evaluate", "evaluate"),
                ):
                    stack.enter_context(patch.object(owner, method,
                                                     side_effect=traced(event, getattr(owner, method))))
                split = OpenAIUtil.get_split if config.perfect_split else (
                    lambda query, *args, **kwargs: self.load().split_for(query))
                stack.enter_context(patch.object(OpenAIUtil, "get_split",
                                                side_effect=traced("split", split)))
                send = stack.enter_context(patch.object(OpenAIUtil, "create_response"))
                infer = stack.enter_context(patch.object(witness, "get_pos_dict",
                                                         wraps=witness.get_pos_dict))
                validate = stack.enter_context(patch.object(witness, "validate_pos_dict",
                                                            wraps=witness.validate_pos_dict))
                candidates = stack.enter_context(patch.object(witness, "WitnessPosition",
                                                              wraps=witness.WitnessPosition))
                scores = stack.enter_context(patch.object(witness, "_score_position",
                                                          wraps=witness._score_position))
                result = PipelineRunner(config).run(self.path)
                self.assertTrue(result.succeeded, result.error)
                self.assertEqual(result.evaluation.accuracy, 1.0)
                self.assertEqual(events, expected)
                self.assertEqual(infer.call_count, 0 if config.perfect_operator else 1)
                self.assertEqual(validate.call_count, 0 if config.perfect_operator else 1)
                # Whether splits come from the model or truth, only sources need witnesses.
                if not config.perfect_operator:
                    self.assertTrue(candidates.called)
                    self.assertTrue(scores.called)
                sources = {original for original, _target in self.rows}
                self.assertTrue(all(call.args[0] in sources
                                    for call in candidates.call_args_list))
                self.assertTrue(all(set(call.args[2].values()) <= sources
                                    for call in scores.call_args_list))
                send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
