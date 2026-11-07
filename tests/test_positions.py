from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

from Propagate import witness
from Propagate.position import AbsPosPosition
from Propagate.positionop import RegExPos, get_str_from_pos
from Propagate.propagate import Propagate
from Propagate.tokenutil import is_valid_extraction
from Propagate.witness import WitnessPosition, get_pos_dict, validate_pos_dict
from Utils.config import Config


class OperatorStageTests(unittest.TestCase):
    def setUp(self) -> None:
        config = patch.multiple(
            Config, use_cache=False, save_cache=False, perfect_operator=False, verbose=False,
        )
        config.start()
        self.addCleanup(config.stop)

    def test_complete_output_clusters_skip_witnesses_and_keep_final_splits(self) -> None:
        # Output clusters can occur anywhere and need not form one combined cluster.
        clusters = [
            {10: "Smith Alice"},
            {0: "Alice Smith", 5: "Bob Jones"},
            {11: "White Carol", 12: " Brown Bob "},
        ]
        semantic = {
            0: {"first_name": "Alice", "last_name": "Smith"},
            10: {"fragment": "Smi"},  # Accepted model splits may cut within a token.
            11: {},  # An accepted empty split is also final.
            12: {"name": " Brown Bob "},
        }
        originals = {value_id: text for cluster in clusters for value_id, text in cluster.items()}
        with (
            patch.object(witness, "WitnessPosition", wraps=witness.WitnessPosition) as infer,
            patch.object(witness, "_score_position", wraps=witness._score_position) as score,
        ):
            operators = witness.get_cluster_operators(
                clusters, semantic, list(originals.values()), list(originals), "Names.csv",
                output_ids={10, 11, 12},
            )
        self.assertEqual({call.args[0] for call in infer.call_args_list}, {"Alice Smith"})
        self.assertTrue(score.called)
        self.assertTrue(all(call.args[2] == clusters[1] for call in score.call_args_list))
        self.assertEqual(operators[0], {})
        self.assertEqual(operators[2], {})
        self.assertTrue(operators[1])
        propagated = Propagate.propagate(clusters, operators, semantic)
        self.assertEqual(propagated[5], {"first_name": "Bob", "last_name": "Jones"})
        for value_id in (10, 11, 12):
            self.assertEqual(propagated[value_id], semantic[value_id])

    def test_source_mixed_and_incomplete_output_clusters_still_infer_operators(self) -> None:
        for name, cluster, semantic, output_ids in (
            ("source", {0: "Alice Smith", 5: "Bob Jones"},
             {0: {"name": "Alice Smith"}, 5: {"name": "Bob Jones"}}, {10, 11}),
            ("mixed", {0: "Alice Smith", 10: "Bob Jones"},
             {0: {"name": "Alice Smith"}, 10: {"name": "Bob Jones"}}, {10}),
            ("missing split", {10: "Alice Smith", 11: "Bob Jones"},
             {10: {"name": "Alice Smith"}}, {10, 11}),
        ):
            with (
                self.subTest(cluster=name),
                patch.object(witness, "WitnessPosition", wraps=witness.WitnessPosition) as infer,
                patch.object(witness, "_score_position", wraps=witness._score_position) as score,
            ):
                operators = witness.get_cluster_operators(
                    [cluster], semantic, list(cluster.values()), list(cluster), "Names.csv",
                    output_ids=output_ids,
                )
                self.assertEqual(
                    {call.args[0] for call in infer.call_args_list},
                    {cluster[value_id] for value_id in semantic},
                )
                self.assertTrue(score.called)
                self.assertTrue(operators[0])
                self.assertEqual(
                    Propagate.propagate([cluster], operators, semantic),
                    {value_id: {"name": original} for value_id, original in cluster.items()},
                )


class PositionTests(unittest.TestCase):
    def test_negative_absolute_and_regex_positions(self) -> None:
        pair = (AbsPosPosition(-5), AbsPosPosition(None))
        self.assertEqual(get_str_from_pos("Bob Jones", pair), "Jones")
        self.assertEqual(RegExPos("one two", r"\b[A-Za-z]+\b", -1, "L"), 4)
        self.assertIsNone(RegExPos("one", r"\d+", 0, "L"))

    def test_boundary_validation(self) -> None:
        self.assertTrue(is_valid_extraction("Alice Smith", "Alice", 0, 5))
        self.assertFalse(is_valid_extraction("Alice Smith", "Ali", 0, 3))
        self.assertFalse(is_valid_extraction("Alice Smith", "Alice", None, 5))

    def test_witnesses_propagate_to_structurally_similar_string(self) -> None:
        clusters = [{0: "Alice Smith", 1: "Bob Jones"}]
        semantic = {0: {"first_name": "Alice", "last_name": "Smith"}}
        positions = get_pos_dict(semantic, ["Alice Smith"], [0], "Names.csv")
        valid = validate_pos_dict(clusters, positions, semantic)
        result = Propagate.propagate(clusters, valid, semantic)
        self.assertEqual(result[1], {"first_name": "Bob", "last_name": "Jones"})

    def test_empty_component_has_no_witness(self) -> None:
        self.assertEqual(WitnessPosition("anything", ""), [])

    def test_trimming_only_applies_when_propagating_witness_values(self) -> None:
        original = "| Alice  Smith |"
        component = " Alice  Smith "
        pair = (AbsPosPosition(1), AbsPosPosition(-1))
        witnesses = WitnessPosition(original, component)
        self.assertIn(pair, witnesses)
        self.assertTrue(all(get_str_from_pos(original, item) == component
                            for item in witnesses))

        clusters = [{0: original, 1: "| Bob  Jones |"}]
        semantic = {0: {"name": component}}
        valid = validate_pos_dict(clusters, {0: {"name": [pair]}}, semantic)
        self.assertEqual(valid, {0: {"name": [pair]}})
        result = Propagate.propagate(clusters, valid, semantic)
        self.assertEqual(result, {0: {"name": component}, 1: {"name": "Bob  Jones"}})

    def test_selected_operators_are_pretty_json(self) -> None:
        clusters = [{0: "Alice Smith", 1: "Bob Jones"}]
        semantic = {0: {"first_name": "Alice", "last_name": "Smith"}}
        positions = get_pos_dict(semantic, ["Alice Smith"], [0], "Names.csv")
        previous_verbose = Config.verbose
        Config.verbose = True
        try:
            output = io.StringIO()
            with redirect_stdout(output):
                validate_pos_dict(clusters, positions, semantic)
        finally:
            Config.verbose = previous_verbose

        rendered = output.getvalue()
        self.assertIn("Selected operators for cluster 0:", rendered)
        operators = json.loads(rendered[rendered.index("{") :])
        self.assertIn("first_name", operators)
        operator_types = {
            boundary["type"]
            for pair in operators["first_name"]
            for boundary in pair
        }
        self.assertTrue(
            operator_types <= {"AbsPosPosition", "RegExPosPosition"}
        )
        self.assertTrue(operator_types)

    def test_propagation_pretty_json_includes_original_string(self) -> None:
        clusters = [{0: "Alice Smith", 1: "Bob Jones"}]
        semantic = {0: {"first_name": "Alice", "last_name": "Smith"}}
        positions = get_pos_dict(semantic, ["Alice Smith"], [0], "Names.csv")
        valid = validate_pos_dict(clusters, positions, semantic)
        previous_verbose = Config.verbose
        Config.verbose = True
        try:
            output = io.StringIO()
            with redirect_stdout(output):
                Propagate.propagate(clusters, valid, semantic)
        finally:
            Config.verbose = previous_verbose

        rendered = output.getvalue()
        self.assertIn("Propagation for cluster 0:", rendered)
        propagated = json.loads(rendered[rendered.index("[") :])
        self.assertEqual(propagated[1]["original string"], "Bob Jones")
        self.assertEqual(
            propagated[1]["propagation"],
            {"first_name": "Bob", "last_name": "Jones"},
        )


if __name__ == "__main__":
    unittest.main()
