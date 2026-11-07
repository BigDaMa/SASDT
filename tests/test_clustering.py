from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Clustering.collapsecluster import CollapseCluster
from Clustering.regexcluster import RegexProcessor
from DataAccess.dataaccess import DataAccess
from DataAccess.datafeature import DataFeature
from LLMUtil.semantic import OpenAIUtil_Prompt


class StructuralClusteringTests(unittest.TestCase):
    def test_prompt_stage_gets_type_sequences_for_example_inputs_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "Names.csv")
            path.write_text(
                "Alice Smith,Smith Alice\n"
                "Bob Jones,Jones Bob\n"
                "Carol White,White Carol\n",
                encoding="utf-8",
            )
            previous_directory = DataAccess.DATA_FOLDER
            previous_examples = DataAccess.NUM_EXAMPLES
            previous_features = DataFeature.data_features.copy()
            DataAccess.DATA_FOLDER = directory
            DataAccess.NUM_EXAMPLES = 2
            split = {
                0: ["[first_name]", "[last_name]"],
                1: ["[first_name]", "[last_name]"],
                3: ["[last_name]", "[first_name]"],
                4: ["[last_name]", "[first_name]"],
            }
            try:
                with patch.object(
                    OpenAIUtil_Prompt,
                    "get_split_prompt",
                    return_value=split,
                ) as split_prompt, patch.object(
                    OpenAIUtil_Prompt, "get_labels_for_types",
                    return_value={2: ["[first_name]", "[last_name]"]},
                ):
                    CollapseCluster._run_prompt_split(path.name)
            finally:
                DataAccess.DATA_FOLDER = previous_directory
                DataAccess.NUM_EXAMPLES = previous_examples
                DataFeature.data_features = previous_features

        queried_strings = split_prompt.call_args.args[0]
        self.assertEqual(
            queried_strings,
            {
                0: "Alice Smith",
                1: "Bob Jones",
                3: "Smith Alice",
                4: "Jones Bob",
            },
        )

    def test_output_examples_are_kept_in_the_final_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "Sample.csv")
            path.write_text("Alice,A\n" + "UNUSED,UNUSED\n" * 4 + "Bob,B\n123,1\n",
                            encoding="utf-8")
            previous_directory = DataAccess.DATA_FOLDER
            previous_examples = DataAccess.NUM_EXAMPLES
            try:
                DataAccess.DATA_FOLDER = directory
                DataAccess.NUM_EXAMPLES = 1
                clusters = CollapseCluster.run_cluster_collapse(path.name)
            finally:
                DataAccess.DATA_FOLDER = previous_directory
                DataAccess.NUM_EXAMPLES = previous_examples
        self.assertEqual(clusters[-1], {7})
        self.assertIn({0, 5}, clusters)
        self.assertIn({6}, clusters)
        self.assertTrue(set(range(1, 5)).isdisjoint(set().union(*clusters)))

    def test_regex_cleanup_and_nested_alternation_expansion(self) -> None:
        self.assertEqual(str(RegexProcessor(r"\d\d---").regex_clean()), r"\d{2}-{3}")
        self.assertEqual(
            RegexProcessor(r"prefix(a|b)(1|2)").expand(),
            ["prefixa1", "prefixa2", "prefixb1", "prefixb2"],
        )


if __name__ == "__main__":
    unittest.main()
