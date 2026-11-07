from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from LLMUtil.LLMUtil import OpenAIUtil
from gdt.config import ExperimentConfig
from gdt.pipeline import PipelineRunner


class PipelineTests(unittest.TestCase):
    def test_exact_split_selection_uses_examples_and_unrepresented_clusters(self) -> None:
        split_clusters = [
            {0, 2},
            {3, 4},
            {1},
            {5, 6},
        ]
        mapped_clusters = [
            {0: "example zero", 2: "test two"},
            {3: "test three", 4: "test four"},
            {1: "example one"},
            {5: "output zero", 6: "output one"},
        ]

        self.assertEqual(
            PipelineRunner._select_semantic_split_ids(
                split_clusters,
                mapped_clusters,
                num_examples=2,
            ),
            [0, 3, 1, 5, 6],
        )

    def test_structural_pipeline_runs_end_to_end_with_a_stubbed_splitter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Names.csv"
            path.write_text(
                "Alice Smith,Smith Alice\n" + "UNUSED,UNUSED\n" * 4 + "Bob Brown,Brown Bob\n",
                encoding="utf-8",
            )
            config = ExperimentConfig(
                data_directory=path.parent,
                output_directory=path.parent / "results",
                num_examples=1,
                cluster_config="collapse",
            )

            semantic_components = {
                0: {"first_name": "Alice", "last_name": "Smith"},
                6: {"last_name": "Smith", "first_name": "Alice"},
            }
            output = io.StringIO()
            with (
                patch.object(OpenAIUtil, "get_split", return_value=semantic_components),
                redirect_stdout(output),
            ):
                result = PipelineRunner(config).run(path)

            self.assertTrue(result.succeeded, result.error)
            assert result.evaluation is not None
            self.assertEqual(result.evaluation.accuracy, 1.0)
            self.assertEqual(result.evaluation.correct, 1)
            self.assertEqual(
                result.evaluation.output_items,
                [
                    {
                        "original string": "Bob Brown",
                        "sem. components": [
                            {"first_name": "Bob"},
                            {"last_name": "Brown"},
                        ],
                        "transformation": "Brown Bob",
                        "ground truth": "Brown Bob",
                    }
                ],
            )
            rendered = output.getvalue()
            self.assertIn("Transformation function:", rendered)
            function_json = rendered[rendered.index("[") :]
            function_description, _ = json.JSONDecoder().raw_decode(function_json)
            self.assertEqual(
                function_description,
                [
                    {"semantic type": "last_name", "occurrence": 1},
                    {"constant string": " "},
                    {"semantic type": "first_name", "occurrence": 1},
                ],
            )

    def test_verbose_function_is_printed_once_before_transformation_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Names.csv"
            path.write_text(
                "Alice Smith,Smith Alice\n" + "UNUSED,UNUSED\n" * 4 + "Bob Brown,Brown Bob\n",
                encoding="utf-8",
            )
            config = ExperimentConfig(
                data_directory=path.parent,
                num_examples=1,
                cluster_config="collapse",
                verbose=True,
            )
            semantic_components = {
                0: {"first_name": "Alice", "last_name": "Smith"},
                6: {"last_name": "Smith", "first_name": "Alice"},
            }
            output = io.StringIO()
            with (
                patch.object(OpenAIUtil, "get_split", return_value=semantic_components),
                redirect_stdout(output),
            ):
                result = PipelineRunner(config).run(path)

            self.assertTrue(result.succeeded, result.error)
            rendered = output.getvalue()
            self.assertEqual(rendered.count("Transformation function:"), 1)
            self.assertLess(
                rendered.index("Transformation function:"),
                rendered.index("Transformation results:"),
            )


if __name__ == "__main__":
    unittest.main()
