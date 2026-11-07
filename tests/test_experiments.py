from __future__ import annotations

from contextlib import redirect_stdout
import io
import unittest

from gdt.output import DATASET_SEPARATOR, dataset_output, print_run_result
from gdt.results import EvaluationResult, RunResult


class ExperimentOutputTests(unittest.TestCase):
    def test_result_summary_includes_runtime_counts_accuracy_and_separator(self) -> None:
        result = RunResult(
            dataset="Names.csv",
            model="test-model",
            mode="gen",
            num_examples=2,
            duration_seconds=1.234,
            evaluation=EvaluationResult(accuracy=0.75, correct=3, incorrect=1),
        )
        output = io.StringIO()
        with redirect_stdout(output):
            print_run_result(result)

        self.assertEqual(
            output.getvalue(),
            "Names.csv | model=test-model | mode=gen | examples=2 | runtime: 1.23s | "
            "correct: 3 | error: 1 | accuracy: 0.7500\n\n",
        )

    def test_dataset_output_has_obvious_start_and_end_markers(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            with dataset_output("Names.csv"):
                print("dataset output")

        self.assertEqual(output.getvalue().count(DATASET_SEPARATOR), 4)
        self.assertIn("### DATASET START: Names.csv ###", output.getvalue())
        self.assertIn("### DATASET END: Names.csv ###", output.getvalue())
        self.assertLess(
            output.getvalue().index("DATASET START"),
            output.getvalue().index("dataset output"),
        )
        self.assertLess(
            output.getvalue().index("dataset output"),
            output.getvalue().index("DATASET END"),
        )


if __name__ == "__main__":
    unittest.main()
