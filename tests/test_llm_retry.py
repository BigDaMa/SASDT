from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from LLMUtil.providers import request_split_with_retry
from gdt.config import ExperimentConfig
from gdt.diagnostics import dataset_diagnostics


class SplitRetryTests(unittest.TestCase):
    def test_split_count_mismatch_is_retried(self) -> None:
        responses = iter(
            [
                "first_name: Alice",
                "first_name: Alice\n[ITEM]\nlast_name: Jones",
            ]
        )
        calls = 0

        def request() -> str:
            nonlocal calls
            calls += 1
            return next(responses)

        response, items = request_split_with_retry(
            request,
            prompt="split Alice Jones",
            expected_count=2,
            operation="test semantic split",
            num_retry=1,
        )

        self.assertEqual(calls, 2)
        self.assertIn("[ITEM]", response)
        self.assertEqual(items, ["first_name: Alice", "last_name: Jones"])

    def test_exhausted_split_retries_log_rejected_responses_and_return_empty_results(self) -> None:
        calls = 0

        def request() -> str:
            nonlocal calls
            calls += 1
            return "first_name: Alice"

        with tempfile.TemporaryDirectory() as directory:
            with dataset_diagnostics(directory, "test.csv", ExperimentConfig()):
                _, items = request_split_with_retry(
                    request,
                    prompt="FULL TEST PROMPT",
                    expected_count=2,
                    operation="test semantic split",
                    num_retry=2,
                )
            rendered = (Path(directory) / "errors.txt").read_text(encoding="utf-8")
        self.assertEqual(calls, 3)
        self.assertEqual(items, ["", ""])
        self.assertEqual(rendered.count("FULL TEST PROMPT"), 3)
        self.assertEqual(rendered.count("first_name: Alice"), 3)
        self.assertIn("TOLERATED: EMPTY RESULT for 2", rendered)


if __name__ == "__main__":
    unittest.main()
