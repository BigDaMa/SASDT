from __future__ import annotations

import unittest
from unittest.mock import patch

from LLMUtil.providers import get_value_split
from LLMUtil.providers import OpenAIUtil
from LLMUtil.semantic import OpenAIUtil_Prompt
from Utils.BenchmarkCache import BenchmarkCache
from Utils.CacheUtil import _cache_filename
from gdt.config import Config, ExperimentConfig


class _ExactExtractionProvider:
    __MODEL__ = "gpt-test"
    calls = 0

    @classmethod
    def get_semantic_components(cls, *args, **kwargs) -> str:
        cls.calls += 1
        return (
            "street_name: Sample St.\n"
            "house_number: 12\n"
            "[ITEM]\n"
            "street_name: Broadway"
        )

    @staticmethod
    def get_further_split(*args, **kwargs) -> str:
        raise AssertionError("exact extraction must not refine semantic types")

    @staticmethod
    def get_final_split(*args, **kwargs) -> str:
        raise AssertionError("exact extraction must not rediscover semantic types")


class LLMWorkflowTests(unittest.TestCase):
    def tearDown(self) -> None:
        Config.apply(ExperimentConfig())
        BenchmarkCache.clear()

    def test_openai_extraction_uses_selected_llm_independently_of_slm(self) -> None:
        for slm_split, expected in ((False, "gpt-4.1"), (True, "gpt-5.1")):
            with self.subTest(slm_split=slm_split):
                Config.apply(ExperimentConfig(
                    model="gpt-5.1", llm_model="gpt-4.1", slm_split=slm_split,
                ))
                with patch.object(OpenAIUtil, "create_response", return_value="name: Alice") as send:
                    OpenAIUtil.get_semantic_components("Alice")
                    OpenAIUtil.get_further_split("name: Alice", "name")
                    OpenAIUtil.get_final_split("Alice", ["name"])
                self.assertEqual([call.kwargs["model"] for call in send.call_args_list],
                                 [expected] * 3)

    def test_value_cache_uses_extraction_model_when_a_local_model_is_selected(self) -> None:
        Config.apply(ExperimentConfig(model="phi4", money_saver=True))
        with patch.object(OpenAIUtil, "create_response", return_value=(
            "ORIGINAL_STRING: Alice\nname: Alice"
        )) as send:
            for model in ("phi4", "qwen3-4b"):
                Config.model = model
                result = OpenAIUtil.get_split(
                    {0: "Alice"}, "names.csv", [0], [], use_type=True, type_list=["name"],
                )
                self.assertEqual(result, {0: {"name": "Alice"}})
            self.assertEqual(send.call_count, 1)
            self.assertEqual(send.call_args.kwargs["model"], "gpt-5.1")
            Config.llm_model = "gpt-4.1"
            OpenAIUtil.get_split(
                {0: "Alice"}, "names.csv", [0], [], use_type=True, type_list=["name"],
            )
            self.assertEqual(send.call_count, 2)
            self.assertEqual(send.call_args.kwargs["model"], "gpt-4.1")

    def test_discovery_model_routes_requests_and_separates_response_cache(self) -> None:
        BenchmarkCache.clear()
        self.addCleanup(BenchmarkCache.clear)
        with patch.object(Config, "money_saver", True), \
             patch.object(Config, "model", "phi4"), \
             patch.object(OpenAIUtil, "create_response", return_value="name") as request:
            for model in ("gpt-4.1", "gpt-5.1"):
                with patch.object(Config, "llm_model", model):
                    OpenAIUtil_Prompt.get_semantic_components("Alice")
                    OpenAIUtil_Prompt.get_semantic_components("Alice")
            self.assertEqual(request.call_count, 2)
            self.assertEqual(
                [call.kwargs["model"] for call in request.call_args_list],
                ["gpt-4.1", "gpt-5.1"],
            )

    def test_disk_cache_separates_discovery_models(self) -> None:
        with patch.object(Config, "llm_model", "gpt-4.1"):
            first = _cache_filename(get_value_split, (), {})
        with patch.object(Config, "llm_model", "gpt-5.1"):
            second = _cache_filename(get_value_split, (), {})
        self.assertNotEqual(first, second)

    def test_disk_cache_separates_slm_query_strategies(self) -> None:
        with patch.object(Config, "language_model_batch", False):
            first = _cache_filename(get_value_split, (), {})
        with patch.object(Config, "language_model_batch", True):
            second = _cache_filename(get_value_split, (), {})
        self.assertNotEqual(first, second)

    def test_disk_cache_separates_split_providers(self) -> None:
        with patch.object(Config, "slm_split", False):
            first = _cache_filename(get_value_split, (), {})
        with patch.object(Config, "slm_split", True):
            second = _cache_filename(get_value_split, (), {})
        self.assertNotEqual(first, second)

    @patch.object(Config, "query_individual_factor", None)
    def test_supplied_types_use_one_exact_extraction_request(self) -> None:
        _ExactExtractionProvider.calls = 0
        result = get_value_split(
            _ExactExtractionProvider,
            {10: "Sample St. 12", 11: "Broadway"},
            "Addresses.csv",
            [10],
            [],
            "label+string",
            False,
            True,
            ["street_name", "house_number"],
        )

        self.assertEqual(_ExactExtractionProvider.calls, 1)
        self.assertEqual(
            result,
            {
                10: {"street name": "Sample St.", "house number": "12"},
                11: {"street name": "Broadway"},
            },
        )


if __name__ == "__main__":
    unittest.main()
