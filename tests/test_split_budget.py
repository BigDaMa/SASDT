from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import io
import re
import unittest
from unittest.mock import Mock, patch

from gdt.cli import parse_arguments
from gdt.config import Config, ExperimentConfig
from LLMUtil.providers import (
    OpenAIUtil, SplitQueryBudget, get_value_split, request_split_batch,
)
from LLMUtil.semantic import OpenAIUtil_Prompt
from LLMUtil.ResponseParser import RParser_Prompt


def echoed(original: str, body: str = "[name]") -> str:
    return f"ORIGINAL_STRING: {original}\n{body}"


def originals_in(query: str) -> list[str]:
    return re.findall(r"^ORIGINAL_STRING: (.*)$", query, re.M)


class SplitBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.object(Config, "query_individual_factor", 2.5)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_reordered_partial_responses_retry_only_missing_originals(self) -> None:
        request = Mock(side_effect=[
            echoed("C") + "\n[ITEM]\n" + echoed("A"), echoed("B"),
        ])
        budget = SplitQueryBudget.for_strings(3)
        _, items = request_split_batch(request, ["A", "B", "C"],
                                      instructions="labels", operation="test", budget=budget)
        self.assertEqual(items, ["[name]"] * 3)
        self.assertEqual(originals_in(request.call_args_list[1].args[0]), ["B"])
        self.assertEqual(budget.used, 4)
        self.assertEqual(budget.limit, 8)

    def test_echo_is_literal_text_with_quotes_backslashes_and_multiline_prefixes(self):
        originals = ['Alice', 'Alice\nSmith', '"quoted" \\path', '  padded  ']
        request = Mock(return_value="\n[ITEM]\n".join(
            echoed(value, f"[label_{index}]")
            for index, value in reversed(list(enumerate(originals)))
        ))
        _, items = request_split_batch(request, originals, instructions="labels", operation="test")
        self.assertEqual(items, [f"[label_{index}]" for index in range(len(originals))])
        self.assertEqual(request.call_args.args[0], "\n[ITEM]\n".join(
            f"ORIGINAL_STRING: {value}" for value in originals
        ))

    def test_duplicates_unknown_originals_and_escaped_strings(self) -> None:
        special = ' Ä "quote"\n[ITEM] '
        request = Mock(side_effect=[
            echoed("unknown") + "\n[ITEM]\n" + echoed(special, "[first]"),
            echoed(special, "[second]"),
        ])
        _, items = request_split_batch(request, [special, special],
                                      instructions="labels", operation="test")
        self.assertEqual(items, ["[first]", "[second]"])
        self.assertEqual(request.call_args_list[1].args[0],
                         f"ORIGINAL_STRING: {special}")

    def test_refinement_keeps_its_distinct_split_body(self) -> None:
        request = Mock(return_value=echoed("Alice", "[first_name]"))
        request_split_batch(
            request,
            ["Alice"],
            bodies=["[name]"],
            instructions="refine",
            operation="test",
        )
        self.assertEqual(request.call_args.args[0], "ORIGINAL_STRING: Alice\n[name]")

    def test_orphan_and_unknown_compositions_do_not_invalidate_complete_results(self):
        response = (
            "ORIGINAL_STRING: Steve Jobs\n\n[first_name][last_name]\n\n"
            "ORIGINAL_STRING: Cook, Tim\n\n[last_name][first_name]\n\n"
            "[last_name][first_name]\n\n"
            "ORIGINAL_STRING: something_not_provided_in_the_dataset\n\n"
            "[first_name][middle_name][last_name]"
        )
        originals = ["Steve Jobs", "Cook, Tim", "Gates, Bill", "Ada Augusta Lovelace"]
        request = Mock(side_effect=[
            response,
            echoed(originals[2], "[last_name][first_name]") + "\n[ITEM]\n"
            + echoed(originals[3], "[first_name][middle_name][last_name]"),
        ])
        with patch.object(Config, "query_individual_factor", 1.5):
            budget = SplitQueryBudget.for_strings(len(originals))
            _, items = request_split_batch(
                request, originals, instructions="labels", operation="test", budget=budget,
                item_is_valid=RParser_Prompt.is_composition,
            )
        self.assertEqual(items, [
            "[first_name][last_name]", "[last_name][first_name]",
            "[last_name][first_name]", "[first_name][middle_name][last_name]",
        ])
        self.assertEqual(originals_in(request.call_args_list[1].args[0]), originals[2:])
        self.assertEqual(budget.used, 6)

    def test_orphan_exact_split_does_not_contaminate_complete_multiline_split(self):
        class GPTProvider:
            __MODEL__ = "gpt-test"
            get_semantic_components = Mock(side_effect=[
                "first_name: orphan\n[ITEM]\n"
                + echoed("Steve Jobs", "first_name: Steve\nlast_name: Jobs")
                + "\n[ITEM]\nfirst_name: missing header\nlast_name: invalid\n[ITEM]\n"
                + echoed("unknown", "first_name: unknown"),
                echoed("Tim Cook", "first_name: Tim\nlast_name: Cook"),
            ])

        with patch.object(Config, "query_individual_factor", 1.5):
            result = get_value_split(
                GPTProvider, {1: "Steve Jobs", 2: "Tim Cook"}, "test.csv", [], [],
                "label+string", False, True, ["first_name", "last_name"],
            )
        self.assertEqual(result, {
            1: {"first name": "Steve", "last name": "Jobs"},
            2: {"first name": "Tim", "last name": "Cook"},
        })
        self.assertEqual(originals_in(GPTProvider.get_semantic_components.call_args.args[0]),
                         ["Tim Cook"])

    def test_header_without_composition_is_retried_but_multiline_labels_are_kept(self):
        request = Mock(side_effect=[
            echoed("A", "") + "\n[ITEM]\n" + echoed("B", "[first_name]\n[last_name]"),
            echoed("A", "[name]"),
        ])
        _, items = request_split_batch(
            request, ["A", "B"], instructions="labels", operation="test",
            item_is_valid=RParser_Prompt.is_composition,
        )
        self.assertEqual(items, ["[name]", "[first_name]\n[last_name]"])
        self.assertEqual(originals_in(request.call_args.args[0]), ["A"])

    def test_individual_response_needs_no_header_but_wrong_echo_is_retried(self):
        request = Mock(side_effect=["[name]", echoed("unknown"), "[surname]"])
        with patch.object(Config, "query_individual_factor", 1.5):
            _, items = request_split_batch(
                request, ["A", "B"], instructions="labels", operation="test",
                one_at_a_time=True, item_is_valid=RParser_Prompt.is_composition,
            )
        self.assertEqual(items, ["[name]", "[surname]"])
        self.assertEqual([call.args[0] for call in request.call_args_list], ["A", "B", "B"])

    def test_failed_individual_does_not_starve_other_retries(self):
        request = Mock(side_effect=["bad", "bad", "[name]", "bad", "[surname]"])
        with patch.object(Config, "query_individual_factor", 1.5):
            _, items = request_split_batch(
                request, ["A", "B", "C"], instructions="labels", operation="test",
                one_at_a_time=True, item_is_valid=RParser_Prompt.is_composition,
            )
        self.assertEqual(items, ["", "[surname]", "[name]"])
        self.assertEqual([call.args[0] for call in request.call_args_list],
                         ["A", "B", "C", "A", "B"])

    def test_budget_is_rounded_up_and_last_batch_is_truncated(self) -> None:
        request = Mock(return_value="malformed")
        budget = SplitQueryBudget.for_strings(3)
        _, items = request_split_batch(request, ["A", "B", "C"], instructions="labels",
                                       operation="test", budget=budget)
        self.assertEqual(items, ["", "", ""])
        self.assertEqual(budget.used, 8)
        self.assertEqual([len(originals_in(call.args[0])) for call in request.call_args_list],
                         [3, 3, 2])

    def test_transport_failures_count_against_budget(self) -> None:
        request = Mock(side_effect=[RuntimeError("offline"), echoed("A")])
        budget = SplitQueryBudget.for_strings(1)
        request_split_batch(request, ["A"], instructions="labels", operation="test", budget=budget)
        self.assertEqual(budget.used, 2)

    def test_factor_one_counts_initial_success_and_allows_empty_split(self) -> None:
        with patch.object(Config, "query_individual_factor", 1):
            budget = SplitQueryBudget.for_strings(1)
            _, items = request_split_batch(Mock(return_value=echoed("A", "")), ["A"],
                                          instructions="labels", operation="test", budget=budget)
        self.assertEqual(items, [""])
        self.assertEqual(budget.used, 1)

    def test_discovery_and_exact_extraction_have_independent_budgets(self) -> None:
        inputs = {index: str(index) for index in range(10)}
        def labels(query, *_args, **_kwargs):
            return "\n[ITEM]\n".join(echoed(value) for value in originals_in(query))
        def values(query, *_args, **_kwargs):
            return "\n[ITEM]\n".join(echoed(value, f"name: {value}")
                                     for value in originals_in(query))
        with patch.object(Config, "query_individual_factor", 1), \
             patch.object(Config, "use_cache", False), \
             patch.object(Config, "save_cache", False), \
             patch.object(Config, "perfect_split", False), \
             patch.object(OpenAIUtil_Prompt, "get_semantic_components", side_effect=labels) as first:
            split = OpenAIUtil_Prompt.get_split_prompt(inputs, "test.csv", list(range(5)),
                                                      list(range(5, 10)))
            self.assertEqual(len(split), 10)
            self.assertEqual(first.call_count, 1)
            provider = Mock()
            provider.get_semantic_components.side_effect = values
            result = get_value_split(provider, {**inputs, 10: "10"}, "test.csv", [], [],
                                     "label+string", False, True, ["name"])
            self.assertEqual(len(result), 11)
            self.assertEqual(provider.get_semantic_components.call_count, 1)

    def test_slm_exact_extraction_isolated_then_retries_malformed_and_failed_items(self) -> None:
        calls = []

        class SLMProvider:
            __MODEL__ = "Phi4"
            responses = iter([
                echoed("A", "name: A"),
                echoed("B", "```python\nprint('B')\n```"),
                RuntimeError("offline"),
                echoed("B", "name: B"),
                echoed("C", "name: C"),
            ])

            @classmethod
            def get_semantic_components(cls, query, *_args, **_kwargs):
                calls.append(query)
                response = next(cls.responses)
                if isinstance(response, Exception):
                    raise response
                return response

        with patch.object(Config, "query_individual_factor", 2):
            result = get_value_split(
                SLMProvider,
                {1: "A", 2: "B", 3: "C"},
                "test.csv",
                [],
                [],
                "label+string",
                False,
                True,
                ["name"],
            )

        self.assertEqual(calls, ["A", "B", "C", "B", "C"])
        self.assertEqual(result, {
            1: {"name": "A"},
            2: {"name": "B"},
            3: {"name": "C"},
        })

    def test_gpt_exact_extraction_keeps_batch_request_policy(self) -> None:
        calls = []

        class GPTProvider:
            __MODEL__ = "GPT-4.1"

            @classmethod
            def get_semantic_components(cls, query, *_args, **_kwargs):
                calls.append(query)
                return "\n[ITEM]\n".join(
                    echoed(original, f"name: {original}")
                    for original in originals_in(query)
                )

        with patch.object(Config, "query_individual_factor", 2):
            result = get_value_split(
                GPTProvider,
                {1: "A", 2: "B"},
                "test.csv",
                [],
                [],
                "label+string",
                False,
                True,
                ["name"],
            )

        self.assertEqual(calls, ["ORIGINAL_STRING: A\n[ITEM]\nORIGINAL_STRING: B"])
        self.assertEqual(result, {1: {"name": "A"}, 2: {"name": "B"}})

    def test_slm_exact_extraction_does_not_retry_after_its_quota_is_exhausted(self) -> None:
        calls = []

        class SLMProvider:
            __MODEL__ = "llama3"

            @classmethod
            def get_semantic_components(cls, query, *_args, **_kwargs):
                calls.append(query)
                return "name: not an original substring"

        with patch.object(Config, "query_individual_factor", 1):
            result = get_value_split(
                    SLMProvider,
                    {1: "A", 2: "B"},
                    "test.csv",
                    [],
                    [],
                    "label+string",
                    False,
                    True,
                    ["name"],
            )

        self.assertEqual(result, {1: {}, 2: {}})
        self.assertEqual(calls, ["A", "B"])

    def test_slm_legacy_retries_unresolved_strings_after_the_initial_pass(self) -> None:
        calls = []

        class SLMProvider:
            __MODEL__ = "qWeN"
            responses = iter(["name: A", "unexpected output", "name: C", "name: B"])

            @classmethod
            def get_semantic_components(cls, query, *_args, **_kwargs):
                calls.append(query)
                return next(cls.responses)

        with patch.object(Config, "query_individual_factor", None), \
             patch.object(Config, "num_retry", 1):
            result = get_value_split(
                SLMProvider,
                {1: "A", 2: "B", 3: "C"},
                "test.csv",
                [],
                [],
                "label+string",
                False,
                True,
                ["name"],
            )

        self.assertEqual(calls, ["A", "B", "C", "B"])
        self.assertEqual(result, {
            1: {"name": "A"},
            2: {"name": "B"},
            3: {"name": "C"},
        })

    def test_echo_instructions_and_sdk_retry_policy_are_mode_specific(self) -> None:
        client = Mock()
        client.with_options.return_value = client
        client.responses.create.return_value.output_text = echoed("A")
        with patch.object(OpenAIUtil, "get_client", return_value=client):
            request_split_batch(
                lambda query: OpenAIUtil.create_response(query, instructions="labels only"),
                ["A"], instructions="labels only", operation="test",
            )
            self.assertIn("ORIGINAL_STRING", client.responses.create.call_args.kwargs["instructions"])
            client.with_options.assert_called_once_with(max_retries=0)
            client.reset_mock()
            client.responses.create.return_value.output_text = "[name]"
            with patch.object(Config, "query_individual_factor", None):
                request_split_batch(
                    lambda query: OpenAIUtil.create_response(query, instructions="labels only"),
                    ["A"], instructions="labels only", operation="test",
                )
            self.assertEqual(client.responses.create.call_args.kwargs["instructions"], "labels only")
            self.assertNotIn("ORIGINAL_STRING", client.responses.create.call_args.kwargs["input"])
            client.with_options.assert_not_called()

    def test_individual_budget_disables_hidden_retries_without_echo_requirements(self):
        client = Mock()
        client.with_options.return_value = client
        client.responses.create.side_effect = [RuntimeError("offline"), Mock(output_text="[name]")]
        with patch.object(OpenAIUtil, "get_client", return_value=client), \
             patch("LLMUtil.providers.time.sleep") as sleep:
            _, items = request_split_batch(
                lambda query: OpenAIUtil.create_response(query, instructions="labels only"),
                ["A"], instructions="labels only", operation="test", one_at_a_time=True,
                item_is_valid=RParser_Prompt.is_composition,
            )
        self.assertEqual(items, ["[name]"])
        self.assertEqual(client.responses.create.call_count, 2)
        self.assertEqual(client.with_options.call_count, 2)
        for call in client.with_options.call_args_list:
            self.assertEqual(call.kwargs, {"max_retries": 0})
        for call in client.responses.create.call_args_list:
            self.assertEqual(call.kwargs["instructions"], "labels only")
            self.assertEqual(call.kwargs["input"], "A")
        sleep.assert_not_called()

    def test_retries_in_discovery_do_not_reduce_extraction_budget(self) -> None:
        inputs = {index: str(index) for index in range(10)}
        label_response = "\n[ITEM]\n".join(echoed(value) for value in inputs.values())
        value_response = "\n[ITEM]\n".join(echoed(value, f"name: {value}")
                                            for value in inputs.values())
        with patch.object(Config, "use_cache", False), \
             patch.object(Config, "save_cache", False), \
             patch.object(Config, "perfect_split", False), \
             patch.object(OpenAIUtil_Prompt, "get_semantic_components",
                          side_effect=["", label_response]) as first:
            OpenAIUtil_Prompt.get_split_prompt(inputs, "test.csv", list(range(5)),
                                               list(range(5, 10)))
            provider = Mock()
            provider.get_semantic_components.side_effect = ["", value_response]
            result = get_value_split(provider, inputs, "test.csv", [], [],
                                     "label+string", False, True, ["name"])
        self.assertEqual(len(result), 10)
        self.assertEqual(first.call_count, 2)
        self.assertEqual(provider.get_semantic_components.call_count, 2)
        # Each phase sent 20 strings within its own 25-string budget.
        self.assertEqual(sum(len(originals_in(call.args[0])) for call in first.call_args_list), 20)
        self.assertEqual(sum(len(originals_in(call.args[0]))
                             for call in provider.get_semantic_components.call_args_list), 20)

    def test_legacy_mode_retries_whole_batch(self) -> None:
        request = Mock(side_effect=["[name]", "[name]\n[ITEM]\n[surname]"])
        with patch.object(Config, "query_individual_factor", None), patch.object(Config, "num_retry", 1):
            request_split_batch(request, ["A", "B"], instructions="labels", operation="test")
        self.assertEqual(request.call_args_list[0], request.call_args_list[1])
        self.assertNotIn("ORIGINAL_STRING", request.call_args.args[0])


class BudgetConfigTests(unittest.TestCase):
    def test_defaults_aliases_and_legacy_selection(self) -> None:
        self.assertEqual(ExperimentConfig().query_individual_factor, 1.5)
        self.assertIsNone(ExperimentConfig(num_retry=2).query_individual_factor)
        with patch.object(Config, "apply"):
            self.assertEqual(parse_arguments([]).config.query_individual_factor, 1.5)
            for alias in ["--query-individual-factor", "--query_individual_factor", "-qif", "--qif"]:
                self.assertEqual(parse_arguments([alias, "1.2"]).config.query_individual_factor, 1.2)
            config = parse_arguments(["--num-retry", "0"]).config
            self.assertIsNone(config.query_individual_factor)
            self.assertEqual(config.num_retry, 0)

    def test_invalid_factors_and_conflicting_options(self) -> None:
        with patch.object(Config, "apply"), redirect_stderr(io.StringIO()):
            for args in [["-qif", "0.9"], ["-qif", "nan"], ["-qif", "inf"],
                         ["-qif", "2.5", "--num-retry", "1"]]:
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    parse_arguments(args)
