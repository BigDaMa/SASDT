from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from gdt.config import Config, ExperimentConfig
from LLMUtil.LLMPrompt import LLMPrompt
from LLMUtil.providers import OpenAIUtil, SplitQueryBudget, request_split_batch
from LLMUtil.ResponseParser import RParser_Prompt
from LLMUtil.semantic import OpenAIUtil_Prompt
from tests.test_split_budget import echoed
from Utils.BenchmarkCache import BenchmarkCache


class PairedCompositionTests(unittest.TestCase):
    def setUp(self):
        Config.apply(ExperimentConfig())
        self.addCleanup(Config.apply, ExperimentConfig())
        BenchmarkCache.clear()
        self.addCleanup(BenchmarkCache.clear)

    def test_pairs_are_adjacent_but_responses_still_have_one_block_per_string(self):
        originals = {0: "tim.cook@apple.com", 1: "Ada Lovelace",
                     8: "cook, tim", 9: "Lovelace, Ada"}
        labels = {0: "[first_name][last_name][email_domain]", 1: "[first_name][last_name]",
                  8: "[last_name][first_name]", 9: "[last_name][first_name]"}
        for budgeted in (False, True):
            with self.subTest(budgeted=budgeted):
                Config.apply(ExperimentConfig(num_retry=None if budgeted else 0))
                response_order = [9, 8, 1, 0] if budgeted else [0, 8, 1, 9]
                response = LLMPrompt.join_items([
                    echoed(originals[i], labels[i]) if budgeted else labels[i]
                    for i in response_order
                ])
                with patch.object(OpenAIUtil, "create_response", return_value=response) as send:
                    result = OpenAIUtil_Prompt.get_split_prompt(originals, "test.csv", [0, 1], [8, 9])
                prefix = "ORIGINAL_STRING: " if budgeted else ""
                self.assertEqual(send.call_args.args[0],
                                 f"{prefix}tim.cook@apple.com\n{prefix}cook, tim\n[ITEM]\n"
                                 f"{prefix}Ada Lovelace\n{prefix}Lovelace, Ada")
                self.assertEqual(result, {0: ["[first_name]", "[last_name]", "[email_domain]"],
                                          1: ["[first_name]", "[last_name]"],
                                          8: ["[last_name]", "[first_name]"],
                                          9: ["[last_name]", "[first_name]"]})
                self.assertEqual(send.call_count, 1)

    def test_partial_retry_keeps_partner_context_and_previously_accepted_composition(self):
        request = Mock(side_effect=[
            LLMPrompt.join_items([echoed("A", "[accepted]"), echoed("C"), echoed("D")]),
            LLMPrompt.join_items([echoed("A", "[changed]"), echoed("B", "[recovered]")]),
        ])
        budget = SplitQueryBudget(limit=6)
        _, items = request_split_batch(
            request, ["A", "B", "C", "D"], instructions="paired labels", operation="test",
            budget=budget, paired=True, item_is_valid=RParser_Prompt.is_composition,
        )
        self.assertEqual(items, ["[accepted]", "[recovered]", "[name]", "[name]"])
        self.assertEqual(request.call_args_list[1].args[0], "ORIGINAL_STRING: A\nORIGINAL_STRING: B")
        self.assertEqual(request.call_count, 2)
        self.assertEqual(budget.used, 6)

    def test_odd_remaining_budget_never_sends_an_orphan_string(self):
        request = Mock(return_value=echoed("A"))
        budget = SplitQueryBudget(limit=3)
        _, items = request_split_batch(
            request, ["A", "B"], instructions="paired labels", operation="test",
            budget=budget, paired=True, item_is_valid=RParser_Prompt.is_composition,
        )
        self.assertEqual(items, ["[name]", ""])
        self.assertEqual(budget.used, 2)
        self.assertEqual(request.call_count, 1)

    def test_refinement_includes_original_pairs_and_their_current_compositions(self):
        for budgeted in (False, True):
            with self.subTest(budgeted=budgeted):
                Config.apply(ExperimentConfig(query_individual_factor=2) if budgeted
                             else ExperimentConfig(num_retry=0))
                def response(first, second):
                    return LLMPrompt.join_items([echoed("Alice", first), echoed("Alice Smith", second)]
                                                if budgeted else [first, second])
                with patch.object(OpenAIUtil, "create_response", side_effect=[
                    response("[name]", "[name]"),
                    response("[first_name]", "[first_name][last_name]"),
                ]) as send:
                    result = OpenAIUtil_Prompt.get_split_prompt(
                        {0: "Alice", 6: "Alice Smith"}, "test.csv", [0], [6], always_split_further=True,
                    )
                prefix = "ORIGINAL_STRING: " if budgeted else ""
                self.assertEqual(send.call_args_list[1].args[0],
                                 f"{prefix}Alice\nCurrent composition: [name]\n"
                                 f"{prefix}Alice Smith\nCurrent composition: [name]")
                self.assertEqual(result, {0: ["[first_name]"], 6: ["[first_name]", "[last_name]"]})

    def test_money_saver_reuses_complete_pairs_and_queries_missing_pairs_together(self):
        Config.apply(ExperimentConfig(money_saver=True))
        with patch.object(OpenAIUtil, "create_response", side_effect=[
            LLMPrompt.join_items([echoed("A"), echoed("B")]),
            LLMPrompt.join_items([echoed("C"), echoed("D")]),
        ]) as send:
            OpenAIUtil_Prompt.get_split_prompt({0: "A", 6: "B"}, "test.csv", [0], [6])
            result = OpenAIUtil_Prompt.get_split_prompt(
                {10: "A", 11: "C", 16: "B", 17: "D"}, "test.csv", [10, 11], [16, 17],
            )
            OpenAIUtil_Prompt.get_split_prompt({0: "A", 6: "B"}, "test.csv", [0], [6])
        self.assertEqual(send.call_count, 2)
        self.assertEqual(send.call_args.args[0], "ORIGINAL_STRING: C\nORIGINAL_STRING: D")
        self.assertEqual(result, {10: ["[name]"], 11: ["[name]"], 16: ["[name]"], 17: ["[name]"]})

    def test_cached_string_cannot_replace_discovery_with_a_different_partner(self):
        Config.apply(ExperimentConfig(money_saver=True))
        with patch.object(OpenAIUtil, "create_response", side_effect=[
            LLMPrompt.join_items([echoed("A", "[whole]"), echoed("B", "[whole]")]),
            LLMPrompt.join_items([echoed("A", "[part][other]"), echoed("C", "[part]")]),
            echoed("A", "[part][other]"),
        ]) as send:
            original = OpenAIUtil_Prompt.get_split_prompt({0: "A", 6: "B"}, "test.csv", [0], [6])
            changed = OpenAIUtil_Prompt.get_split_prompt({0: "A", 6: "C"}, "test.csv", [0], [6])
            self.assertEqual(send.call_args.args[0], "ORIGINAL_STRING: A\nORIGINAL_STRING: C")
            again = OpenAIUtil_Prompt.get_split_prompt({0: "A", 6: "B"}, "test.csv", [0], [6])
            labels = OpenAIUtil_Prompt.get_labels_for_types(
                {8: "A"}, "test.csv", ["[whole]", "[part]", "[other]"],
            )
        self.assertEqual(send.call_count, 3)
        self.assertEqual(original, again)
        self.assertEqual(changed[0], ["[part]", "[other]"])
        self.assertEqual(labels, {8: ["[part]", "[other]"]})


if __name__ == "__main__":
    unittest.main()
