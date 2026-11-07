import tempfile
import unittest
from unittest.mock import Mock, patch

from gdt.config import Config, ExperimentConfig
from gdt.diagnostics import dataset_diagnostics
from LLMUtil.providers import (
    _exact_component_item_is_valid,
    get_value_split,
    request_split_batch,
)
from LLMUtil.ResponseParser import RParser_Prompt
from LLMUtil.semantic import OpenAIUtil_Prompt


class CompositionNormalizationTests(unittest.TestCase):
    def test_punctuation_is_removed_inside_and_between_labels(self):
        response = r'**[First\_Name]**, / _ [last-name2];' + '\n"[café!?]"'
        expected = ["[first_name]", "[lastname2]", "[café]"]
        self.assertTrue(RParser_Prompt.is_composition("", response))
        self.assertEqual(RParser_Prompt.parse_component_str([response]), {0: expected})
        self.assertEqual(RParser_Prompt.get_types_from_split(response), expected)
        self.assertEqual(
            RParser_Prompt.parse_component_str([response], escapespecchar=False)[0],
            ["[First_Name]", "[lastname2]", "[café]"],
        )

    def test_spaces_and_newlines_are_ignored_in_label_text(self):
        response = "[first name] + [last_\nname]\n[first name]"
        self.assertTrue(RParser_Prompt.is_composition("", response))
        self.assertEqual(RParser_Prompt.parse_component_str([response])[0],
                         ["[firstname]", "[last_name]", "[firstname]"])
        self.assertEqual(RParser_Prompt.get_types_from_split(response),
                         ["[firstname]", "[last_name]"])

    def test_cleanup_does_not_accept_text_or_broken_compositions(self):
        for response in ("[name] and [surname]", "Here is [name]", "[name] 2",
                         "[name", "[[name]]", "[]", "[!!!]", "[_]", "[ITEM]", "!!!"):
            with self.subTest(response=response):
                self.assertFalse(RParser_Prompt.is_composition("", response))


class EchoToleranceTests(unittest.TestCase):
    def test_compositions_accept_bare_or_prefixed_echoes_in_all_retry_modes(self):
        originals = ["Alice Smith", "Jones, Bob"]
        bodies = [r"[first\_name], [last_name]", "[last_name] /\n[first_name]"]
        for budgeted in (False, True):
            for individual in (False, True):
                for prefix in ("", "ORIGINAL_STRING: "):
                    for separator in ("\n", " ", ": ", ":\n", ""):
                        with self.subTest(budgeted=budgeted, individual=individual,
                                          prefix=prefix, separator=separator), \
                             patch.object(Config, "query_individual_factor", 1 if budgeted else None), \
                             patch.object(Config, "num_retry", 0):
                            responses = [f"{prefix}{original}{separator}{body}"
                                         for original, body in zip(originals, bodies)]
                            request = (Mock(side_effect=responses) if individual else
                                       Mock(return_value="\n[ITEM]\n".join(reversed(responses))))
                            _, items = request_split_batch(
                                request, originals, instructions="labels", operation="test",
                                one_at_a_time=individual,
                                item_is_valid=RParser_Prompt.is_composition,
                            )
                            self.assertEqual(RParser_Prompt.parse_component_str(items), {
                                0: ["[first_name]", "[last_name]"],
                                1: ["[last_name]", "[first_name]"],
                            })
                            self.assertEqual(request.call_count, 2 if individual else 1)

    def test_bare_echoes_preserve_literal_originals_and_longest_prefix(self):
        originals = ["Alice", "Alice\nSmith", '  Ä "quote"\\path\n[ITEM]  ', "[brackets]"]
        response = "\n[ITEM]\n".join(
            f"{original}: [label_{index}]" for index, original in reversed(list(enumerate(originals)))
        )
        for budgeted in (False, True):
            with self.subTest(budgeted=budgeted), \
                 patch.object(Config, "query_individual_factor", 1 if budgeted else None), \
                 patch.object(Config, "num_retry", 0):
                _, items = request_split_batch(
                    Mock(return_value=response), originals, instructions="labels", operation="test",
                    item_is_valid=RParser_Prompt.is_composition,
                )
                self.assertEqual(items, [f"[label_{index}]" for index in range(4)])

    def test_inline_item_separators_and_mixed_echo_formats(self):
        response = "Bob: [surname][ITEM]ORIGINAL_STRING: Alice [first_name]"
        with patch.object(Config, "query_individual_factor", 1):
            _, items = request_split_batch(
                Mock(return_value=response), ["Alice", "Bob"],
                instructions="labels", operation="test", item_is_valid=RParser_Prompt.is_composition,
            )
        self.assertEqual(items, ["[first_name]", "[surname]"])

    def test_partial_bare_echoes_keep_duplicates_and_retry_only_unresolved_inputs(self):
        request = Mock(side_effect=[
            "B: [b]\n[ITEM]\nA [first_a]\n[ITEM]\nUnknown: [wrong]\n[ITEM]\n[orphan]",
            "A\n[second_a]",
        ])
        with patch.object(Config, "query_individual_factor", 1.5):
            _, items = request_split_batch(
                request, ["A", "A", "B"], instructions="labels", operation="test",
                item_is_valid=RParser_Prompt.is_composition,
            )
        self.assertEqual(items, ["[first_a]", "[second_a]", "[b]"])
        self.assertEqual(request.call_args_list[1].args[0], "ORIGINAL_STRING: A")

    def test_bare_prefixes_and_case_changes_do_not_match_an_original(self):
        for response in ("AliceExtra: [name]", "alice: [name]", "Unknown: [name]"):
            with self.subTest(response=response), patch.object(Config, "query_individual_factor", 1):
                _, items = request_split_batch(
                    Mock(return_value=response), ["Alice"], instructions="labels", operation="test",
                    item_is_valid=RParser_Prompt.is_composition,
                )
                self.assertEqual(items, [""])

    def test_legacy_retries_retain_identity_matched_partial_replies(self):
        request = Mock(side_effect=["Bob: [surname]", "Alice: [first_name]"])
        with patch.object(Config, "query_individual_factor", None), \
             patch.object(Config, "num_retry", 1):
            _, items = request_split_batch(
                request, ["Alice", "Bob"], instructions="labels", operation="test",
                item_is_valid=RParser_Prompt.is_composition,
            )
        self.assertEqual(items, ["[first_name]", "[surname]"])
        self.assertEqual([call.args[0] for call in request.call_args_list],
                         ["Alice\n[ITEM]\nBob"] * 2)

    def test_empty_original_cannot_match_an_unechoed_composition(self):
        with patch.object(Config, "query_individual_factor", 1):
            for response, expected in (("[name]", ""), ("ORIGINAL_STRING: \n[name]", "[name]")):
                with self.subTest(response=response):
                    _, items = request_split_batch(
                        Mock(return_value=response), [""], instructions="labels", operation="test",
                        item_is_valid=RParser_Prompt.is_composition,
                    )
                    self.assertEqual(items, [expected])

    def test_individual_bare_echo_retains_leading_spaces_and_literal_brackets(self):
        with patch.object(Config, "query_individual_factor", 1):
            _, items = request_split_batch(
                Mock(side_effect=["  Alice  : [name]", "[name]"]), ["  Alice  ", "[name]"],
                instructions="labels", operation="test", one_at_a_time=True,
                item_is_valid=RParser_Prompt.is_composition,
            )
        self.assertEqual(items, ["[name]", "[name]"])

    def test_accepted_format_variations_do_not_record_lm_failures(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(Config, "query_individual_factor", 1), \
             dataset_diagnostics(directory, "test.csv", ExperimentConfig()) as diagnostics:
            _, items = request_split_batch(
                Mock(return_value=r"Alice: **[na!me]**, [sur\_name]"), ["Alice"],
                instructions="labels", operation="test", item_is_valid=RParser_Prompt.is_composition,
            )
            self.assertEqual(RParser_Prompt.parse_component_str(items)[0], ["[name]", "[sur_name]"])
        self.assertFalse(diagnostics.result_fields()["has_lm_failures"])

    def test_paired_discovery_uses_normalized_labels_with_correct_string_ids(self):
        with patch.object(Config, "query_individual_factor", 1), \
             patch.object(Config, "use_cache", False), patch.object(Config, "save_cache", False), \
             patch.object(Config, "money_saver", False), patch.object(Config, "perfect_split", False), \
             patch.object(OpenAIUtil_Prompt, "get_semantic_components", return_value=(
                 "Smith, Alice: [last_name], [first\\_name]\n[ITEM]\n"
                 "Alice Smith\n[first\\_name] / [last_name]"
             )) as request:
            result = OpenAIUtil_Prompt.get_split_prompt(
                {0: "Alice Smith", 10: "Smith, Alice"}, "test.csv", [0], [10],
            )
        self.assertEqual(result, {0: ["[first_name]", "[last_name]"],
                                  10: ["[last_name]", "[first_name]"]})
        self.assertEqual(request.call_count, 1)

    def test_exact_values_retain_punctuation_with_bare_inline_and_multiline_echoes(self):
        originals = {2: "Alice_Smith", 8: "https://example.com/a_b?x=1&y=2"}
        for budgeted in (False, True):
            for individual in (False, True):
                with self.subTest(budgeted=budgeted, individual=individual), \
                     patch.object(Config, "query_individual_factor", 1 if budgeted else None), \
                     patch.object(Config, "num_retry", 0), \
                     patch.object(Config, "language_model_batch", not individual):
                    responses = [f"{original} name: {original}" for original in originals.values()]

                    class Provider:
                        __MODEL__ = "phi4"
                        get_semantic_components = (
                            Mock(side_effect=responses) if individual else
                            Mock(return_value="\n[ITEM]\n".join(reversed(responses)))
                        )

                    result = get_value_split(
                        Provider, originals, "test.csv", [], [], "label+string", False, True, ["name"],
                    )
                    self.assertEqual(result, {key: {"name": value} for key, value in originals.items()})
        self.assertFalse(_exact_component_item_is_valid(["name"])("Alice_Smith", "name: AliceSmith"))

    def test_literal_item_marker_inside_an_exact_value_is_preserved(self):
        original = "Alice[ITEM]Smith"
        with patch.object(Config, "query_individual_factor", 1):
            _, items = request_split_batch(
                Mock(return_value=f"{original}\nname: {original}"), [original],
                instructions="extract", operation="test",
                item_is_valid=_exact_component_item_is_valid(["name"]),
            )
        self.assertEqual(items, [f"name: {original}"])


if __name__ == "__main__":
    unittest.main()
