from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import itertools
import unittest

from gdt.config import Config
from transform.Evaluation import Evaluation
from transform.Transform import ConstantString, SemanticType, Transform


class TransformTests(unittest.TestCase):
    def test_reordering_and_constants_are_inferred(self) -> None:
        input_strings = {0: "John Smith", 1: "Jane Doe"}
        output_strings = {2: "Smith, John", 3: "Doe, Jane"}
        input_components = {
            0: {"first_name": "John", "last_name": "Smith"},
            1: {"first_name": "Jane", "last_name": "Doe"},
        }
        output_components = {
            0: {"last_name": "Smith", "first_name": "John"},
            1: {"last_name": "Doe", "first_name": "Jane"},
        }
        query = {
            "Bob Brown": {
                "id": 2,
                "components": {"first_name": "Bob", "last_name": "Brown"},
            }
        }
        result = Transform.transform(
            input_strings,
            input_components,
            output_strings,
            output_components,
            query,
            [
                {"input": "John Smith", "output": "Smith, John"},
                {"input": "Jane Doe", "output": "Doe, Jane"},
            ],
        )
        self.assertEqual(result[2]["transformation"], "Brown, Bob")

    def test_evaluation_counts_missing_predictions(self) -> None:
        result = Evaluation.evaluate(
            {1: {"transformation": "expected", "original": {}}},
            {1: "expected", 2: "missing"},
            {1: "first", 2: "second"},
        )
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["incorrect"], 1)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(
            result["output_items"][1],
            {
                "original string": "second",
                "sem. components": [],
                "transformation": "",
                "ground truth": "missing",
            },
        )

    def test_verbose_evaluation_prints_pretty_json_for_every_transformation(self) -> None:
        result = Evaluation.evaluate(
            {
                1: {
                    "transformation": "Brown, Bob",
                    "orig_txt": "Bob Brown",
                    "original": {"first_name": "Bob", "last_name": "Brown"},
                }
            },
            {1: "Brown, Bob"},
        )
        previous_verbose = Config.verbose
        Config.verbose = True
        try:
            output = io.StringIO()
            with redirect_stdout(output):
                Evaluation.print_eval_result(result)
        finally:
            Config.verbose = previous_verbose

        rendered = output.getvalue()
        self.assertIn("Accuracy: 1.0", rendered)
        self.assertIn("Transformation results:", rendered)
        self.assertLess(rendered.index("Correct:"), rendered.index("Wrong:"))
        json_text = rendered[rendered.index("[", rendered.index("Correct:")) :]
        correct_items, end = json.JSONDecoder().raw_decode(json_text)
        wrong_items, _ = json.JSONDecoder().raw_decode(json_text[end:].lstrip("\nWrong:\n"))
        self.assertEqual(correct_items, result["output_items"])
        self.assertEqual(wrong_items, [])
        self.assertNotIn("Incorrect items:", rendered)

    def test_verbose_results_print_correct_items_before_wrong_items(self) -> None:
        result = Evaluation.evaluate(
            {
                1: {"transformation": "wrong", "orig_txt": "first"},
                2: {"transformation": "expected", "orig_txt": "second"},
            },
            {1: "expected", 2: "expected"},
        )
        previous_verbose = Config.verbose
        Config.verbose = True
        try:
            output = io.StringIO()
            with redirect_stdout(output):
                Evaluation.print_eval_result(result)
        finally:
            Config.verbose = previous_verbose

        rendered = output.getvalue()
        correct_json = rendered[rendered.index("[", rendered.index("Correct:")) :]
        correct_items, end = json.JSONDecoder().raw_decode(correct_json)
        wrong_items, _ = json.JSONDecoder().raw_decode(
            correct_json[end:].lstrip("\nWrong:\n")
        )
        self.assertEqual([item["original string"] for item in correct_items], ["second"])
        self.assertEqual([item["original string"] for item in wrong_items], ["first"])

    def test_non_verbose_results_print_only_wrong_items(self) -> None:
        result = Evaluation.evaluate(
            {
                1: {"transformation": "expected", "orig_txt": "first"},
                2: {"transformation": "wrong", "orig_txt": "second"},
            },
            {1: "expected", 2: "expected"},
        )
        previous_verbose = Config.verbose
        Config.verbose = False
        try:
            output = io.StringIO()
            with redirect_stdout(output):
                Evaluation.print_eval_result(result)
        finally:
            Config.verbose = previous_verbose

        rendered = output.getvalue()
        self.assertNotIn("Correct:", rendered)
        self.assertNotIn("Accuracy:", rendered)
        wrong_json = rendered[rendered.index("[", rendered.index("Wrong:")) :]
        self.assertEqual(
            [item["original string"] for item in json.loads(wrong_json)],
            ["second"],
        )

    def test_one_example_distinguishes_copied_values_from_output_constants(self) -> None:
        result = Transform.transform(
            {0: "John"},
            {0: {"name": "John"}},
            {1: "John USA"},
            {0: {"name": "John", "country": "USA"}},
            {
                "Bob": {
                    "id": 1,
                    "text": "Bob",
                    "components": {"name": "Bob"},
                }
            },
            [{"input": "John", "output": "John USA"}],
        )
        self.assertEqual(result[1]["transformation"], "Bob USA")

    def test_majority_output_composition_defines_the_transformation(self) -> None:
        result = Transform.transform(
            {0: "John Smith", 1: "Jane Doe", 2: "Ada Lovelace"},
            {
                0: {"first_name": "John", "last_name": "Smith"},
                1: {"first_name": "Jane", "last_name": "Doe"},
                2: {"first_name": "Ada", "last_name": "Lovelace"},
            },
            {3: "Smith, John", 4: "Doe, Jane", 5: "Ada Lovelace"},
            {
                0: {"last_name": "Smith", "first_name": "John"},
                1: {"last_name": "Doe", "first_name": "Jane"},
                2: {"first_name": "Ada", "last_name": "Lovelace"},
            },
            {
                "Bob Brown": {
                    "id": 3,
                    "components": {"first_name": "Bob", "last_name": "Brown"},
                }
            },
            [
                {"input": "John Smith", "output": "Smith, John"},
                {"input": "Jane Doe", "output": "Doe, Jane"},
                {"input": "Ada Lovelace", "output": "Ada Lovelace"},
            ],
        )

        self.assertEqual(result[3]["transformation"], "Brown, Bob")

    def test_equal_support_prefers_shortest_total_constants(self) -> None:
        components = {0: {"first": "A", "last": "B"},
                      1: {"first": "C", "last": "D"}}
        for outputs, expected in (
            # More constant segments can still contain fewer total characters.
            ({0: "A----B", 1: "(C-D)"}, "(X-Y)"),
            # Sum all segments, rather than comparing individual segment lengths.
            ({0: "---A---B", 1: "C----D"}, "X----Y"),
            ({0: "[A]-[B]", 1: "CD"}, "XY"),
        ):
            with self.subTest(outputs=outputs):
                function = Transform.find_transformation_function(components, components, outputs)
                self.assertEqual(Transform.apply_transformation_function(
                    function, {"first": "X", "last": "Y"}), expected)

    def test_unified_semantic_function_beats_shorter_partial_function(self) -> None:
        function = Transform.find_transformation_function(
            {0: {"name": "A", "whole": "Prefix A"},
             1: {"name": "B", "whole": "wrong"}},
            {0: {"whole": "Prefix A"}, 1: {"name": "B"}},
            {0: "Prefix A", 1: "Prefix B"},
        )
        self.assertEqual(function, [ConstantString("Prefix "), SemanticType("name", "name")])

    def test_literal_only_function_has_lower_priority_despite_greater_support(self) -> None:
        function = Transform.find_transformation_function(
            {0: {"name": "Alice"}, 1: {"name": "Bob"}, 2: {"name": "Cara"}},
            {0: {"name": "x"}, 1: {"name": "x"}, 2: {"name": "Cara"}},
            {0: "x", 1: "x", 2: "Name: Cara!"},
        )
        self.assertEqual(Transform.apply_transformation_function(function, {"name": "Dana"}),
                         "Name: Dana!")

    def test_unified_literal_does_not_hide_a_later_semantic_function(self) -> None:
        function = Transform.find_transformation_function(
            {0: {"name": "unrelated"}, 1: {"name": "A"}},
            {0: {"whole": "A!"}, 1: {"name": "A"}},
            {0: "A!", 1: "A!"},
        )
        self.assertEqual(function, [SemanticType("name", "name"), ConstantString("!")])

    def test_tied_constants_use_the_first_example(self) -> None:
        function = Transform.find_transformation_function(
            {
                0: {"first_name": "John", "last_name": "Smith"},
                1: {"first_name": "Jane", "last_name": "Doe"},
            },
            {
                0: {"last_name": "Smith", "first_name": "John"},
                1: {"last_name": "Doe", "first_name": "Jane"},
            },
            {0: "Smith, John", 1: "Doe; Jane"},
        )

        self.assertEqual(
            Transform.describe_transformation_function(function),
            [
                {"semantic type": "last_name", "occurrence": 1},
                {"constant string": ", "},
                {"semantic type": "first_name", "occurrence": 1},
            ],
        )
        self.assertEqual(
            Transform.apply_transformation_function(
                function,
                {"first_name": "Bob", "last_name": "Brown"},
            ),
            "Brown, Bob",
        )

    def test_complete_template_tie_keeps_first_without_combining_constants(self) -> None:
        components = {
            0: {"first": "A", "last": "B"},
            1: {"first": "C", "last": "D"},
            2: {"first": "E", "last": "F"},
        }
        function = Transform.find_transformation_function(
            components, components, {0: "[A;B!", 1: "(C,D!", 2: "(E,F?"}
        )
        self.assertEqual(
            Transform.apply_transformation_function(function, {"first": "X", "last": "Y"}),
            "[X;Y!",
        )

    def test_independent_majorities_cannot_create_an_unobserved_template(self) -> None:
        components = {0: {"first": "A", "last": "B"},
                      1: {"first": "C", "last": "D"},
                      2: {"first": "E", "last": "F"}}
        outputs = {0: "[A,B!", 1: "(C;D!", 2: "(E,F?"}
        function = Transform.find_transformation_function(components, components, outputs)
        self.assertEqual(Transform.apply_transformation_function(
            function, {"first": "X", "last": "Y"}), "[X,Y!")
        self.assertEqual(sum(Transform.apply_transformation_function(function, value) == outputs[i]
                             for i, value in components.items()), 1)

    def test_whole_output_support_can_beat_the_most_frequent_composition(self) -> None:
        inputs = {0: {"first": "A", "last": "B"},
                  1: {"first": "C", "last": "D"},
                  2: {"first": "E", "last": "F"}}
        # The first two output splits have a different label sequence, but all
        # three pairs support the complete last + comma + first template.
        outputs = {0: {"merged": "B, A"}, 1: {"merged": "D, C"},
                   2: {"last": "F", "first": "E"}}
        function = Transform.find_transformation_function(
            inputs, outputs, {0: "B, A", 1: "D, C", 2: "F, E"})
        self.assertEqual(Transform.apply_transformation_function(
            function, {"first": "X", "last": "Y"}), "Y, X")

    def test_tie_keeps_first_obtained_template_even_if_other_leads_earlier(self) -> None:
        components = {i: {"name": value} for i, value in enumerate("ABCD")}
        function = Transform.find_transformation_function(
            components, components, {0: "[A]", 1: "(B)", 2: "(C)", 3: "[D]"})
        self.assertEqual(Transform.apply_transformation_function(function, {"name": "X"}), "[X]")

    def test_missing_component_boundaries_do_not_vote_against_known_constants(self) -> None:
        components = {
            0: {"first": "A", "last": "B"},
            1: {"first": "C", "last": "D"},
            2: {"first": "E", "last": "F"},
        }
        function = Transform.find_transformation_function(
            components, components, {0: "[A,B]", 1: "[C,missing]", 2: "[E,missing]"}
        )
        self.assertEqual(
            Transform.apply_transformation_function(function, {"first": "X", "last": "Y"}),
            "[X,Y]",
        )

    def test_unmatched_or_overlapping_components_do_not_raise(self) -> None:
        for output, components in [("unmatched", {"first": "A", "last": "B"}),
                                   ("ABC", {"first": "AB", "last": "BC"})]:
            with self.subTest(output=output):
                function = Transform.find_transformation_function(
                    {0: components},
                    {0: components},
                    {0: output},
                )
                self.assertEqual(function, [])

    def test_exact_split_order_does_not_change_the_transformation(self) -> None:
        inputs = {"author": "Dale A. Stirling", "title": "Information rules", "year": "2003"}
        outputs = {"author": "Dale A. Stirling", "title": "Information rules"}
        for input_items in itertools.permutations(inputs.items()):
            for output_items in itertools.permutations(outputs.items()):
                function = Transform.find_transformation_function(
                    {0: dict(input_items)}, {0: dict(output_items)},
                    {0: "Information rules by Dale A. Stirling"},
                )
                self.assertEqual(function, [SemanticType("title", "title"),
                                           ConstantString(" by "), SemanticType("author", "author")])
                self.assertEqual(Transform.apply_transformation_function(
                    function, {"author": "Bob", "title": "Databases"}), "Databases by Bob")

    def test_repeated_and_overlapping_text_uses_distinct_occurrences(self) -> None:
        for output, components, expected in (
            ("New York / York", {"short": "York", "long": "New York"}, "Big / Small"),
            ("York / New York", {"long": "New York", "short": "York"}, "Small / Big"),
        ):
            for items in itertools.permutations(components.items()):
                function = Transform.find_transformation_function(
                    {0: components}, {0: dict(items)}, {0: output},
                )
                self.assertEqual(Transform.apply_transformation_function(
                    function, {"short": "Small", "long": "Big"}), expected)
        function = Transform.find_transformation_function(
            {0: {"word 2": "A", "word": "A"}, 1: {"word": "B", "word 2": "C"}},
            {0: {"word 2": "A", "word": "A"}, 1: {"word": "B", "word 2": "C"}},
            {0: "A / A", 1: "C / B"},
        )
        self.assertEqual(Transform.apply_transformation_function(
            function, {"word": "X", "word 2": "Y"}), "Y / X")

    def test_numbered_type_references_ignore_mapping_order_and_missing_siblings(self) -> None:
        inputs = {"word 10": "C", "word 2": "B", "word": "A"}
        for items in itertools.permutations(inputs.items()):
            function = Transform.find_transformation_function(
                {0: dict(items)}, {0: {"word 2": "B", "word": "A", "word 10": "C"}},
                {0: "CBA"},
            )
            self.assertEqual([part.occurrence for part in function], [9, 1, 0])
            self.assertEqual(Transform.apply_transformation_function(
                function, {"word_2": "Y", "word_10": "Z", "word": "X"}), "ZYX")
            self.assertEqual(Transform.apply_transformation_function(
                function, {"word_10": "Z"}), "Z")

    def test_gap_extraction_resumes_after_missing_component(self) -> None:
        self.assertEqual(
            Transform._constant_gaps("[A,unknown,C;D]", ["A", "B", "C", "D"], 0),
            ["[", None, None, ";", "]"],
        )

    def test_empty_constant_can_win_the_vote(self) -> None:
        components = {0: {"name": "A"}, 1: {"name": "B"}, 2: {"name": "C"}}
        function = Transform.find_transformation_function(
            components, components, {0: "[A]", 1: "B", 2: "C"}
        )
        self.assertEqual(Transform.apply_transformation_function(function, {"name": "X"}), "X")

    def test_consistent_prefix_infix_and_suffix_become_constants(self) -> None:
        function = Transform.find_transformation_function(
            {0: {"name": "John"}, 1: {"name": "Jane"}},
            {0: {"name": "John"}, 1: {"name": "Jane"}},
            {0: "Name: John!", 1: "Name: Jane!"},
        )

        self.assertEqual(
            Transform.describe_transformation_function(function),
            [
                {"constant string": "Name: "},
                {"semantic type": "name", "occurrence": 1},
                {"constant string": "!"},
            ],
        )
        self.assertEqual(
            Transform.apply_transformation_function(function, {"name": "Bob"}),
            "Name: Bob!",
        )

    def test_output_semantic_type_can_map_to_differently_named_input_type(self) -> None:
        function = Transform.find_transformation_function(
            {
                0: {"surname": "Smith"},
                1: {"surname": "Doe"},
            },
            {
                0: {"family_name": "Smith"},
                1: {"family_name": "Doe"},
            },
            {0: "Smith", 1: "Doe"},
        )

        self.assertEqual(
            Transform.apply_transformation_function(function, {"surname": "Brown"}),
            "Brown",
        )

    def test_inconsistent_semantic_types_use_the_most_correct_matches(self) -> None:
        cases = [
            # The output type's name must not outweigh more correct matches.
            ({0: {"name": "A"}, 1: {"surname": "B"}, 2: {"surname": "C"}}, "surname"),
            # A single candidate remains valid despite a wrong or missing value.
            ({0: {"surname": "A"}, 1: {"surname": "wrong"}, 2: {}}, "surname"),
            # Competing candidates and an example with no correct mapping.
            (
                {
                    0: {"name": "A", "surname": "A"},
                    1: {"name": "wrong", "surname": "B"},
                    2: {"name": "wrong", "surname": "wrong"},
                },
                "surname",
            ),
            # Equal whole-template support preserves the first obtained template.
            ({0: {"surname": "A"}, 1: {"name": "B"}, 2: {}}, "surname"),
        ]
        for inputs, expected_type in cases:
            with self.subTest(inputs=inputs):
                function = Transform.find_transformation_function(
                    inputs,
                    {0: {"name": "A"}, 1: {"name": "B"}, 2: {"name": "C"}},
                    {0: "[A]", 1: "[B]", 2: "[C]"},
                )
                self.assertEqual(
                    function,
                    [
                        ConstantString("["),
                        SemanticType(expected_type, expected_type),
                        ConstantString("]"),
                    ],
                )
                self.assertEqual(
                    Transform.apply_transformation_function(
                        function, {expected_type: "query"}
                    ),
                    "[query]",
                )

    def test_no_semantic_matches_use_the_most_common_literal(self) -> None:
        for outputs, expected in [
            (["A", "B", "B"], "B"),
            (["A", "B", "C"], "A"),
            (["longer", "x", "y"], "x"),
            (["longer", "longer", "x"], "longer"),
        ]:
            with self.subTest(outputs=outputs):
                function = Transform.find_transformation_function(
                    {index: {"name": "unrelated"} for index in range(3)},
                    {index: {"name": value} for index, value in enumerate(outputs)},
                    dict(enumerate(outputs)),
                )
                self.assertEqual(function, [ConstantString(expected)])

    def test_majority_composition_preserves_numbered_component_order(self) -> None:
        function = Transform.find_transformation_function(
            {
                0: {"word": "A", "word 2": "B"},
                1: {"word": "C", "word 2": "D"},
                2: {"word": "E", "word 2": "F"},
            },
            {
                0: {"word 2": "B", "word": "A"},
                1: {"word 2": "D", "word": "C"},
                2: {"word": "E", "word 2": "F"},
            },
            {0: "BA", 1: "DC", 2: "EF"},
        )

        self.assertEqual(
            Transform.apply_transformation_function(
                function,
                {"word": "X", "word 2": "Y"},
            ),
            "YX",
        )

    def test_missing_semantic_type_is_skipped_for_one_input(self) -> None:
        function = [
            SemanticType("first_name", "first name"),
            ConstantString(" "),
            SemanticType("last_name", "last name"),
        ]

        self.assertEqual(
            Transform.apply_transformation_function(
                function,
                {"first_name": "Madonna"},
            ),
            "Madonna ",
        )

    def test_missing_type_does_not_stop_other_transformations(self) -> None:
        transformations, _function = Transform.transform_with_function(
            {0: "John Smith"},
            {0: {"first_name": "John", "last_name": "Smith"}},
            {1: "John Smith"},
            {0: {"first_name": "John", "last_name": "Smith"}},
            {
                "Madonna": {
                    "id": 1,
                    "components": {"first_name": "Madonna"},
                },
                "Bob Brown": {
                    "id": 2,
                    "components": {"first_name": "Bob", "last_name": "Brown"},
                },
            },
            [{"input": "John Smith", "output": "John Smith"}],
        )

        self.assertEqual(transformations[1]["transformation"], "Madonna")
        self.assertEqual(transformations[2]["transformation"], "Bob Brown")


if __name__ == "__main__":
    unittest.main()
