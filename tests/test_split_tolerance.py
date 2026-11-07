from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from gdt.config import Config, ExperimentConfig
from gdt.diagnostics import dataset_diagnostics, split_phase
from gdt.pipeline import PipelineRunner
from LLMUtil.ollama import OLLAMAUtil
from LLMUtil.providers import (
    OpenAIUtil, _exact_component_item_fallback, _exact_component_item_is_valid,
    get_value_split, request_split_batch,
)
from LLMUtil.ResponseParser import RParser_Prompt
from tests.test_split_budget import echoed, originals_in


class SplitToleranceTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(Config.apply, ExperimentConfig())

    def test_extraction_tolerance_waits_for_exhaustion_across_providers_and_retry_modes(self):
        for model in ("phi4", "gpt-test"):
            for budgeted in (False, True):
                for batched in (False, True):
                    with self.subTest(model=model, budgeted=budgeted, batched=batched), \
                         tempfile.TemporaryDirectory() as directory:
                        config = ExperimentConfig(
                            model=model, language_model_batch=batched,
                            num_retry=None if budgeted else 1,
                            query_individual_factor=2 if budgeted else None,
                            output_directory=Path(directory),
                        )
                        Config.apply(config)
                        individual = model == "phi4" and not batched
                        queries = []

                        class Provider:
                            __MODEL__ = model

                            @staticmethod
                            def get_semantic_components(query, *_args, **_kwargs):
                                queries.append(query)
                                originals = ([query] if individual else originals_in(query)
                                             if budgeted else query.split("\n[ITEM]\n"))
                                blocks = []
                                for original in originals:
                                    split = f"name: {original}\nmissing: \nstandalone content"
                                    blocks.append(echoed(original, split)
                                                  if budgeted and not individual else split)
                                return "\n[ITEM]\n".join(blocks)

                        with dataset_diagnostics(directory, "test.csv", config), \
                             split_phase("extraction"):
                            result = get_value_split(
                                Provider, {0: "Alice", 1: "Bob"}, "test.csv", [], [],
                                "label+string", False, True, ["name", "missing"],
                            )
                        self.assertEqual(result, {0: {"name": "Alice"}, 1: {"name": "Bob"}})
                        self.assertEqual(len(queries), 4 if individual else 2)
                        log = (Path(directory) / "errors.txt").read_text(encoding="utf-8")
                        self.assertEqual(log.count("QUERY "), len(queries))
                        self.assertEqual(log.count("TOLERATED:"), 2 if individual else 1)
                        self.assertIn("missing: \nstandalone content", log)

    def test_valid_retry_wins_over_saved_fallback_and_is_not_logged(self):
        for budgeted in (False, True):
            for individual in (False, True):
                with self.subTest(budgeted=budgeted, individual=individual), \
                     tempfile.TemporaryDirectory() as directory:
                    bad = "name: Alice\nmissing:"
                    good = "name: Alice Smith"
                    wrap = lambda body: echoed("Alice Smith", body) if budgeted and not individual else body
                    request = Mock(side_effect=[wrap(bad), wrap(good)])
                    with dataset_diagnostics(directory, "test.csv", ExperimentConfig()), \
                         patch.object(Config, "query_individual_factor", 2 if budgeted else None), \
                         patch.object(Config, "num_retry", 1):
                        _, items = request_split_batch(
                            request, ["Alice Smith"], instructions="extract", operation="test",
                            one_at_a_time=individual,
                            item_is_valid=_exact_component_item_is_valid(["name", "missing"]),
                            item_fallback=_exact_component_item_fallback(["name", "missing"]),
                        )
                    self.assertEqual(items, [good])
                    log = (Path(directory) / "errors.txt").read_text(encoding="utf-8")
                    self.assertEqual(log.count("QUERY "), 1)
                    self.assertNotIn("TOLERATED", log)
                    self.assertNotIn(good, log)

    def test_empty_and_standalone_only_splits_can_fall_back_to_no_components(self):
        for body in ("name:", 'name: ""', "standalone content"):
            for individual in (False, True):
                with self.subTest(body=body, individual=individual), \
                     patch.object(Config, "query_individual_factor", 1):
                    request = Mock(return_value=body if individual else echoed("Alice", body))
                    _, items = request_split_batch(
                        request, ["Alice"], instructions="extract", operation="test",
                        one_at_a_time=individual,
                        item_is_valid=_exact_component_item_is_valid(["name"]),
                        item_fallback=_exact_component_item_fallback(["name"]),
                    )
                    self.assertEqual(items, [""])
                    self.assertEqual(request.call_count, 1)

    def test_cleanup_never_accepts_wrong_substrings_or_unknown_nonempty_labels(self):
        for body in ("street_name: Platz\nempty:", "person: Europaplatz\nstandalone",
                     ": Europaplatz\nempty:"):
            for budgeted in (False, True):
                for individual in (False, True):
                    with self.subTest(body=body, budgeted=budgeted, individual=individual), \
                         patch.object(Config, "query_individual_factor", 2 if budgeted else None), \
                         patch.object(Config, "num_retry", 1):
                        response = echoed("Europaplatz 1", body) if budgeted and not individual else body
                        _, items = request_split_batch(
                            Mock(return_value=response), ["Europaplatz 1"],
                            instructions="extract", operation="test", one_at_a_time=individual,
                            item_is_valid=_exact_component_item_is_valid(["street_name", "empty"]),
                            item_fallback=_exact_component_item_fallback(["street_name", "empty"]),
                        )
                    self.assertEqual(items, [""])
                    self.assertIsNone(_exact_component_item_fallback(
                        ["street_name", "empty"])("Europaplatz 1", body))

    def test_fallback_keeps_literal_values_without_semantic_order_or_overlap_checks(self):
        original = "910 Victoria Ave"
        body = "street_name: Victoria Ave\npostal_code: 910\nstreet_name2: Victoria\nempty:\nstandalone"
        clean = _exact_component_item_fallback(["street_name", "postal_code", "empty"])(original, body)
        self.assertEqual(clean, "street_name: Victoria Ave\npostal_code: 910\nstreet_name2: Victoria")

    def test_budgeted_batch_retries_unresolved_items_and_remembers_earlier_fallback(self):
        first = echoed("A", "name: A\nempty:") + "\n[ITEM]\n" + echoed("B", "name: B")
        request = Mock(side_effect=[first, RuntimeError("offline")])
        with patch.object(Config, "query_individual_factor", 1.5):
            _, items = request_split_batch(
                request, ["A", "B"], instructions="extract", operation="test",
                item_is_valid=_exact_component_item_is_valid(["name", "empty"]),
                item_fallback=_exact_component_item_fallback(["name", "empty"]),
            )
        self.assertEqual(items, ["name: A", "name: B"])
        self.assertEqual(originals_in(request.call_args.args[0]), ["A"])

    def test_duplicate_originals_keep_both_the_valid_and_tolerated_split(self):
        for blocks in (["name: A\nempty:", "name: A"], ["name: A", "name: A\nempty:"]):
            with self.subTest(blocks=blocks), patch.object(Config, "query_individual_factor", 1):
                response = "\n[ITEM]\n".join(echoed("A", body) for body in blocks)
                _, items = request_split_batch(
                    Mock(return_value=response), ["A", "A"], instructions="extract", operation="test",
                    item_is_valid=_exact_component_item_is_valid(["name", "empty"]),
                    item_fallback=_exact_component_item_fallback(["name", "empty"]),
                )
            self.assertEqual(items, ["name: A", "name: A"])

    def test_latest_usable_partial_split_is_selected_without_logging_it_twice(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(Config, "query_individual_factor", 3):
            responses = ["name: Alice\nempty:", "name: Alice Smith\nstandalone", "name: NOT_PRESENT"]
            with dataset_diagnostics(directory, "test.csv", ExperimentConfig()):
                _, items = request_split_batch(
                    Mock(side_effect=responses), ["Alice Smith"], instructions="extract",
                    operation="test", one_at_a_time=True,
                    item_is_valid=_exact_component_item_is_valid(["name", "empty"]),
                    item_fallback=_exact_component_item_fallback(["name", "empty"]),
                )
            self.assertEqual(items, ["name: Alice Smith"])
            log = (Path(directory) / "errors.txt").read_text(encoding="utf-8")
            self.assertEqual(log.count("QUERY "), 3)
            self.assertEqual(log.count("TOLERATED:"), 1)
            selected = [entry for entry in log.split("DATASET:") if "TOLERATED:" in entry]
            self.assertIn(responses[1], selected[0])
            self.assertNotIn(responses[2], selected[0])

    def test_legacy_batch_count_mismatch_returns_empty_results_without_guessing_alignment(self):
        with patch.object(Config, "query_individual_factor", None), \
             patch.object(Config, "num_retry", 1):
            _, items = request_split_batch(
                Mock(return_value="name: A\nempty:"), ["A", "B"],
                instructions="extract", operation="test",
                item_is_valid=_exact_component_item_is_valid(["name", "empty"]),
                item_fallback=_exact_component_item_fallback(["name", "empty"]),
            )
        self.assertEqual(items, ["", ""])

    def test_unmatched_inputs_and_composition_errors_get_empty_placeholders(self):
        with patch.object(Config, "query_individual_factor", 1):
            for response in ("name: A\nempty:", echoed("unknown", "name: A\nempty:"),
                             echoed("A", "name: A\nempty:")):
                with self.subTest(response=response):
                    _, items = request_split_batch(
                        Mock(return_value=response), ["A", "B"], instructions="extract",
                        operation="test", item_is_valid=_exact_component_item_is_valid(["name", "empty"]),
                        item_fallback=_exact_component_item_fallback(["name", "empty"]),
                    )
                    self.assertEqual(items, ["name: A", ""] if response.startswith(
                        "ORIGINAL_STRING: A\n") else ["", ""])
            _, items = request_split_batch(
                Mock(return_value="standalone"), ["A"], instructions="labels",
                operation="test", one_at_a_time=True, item_is_valid=RParser_Prompt.is_composition,
            )
            self.assertEqual(items, [""])

    def test_tolerated_response_does_not_mark_dataset_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "Names.csv")
            path.write_text("Alice Smith,Smith Alice\n" + "UNUSED,UNUSED\n" * 4
                            + "Bob Jones,Jones Bob\nCarol White,White Carol\n", encoding="utf-8")
            discovery = echoed("Alice Smith", "[first_name][last_name]") + "\n[ITEM]\n" + echoed(
                "Smith Alice", "[last_name][first_name]",
            )
            for batched in (False, True):
                config = ExperimentConfig(
                    model="phi4", language_model_batch=batched, slm_split=True, num_examples=1,
                    data_directory=Path(directory),
                    output_directory=Path(directory, str(batched)),
                )

                def respond(prompts, **_kwargs):
                    prompt = prompts[0]
                    originals = (originals_in(prompt) if batched
                                 else [prompt.rsplit("Input:\n", 1)[1].strip()])
                    splits = []
                    for original in originals:
                        if "Examples:" in prompt:
                            body = "[first_name][last_name]"
                        else:
                            body = ("first_name: Alice\nlast_name: Smith" if original == "Alice Smith"
                                    else "last_name: Smith\nfirst_name: Alice")
                            body += "\nfirst_name2:\nstandalone content"
                        splits.append(echoed(original, body) if batched else body)
                    return ["\n[ITEM]\n".join(splits)]

                with self.subTest(batched=batched), \
                     patch.object(OpenAIUtil, "create_response", return_value=discovery), \
                     patch.object(OLLAMAUtil, "get_response_batch", side_effect=respond) as send, \
                     redirect_stdout(io.StringIO()):
                    result = PipelineRunner(config).run(path)
                self.assertTrue(result.succeeded, result.error)
                self.assertIsNone(result.error)
                self.assertEqual(result.evaluation.accuracy, 1.0)
                self.assertEqual(result.evaluation.correct, 2)
                self.assertEqual(send.call_count, 3 if batched else 5)
                log = (config.output_directory / "errors.txt").read_text(encoding="utf-8")
                self.assertEqual(log.count("QUERY "), 2 if batched else 3)
                self.assertEqual(log.count("TOLERATED:"), 2)
                self.assertNotIn("Examples:", log)
