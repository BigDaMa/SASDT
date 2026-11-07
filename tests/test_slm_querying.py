from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from Clustering.collapsecluster import CollapseCluster
from DataAccess.dataaccess import DataAccess
from DataAccess.datafeature import DataFeature
from LLMUtil.LLMPrompt import LLMPrompt
from LLMUtil.ollama import OLLAMAUtil
from LLMUtil.providers import OpenAIUtil
from LLMUtil.semantic import OpenAIUtil_Prompt
from gdt.config import Config, ExperimentConfig
from gdt.diagnostics import clear_errors, dataset_diagnostics, split_phase
from gdt.pipeline import PipelineRunner
from tests.test_split_budget import echoed, originals_in


class SLMQueryingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(Config.apply, ExperimentConfig())
        self.addCleanup(OLLAMAUtil.set_model, OLLAMAUtil.__MODEL__)
        self.addCleanup(OLLAMAUtil.set_mode, OLLAMAUtil.__MODE__)
        Config.apply(ExperimentConfig(model="phi4"))
        OLLAMAUtil.set_model("phi4")

    def test_transport_matrix_for_compositions_and_extraction(self) -> None:
        examples = [("Alice Smith", "[name]"), ("Smith Alice", "[name]")]
        originals = {7: "Bob", 9: "Carol"}
        for mode in ("gen", "chat"):
            for batched in (False, True):
                for budgeted in (False, True):
                    for stage in ("labels", "values"):
                        with self.subTest(mode=mode, batched=batched,
                                          budgeted=budgeted, stage=stage):
                            Config.apply(ExperimentConfig(
                                model="phi4", language_model_batch=batched,
                                num_retry=None if budgeted else 0,
                            ))
                            OLLAMAUtil.set_mode(mode)
                            prompts = []

                            def post(_url, *, json, timeout):
                                self.assertNotIn("keep_alive", json)
                                self.assertIs(json["think"], False)
                                prompt = (json["prompt"] if mode == "gen"
                                          else json["messages"][0]["content"])
                                prompts.append(prompt)
                                query = prompt.rsplit("Input:\n", 1)[1].strip()
                                values = (originals_in(query) if budgeted and batched
                                          else query.split(LLMPrompt.ITEM_SEPARATOR))
                                responses = []
                                for value in values:
                                    body = "[name]" if stage == "labels" else f"name: {value}"
                                    responses.append(echoed(value, body)
                                                     if budgeted and batched else body)
                                # Echo mode must match by original, not response order.
                                if budgeted and batched:
                                    responses.reverse()
                                text = LLMPrompt.join_items(responses)
                                response = Mock()
                                response.json.return_value = {
                                    "response": text, "thinking": "[wrong_type]\nname: WRONG",
                                    "message": {"content": text,
                                                "thinking": "[wrong_type]\nname: WRONG"},
                                }
                                return response

                            request = Mock(side_effect=post)
                            with patch.dict("sys.modules", {"requests": SimpleNamespace(post=request)}):
                                if stage == "labels":
                                    result = OLLAMAUtil.get_labels_for_types(
                                        originals, "test.csv", ["[name]"], examples,
                                    )
                                    self.assertEqual(result, {7: ["[name]"], 9: ["[name]"]})
                                else:
                                    result = OLLAMAUtil.get_split(
                                        originals, "test.csv", [], [],
                                        use_type=True, type_list=["name"],
                                    )
                                    self.assertEqual(result, {7: {"name": "Bob"},
                                                              9: {"name": "Carol"}})
                            self.assertEqual(request.call_count, 1 if batched else 2)
                            for prompt in prompts:
                                self.assertEqual("[ITEM]" in prompt, batched)
                                self.assertEqual("ORIGINAL_STRING:" in prompt, budgeted and batched)
                                if stage == "labels":
                                    for text, composition in examples:
                                        self.assertIn(f"{text}: {composition}", prompt)
                                else:
                                    self.assertNotIn("Examples:", prompt)

    def test_slm_batches_retry_missing_and_malformed_results(self) -> None:
        Config.language_model_batch = True
        for stage in ("labels", "values"):
            with self.subTest(stage=stage):
                body = lambda value: "[name]" if stage == "labels" else f"name: {value}"
                responses = [
                    LLMPrompt.join_items([echoed("C", body("C")),
                                          echoed("A", "invalid output")]),
                    LLMPrompt.join_items([echoed("B", body("B")), echoed("A", body("A"))]),
                ]
                with patch.object(OLLAMAUtil, "_split_response", side_effect=responses) as send:
                    if stage == "labels":
                        result = OLLAMAUtil.get_labels_for_types(
                            {0: "A", 1: "B", 2: "C"}, "test.csv", ["[name]"],
                            [("X", "[name]"), ("Y", "[name]")],
                        )
                    else:
                        result = OLLAMAUtil.get_split(
                            {0: "A", 1: "B", 2: "C"}, "test.csv", [], [],
                            use_type=True, type_list=["name"],
                        )
                self.assertEqual(len(result), 3)
                self.assertEqual([originals_in(call.args[0]) for call in send.call_args_list],
                                 [["A", "B", "C"], ["A", "B"]])

    def test_individual_compositions_finish_initial_pass_before_retry(self) -> None:
        with patch.object(OLLAMAUtil, "_split_response", side_effect=[
            "invalid", "[name]", "[name]",
        ]) as send:
            result = OLLAMAUtil.get_labels_for_types(
                {0: "A", 1: "B"}, "test.csv", ["[name]"], [("X", "[name]"), ("Y", "[name]")],
            )
        self.assertEqual(result, {0: ["[name]"], 1: ["[name]"]})
        self.assertEqual([call.args[0].rsplit("Input:\n", 1)[1].strip()
                          for call in send.call_args_list], ["A", "B", "A"])

    def test_legacy_slm_batches_validate_results_and_retry_the_whole_batch(self) -> None:
        Config.apply(ExperimentConfig(model="phi4", language_model_batch=True, num_retry=1))
        for stage in ("labels", "values"):
            with self.subTest(stage=stage):
                valid = "[name]\n[ITEM]\n[name]" if stage == "labels" else "name: A\n[ITEM]\nname: B"
                with patch.object(OLLAMAUtil, "_split_response", side_effect=[
                    "invalid\n[ITEM]\ninvalid", valid,
                ]) as send:
                    if stage == "labels":
                        result = OLLAMAUtil.get_labels_for_types(
                            {0: "A", 1: "B"}, "test.csv", ["[name]"], [("X", "[name]")],
                        )
                    else:
                        result = OLLAMAUtil.get_split(
                            {0: "A", 1: "B"}, "test.csv", [], [],
                            use_type=True, type_list=["name"],
                        )
                self.assertEqual(len(result), 2)
                self.assertEqual(send.call_count, 2)
                self.assertEqual(send.call_args_list[0], send.call_args_list[1])

    def test_composition_budget_exhaustion_and_empty_input(self) -> None:
        for batched in (False, True):
            with self.subTest(batched=batched):
                Config.language_model_batch = batched
                Config.query_individual_factor = 1
                with patch.object(OLLAMAUtil, "_split_response", return_value="invalid") as send:
                    self.assertEqual(OLLAMAUtil.get_labels_for_types({}, "test.csv", [], []), {})
                    send.assert_not_called()
                    result = OLLAMAUtil.get_labels_for_types(
                        {0: "A", 1: "B"}, "test.csv", ["[name]"], [("X", "[name]")],
                    )
                    self.assertEqual(result, {0: [], 1: []})
                self.assertEqual(send.call_count, 1 if batched else 2)

    def test_gpt_stages_remain_batched_with_either_flag_and_retry_mode(self) -> None:
        for batched in (False, True):
            for budgeted in (False, True):
                with self.subTest(batched=batched, budgeted=budgeted):
                    Config.apply(ExperimentConfig(
                        language_model_batch=batched, num_retry=None if budgeted else 0,
                    ))

                    def respond(query, *, instructions, **_kwargs):
                        originals = (originals_in(query) if budgeted
                                     else query.split(LLMPrompt.ITEM_SEPARATOR))
                        if not budgeted and "Inputs are string pairs" in instructions:
                            originals = [value for pair in originals for value in pair.splitlines()]
                        items = []
                        for original in originals:
                            body = (f"name: {original}" if "type: exact substring" in instructions
                                    else "[name]")
                            items.append(echoed(original, body) if budgeted else body)
                        return LLMPrompt.join_items(items)

                    with patch.object(OpenAIUtil, "create_response", side_effect=respond) as send:
                        OpenAIUtil_Prompt.get_split_prompt({0: "A", 2: "B"}, "test.csv", [0], [2])
                        OpenAIUtil_Prompt.get_labels_for_types({3: "C", 4: "D"}, "test.csv", ["[name]"])
                        OpenAIUtil.get_split({0: "A", 2: "B"}, "test.csv", [0], [2],
                                             use_type=True, type_list=["name"])
                    self.assertEqual(send.call_count, 3)
                    self.assertEqual(send.call_args_list[0].args[0],
                                     "ORIGINAL_STRING: A\nORIGINAL_STRING: B" if budgeted else "A\nB")
                    for call in send.call_args_list[1:]:
                        self.assertIn("\n[ITEM]\n", call.args[0])

    def test_clustering_passes_the_full_example_pool_to_slm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "Names.csv")
            path.write_text("Alice Smith,Smith Alice\nBob Jones,Jones Bob\n"
                            + "UNUSED,UNUSED\n" * 3
                            + "Carol White,HELD_OUT_TARGET\nDora Black,OTHER_TARGET\n",
                            encoding="utf-8")
            example_labels = {0: ["[first]", "[last]"], 1: ["[first]", "[last]"],
                              7: ["[last]", "[first]"], 8: ["[last]", "[first]"]}
            for strategy in (CollapseCluster.run_cluster_collapse_promptsplit,
                             CollapseCluster.run_cluster_collapse_promptsplit_gpt):
                with self.subTest(strategy=strategy.__name__), \
                     patch.object(DataAccess, "DATA_FOLDER", directory), \
                     patch.object(DataAccess, "NUM_EXAMPLES", 2), \
                     patch.object(DataFeature, "data_features", {}), \
                     patch.object(OpenAIUtil_Prompt, "get_split_prompt", return_value=example_labels), \
                     patch.object(OpenAIUtil_Prompt, "get_labels_for_types") as llm, \
                     patch.object(OLLAMAUtil, "get_labels_for_types", return_value={
                         5: ["[first]", "[last]"], 6: ["[first]", "[last]"],
                     }) as slm:
                    clusters = strategy(path.name)
                    self.assertEqual(clusters[-1], {7, 8})
                    self.assertEqual(slm.call_args.args[0], {5: "Carol White", 6: "Dora Black"})
                    self.assertEqual(slm.call_args.args[3], [
                        ("Alice Smith", "[first][last]"), ("Bob Jones", "[first][last]"),
                        ("Smith Alice", "[last][first]"), ("Jones Bob", "[last][first]"),
                    ])
                    llm.assert_not_called()
                    self.assertEqual(OLLAMAUtil.__MODEL__, "phi4")

    def test_each_few_shot_request_and_retry_selects_a_random_companion_and_logs_actual_prompt(self):
        examples = [("Short", "[name]"), ("Ada Augusta Lovelace", "[first][middle][last]"),
                    ("Bob Jones", "[first][last]"), ("Jobs, Steve", "[last][first]")]
        for budgeted in (False, True):
            for batched in (False, True):
                with self.subTest(budgeted=budgeted, batched=batched), \
                     tempfile.TemporaryDirectory() as directory:
                    Config.query_individual_factor = 2 if budgeted else None
                    Config.num_retry = 1
                    Config.language_model_batch = batched
                    # The first attempt fails. Every actual request, including
                    # the retry, must consume exactly one random choice.
                    if batched:
                        valid = (echoed("A") + "\n[ITEM]\n" + echoed("B") if budgeted
                                 else "[name]\n[ITEM]\n[name]")
                        responses = ["invalid", valid]
                    else:
                        responses = ["invalid", "[name]", "[name]"]
                    companions = [examples[0], examples[2], examples[3]][:len(responses)]
                    clear_errors(directory)
                    with dataset_diagnostics(directory, "test.csv", ExperimentConfig()), \
                         split_phase("compositions"), \
                         patch("LLMUtil.ollama.random.choice", side_effect=companions) as choose, \
                         patch.object(OLLAMAUtil, "_split_response", side_effect=responses) as send:
                        OLLAMAUtil.get_labels_for_types({0: "A", 1: "B"}, "test.csv", ["[name]"],
                                                       examples)
                    self.assertEqual(choose.call_count, send.call_count)
                    for call, companion in zip(send.call_args_list, companions):
                        prompt = call.args[0]
                        self.assertEqual(prompt.count("\nInput:\n"), 1)
                        self.assertIn(f"{examples[1][0]}: {examples[1][1]}", prompt)
                        self.assertIn(f"{companion[0]}: {companion[1]}", prompt)
                    for call in choose.call_args_list:
                        self.assertEqual(call.args[0], [examples[0], examples[2], examples[3]])
                    log = (Path(directory) / "errors.txt").read_text(encoding="utf-8")
                    self.assertIn(f"QUERY 1:\n{send.call_args_list[0].args[0]}\nRESPONSE 1:\ninvalid", log)
                    self.assertEqual(log.count("QUERY "), 1)
                    self.assertNotIn(f"{examples[2][0]}: {examples[2][1]}", log)

    def test_pipeline_defaults_to_batched_llm_extraction_and_allows_slm_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "Names.csv")
            path.write_text("Alice Smith,Smith Alice\n" + "UNUSED,UNUSED\n" * 4
                            + "Bob Jones,Jones Bob\nCarol White,White Carol\n"
                            + "Eve-Brown,Brown Eve\n", encoding="utf-8")
            for slm_split, batched in ((False, False), (False, True), (True, False), (True, True)):
                config = ExperimentConfig(
                    model="phi4", llm_model="gpt-4.1", slm_split=slm_split,
                    language_model_batch=batched, num_examples=1,
                    data_directory=Path(directory), output_directory=Path(directory, "results"),
                )
                components = {"Alice Smith": "first_name: Alice\nlast_name: Smith",
                              "Smith Alice": "last_name: Smith\nfirst_name: Alice",
                              "Eve-Brown": "first_name: Eve\nlast_name: Brown"}

                def llm_response(query, *, instructions, model, **_kwargs):
                    self.assertEqual(model, "gpt-4.1")
                    if "type: exact substring" in instructions:
                        self.assertFalse(slm_split)
                        self.assertEqual(originals_in(query), ["Alice Smith", "Eve-Brown", "Smith Alice"])
                        return LLMPrompt.join_items([
                            echoed(value, components[value]) for value in originals_in(query)
                        ])
                    self.assertEqual(originals_in(query), ["Alice Smith", "Smith Alice"])
                    return LLMPrompt.join_items([
                        echoed("Alice Smith", "[first_name][last_name]"),
                        echoed("Smith Alice", "[last_name][first_name]"),
                    ])

                def slm_response(prompt):
                    values = (originals_in(prompt) if batched
                              else [prompt.rsplit("Input:\n", 1)[1].strip()])
                    if "Examples:" in prompt:
                        self.assertIn("Alice Smith: [first_name][last_name]", prompt)
                        self.assertIn("Smith Alice: [last_name][first_name]", prompt)
                        return LLMPrompt.join_items([
                            echoed(value, "[first_name][last_name]") if batched
                            else "[first_name][last_name]" for value in values
                        ])
                    self.assertTrue(slm_split)
                    return LLMPrompt.join_items([
                        echoed(value, components[value]) if batched
                        else components[value] for value in values
                    ])

                with self.subTest(slm_split=slm_split, batched=batched), \
                     patch.object(OpenAIUtil, "create_response", side_effect=llm_response) as llm, \
                     patch.object(OLLAMAUtil, "_split_response", side_effect=slm_response) as slm, \
                     redirect_stdout(io.StringIO()):
                    result = PipelineRunner(config).run(path)
                self.assertTrue(result.succeeded, result.error)
                self.assertEqual(result.evaluation.accuracy, 1.0)
                self.assertEqual(result.evaluation.correct, 3)
                self.assertEqual(llm.call_count, 1 if slm_split else 2)
                self.assertEqual(slm.call_count, (1 if batched else 3) * (2 if slm_split else 1))


if __name__ == "__main__":
    unittest.main()
