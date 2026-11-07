from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from LLMUtil.LLMPrompt import LLMPrompt
from LLMUtil.ollama import OLLAMAUtil
from LLMUtil.providers import _budget_request, NovitaUtil, OpenAIUtil
from gdt.diagnostics import capture_prompt


class PromptTests(unittest.TestCase):
    def test_split_order_is_optional_but_composition_order_is_required(self) -> None:
        for template in (LLMPrompt.GPT_SEM_COMPONENT, LLMPrompt.GPT_SEM_COMPONENT_WITH_TYPES,
                         LLMPrompt.GPT_SEM_FINAL_COMPONENT, LLMPrompt.SLM_SEM_COMPONENT,
                         LLMPrompt.SLM_SEM_COMPONENT_WITH_TYPES, LLMPrompt.SLM_SEM_FINAL_COMPONENT):
            self.assertIn("type: exact substring", template)
            self.assertNotIn("lines in source order", template)
        for template in (LLMPrompt.GPT_LABEL_COMPONENT, LLMPrompt.GPT_LABEL_FINAL_COMPONENT,
                         LLMPrompt.SLM_CLUSTER_LABEL):
            self.assertIn("in source order", " ".join(template.split()))

    def test_slm_semantic_prompts_are_complete_and_preserve_header_mode(self) -> None:
        for budgeted in (False, True):
            token = _budget_request.set(budgeted)
            try:
                with patch.object(OLLAMAUtil, "get_response_batch", return_value=["name: Alice"]) as send:
                    OLLAMAUtil.get_semantic_components("Alice")
                    OLLAMAUtil.get_semantic_components("Alice", use_type=True, type_list=["name"])
                    OLLAMAUtil.get_further_split("name: Alice", "name")
                    OLLAMAUtil.get_final_split("Alice", ["name"])
                for call in send.call_args_list:
                    prompt = call.args[0][0]
                    normalized = " ".join(prompt.split())
                    self.assertIn("Output only the requested split", prompt)
                    self.assertIn("no code, markdown, json", normalized.casefold())
                    self.assertIn("Input:\n", prompt)
                    self.assertEqual("ORIGINAL_STRING: <input>" in prompt, budgeted)
                    self.assertNotIn("{input_text}", prompt)
            finally:
                _budget_request.reset(token)

    def test_slm_type_choices_exclude_only_standalone_unknown_labels(self) -> None:
        for builder in (LLMPrompt.slm_sem_component_with_types,
                        LLMPrompt.slm_sem_refinement,
                        LLMPrompt.slm_sem_final_component,
                        LLMPrompt.slm_cluster_label):
            for batched in (False, True):
                for types, expected in (
                    ("name, UNKNOWN, and city", "name, city"),
                    ("[Unknown], [name], [ unknown ], [city]", "[name], [city]"),
                    ("uNkNoWn and name", "name"),
                    ("name and unknown", "name"),
                    ("unknown, [UNKNOWN]", ""),
                    ("unknown_place and known_unknown", "unknown_place and known_unknown"),
                    ("name and city", "name and city"),
                ):
                    with self.subTest(builder=builder.__name__, batched=batched, types=types):
                        kwargs = dict(input_text="Unknown person", original_strings=True,
                                      batched=batched)
                        if builder == LLMPrompt.slm_cluster_label:
                            kwargs["examples"] = [("Unknown example", "[UNKNOWN][name]")]
                        prompt = builder(semantic_types=types, **kwargs)
                        self.assertEqual(prompt, builder(semantic_types=expected, **kwargs))
                        self.assertIn("Unknown person", prompt)
                        if builder == LLMPrompt.slm_cluster_label:
                            self.assertIn(f"Label choices: {expected}\n", prompt)
                            self.assertIn("Unknown example: [UNKNOWN][name]", prompt)
                            self.assertIn("use [unknown]", prompt.casefold())
                        elif expected:
                            self.assertIn(expected, prompt)
                        self.assertNotIn("uNkNoWn", prompt)

    def test_gpt_type_choices_keep_unknown(self) -> None:
        prompt = LLMPrompt.gpt_sem_component_with_types(
            semantic_types="name and UNKNOWN", original_strings=False,
        )
        self.assertIn("name and UNKNOWN", prompt)

    def test_named_semantic_prompts_preserve_general_and_exact_split_rules(self) -> None:
        exact_prompts = (
            LLMPrompt.gpt_sem_component(original_strings=False),
            LLMPrompt.gpt_sem_component_with_types(
                semantic_types="name", original_strings=False
            ),
            LLMPrompt.gpt_sem_refinement(semantic_types="name", original_strings=False),
            LLMPrompt.gpt_sem_final_component(
                semantic_types="name", original_strings=False
            ),
        )
        composition_prompts = (
            LLMPrompt.gpt_label_component(original_strings=False),
            LLMPrompt.gpt_label_refinement(semantic_types="name", original_strings=False),
            LLMPrompt.gpt_label_final_component(
                semantic_types="name", original_strings=False
            ),
        )
        for prompt in exact_prompts + composition_prompts:
            normalized = " ".join(prompt.split())
            self.assertIn("real-world", prompt)
            self.assertIn("one same-order block" if prompt in exact_prompts else
                          "one composition block per string in the provided order", normalized)
            self.assertNotIn("{original_string_response_format}", prompt)

        for prompt in (
            LLMPrompt.gpt_sem_component(original_strings=False),
            LLMPrompt.gpt_sem_refinement(semantic_types="name", original_strings=False),
            LLMPrompt.gpt_sem_final_component(
                semantic_types="name", original_strings=False
            ),
            LLMPrompt.gpt_label_component(original_strings=False),
            LLMPrompt.gpt_label_refinement(semantic_types="name", original_strings=False),
            LLMPrompt.gpt_label_final_component(
                semantic_types="name", original_strings=False
            ),
        ):
            self.assertIn("smallest", prompt.casefold())

        exact_prompt = LLMPrompt.gpt_sem_component_with_types(
            semantic_types="street_name and house_number",
            original_strings=False,
        )
        self.assertIn("exact substring", exact_prompt)
        self.assertIn("Do not merge, split, or rename", exact_prompt)
        self.assertIn("street_name and house_number", exact_prompt)
        for exact_prompt in (
            exact_prompt,
            LLMPrompt.slm_sem_component_with_types(
                input_text="Europaplatz 1", semantic_types="street_name and house_number",
                original_strings=False,
            ),
        ):
            self.assertIn(
                "Do not merge, split, or rename types. "
                "Do not use the label that is not provided.",
                " ".join(exact_prompt.split()),
            )

    def test_header_mode_is_explicit_in_the_rendered_prompt(self) -> None:
        prompt = LLMPrompt.gpt_label_final_component(
            semantic_types="name",
            original_strings=True,
        )
        self.assertIn("ORIGINAL_STRING: <input>", prompt)
        self.assertIn("return that\nheader once", prompt)
        self.assertIn("do not repeat the input elsewhere", prompt)
        self.assertNotIn("{semantic_types}", prompt)
        self.assertNotIn("Output only bracketed labels and [ITEM]", prompt)

    def test_slm_diagnostics_capture_complete_prompt_without_duplicating_input(self):
        data = 'Alice {Smith} \\path\n"quoted"'
        for batched in (False, True):
            with self.subTest(batched=batched), \
                 patch("LLMUtil.ollama.Config.language_model_batch", batched), \
                 patch.object(OLLAMAUtil, "get_response_batch", return_value=["bad"]) as send:
                requests = [
                    lambda: OLLAMAUtil.get_semantic_components(data),
                    lambda: OLLAMAUtil.get_semantic_components(data, use_type=True, type_list=["name"]),
                    lambda: OLLAMAUtil.get_further_split(f"name: {data}", "name"),
                    lambda: OLLAMAUtil.get_final_split(data, ["name"]),
                ]
                for request in requests:
                    with capture_prompt() as captured:
                        request()
                    self.assertEqual(captured, [send.call_args.args[0][0]])
                    self.assertEqual(captured[0].count("\nInput:\n"), 1)
                    self.assertIn(data, captured[0])
                    self.assertNotIn("{input_text}", captured[0])
                    self.assertNotIn("{semantic_types}", captured[0])

    def test_gpt_diagnostics_capture_actual_instructions_after_header_insertion(self):
        for provider in (OpenAIUtil, NovitaUtil):
            with self.subTest(provider=provider.__name__):
                client = Mock()
                client.with_options.return_value = client
                client.responses.create.return_value.output_text = "bad"
                client.chat.completions.create.return_value.choices = [
                    Mock(message=Mock(content="bad")),
                ]
                token = _budget_request.set(True)
                try:
                    with patch.object(provider, "get_client", return_value=client), \
                         capture_prompt() as captured:
                        provider.create_response('ORIGINAL_STRING: Alice {Smith}',
                                                 instructions="Return a split.")
                    if provider is OpenAIUtil:
                        sent = client.responses.create.call_args.kwargs
                        instructions, query = sent["instructions"], sent["input"]
                    else:
                        sent = client.chat.completions.create.call_args.kwargs["messages"]
                        instructions, query = sent[0]["content"], sent[1]["content"]
                    self.assertEqual(captured, [LLMPrompt.diagnostic(
                        instructions=instructions, input_text=query,
                    )])
                    self.assertIn("ORIGINAL_STRING: <input>", instructions)
                    self.assertEqual(captured[0].count("\nInput:\n"), 1)
                finally:
                    _budget_request.reset(token)

    def test_slm_cluster_label_prompt_uses_two_composition_examples(self) -> None:
        prompt = LLMPrompt.slm_cluster_label(
            input_text="query source",
            semantic_types="[component_one], [component_two]",
            examples=[
                ("example input", "[component_one][component_two]"),
                ("example output", "[component_two][component_one]"),
            ],
        )
        self.assertIn("[component_one]", prompt)
        self.assertIn("[component_two]", prompt)
        self.assertIn("query source", prompt)
        self.assertNotIn("demonstration source", prompt)
        self.assertIn("Examples:", prompt)
        self.assertIn("example input: [component_one][component_two]", prompt)
        self.assertIn("example output: [component_two][component_one]", prompt)
        self.assertNotIn("[ITEM]", prompt)
        self.assertNotIn("Expected output:", prompt)

    def test_direct_transformation_prompt_builders_keep_all_prompt_text_centralized(self) -> None:
        examples = [{"input": "Alice Smith", "output": "Smith, Alice"}]
        self.assertEqual(
            LLMPrompt.direct_transform_examples(examples),
            "Q: Alice Smith A: Smith, Alice\n",
        )
        self.assertEqual(
            LLMPrompt.direct_transform_questions(["Bob Jones"]),
            "Q: Bob Jones A:",
        )
        prompt = LLMPrompt.direct_transform_question(
            examples=examples,
            question="Bob Jones",
        )
        self.assertIn(LLMPrompt.GPT_DIRECT_TRANSFORM, prompt)
        self.assertTrue(prompt.endswith("Q: Bob Jones A:"))

    def test_few_shot_prompt_has_one_format_rule_and_only_batches_request_headers(self):
        for batched in (False, True):
            for original_strings in (False, True):
                with self.subTest(batched=batched, original_strings=original_strings):
                    prompt = LLMPrompt.slm_cluster_label(
                        input_text="Tim Cook", semantic_types="[first_name], [last_name]",
                        examples=[("Steve Jobs", "[first_name][last_name]"),
                                  ("Cook, Tim", "[last_name][first_name]")],
                        original_strings=original_strings, batched=batched,
                    )
                    self.assertEqual(prompt.count("ORIGINAL_STRING"),
                                     int(batched and original_strings))
                    self.assertEqual(prompt.count("[ITEM]"), int(batched))
                    self.assertEqual(prompt.count("Steve Jobs: [first_name][last_name]"), 1)
                    self.assertEqual(prompt.count("Cook, Tim: [last_name][first_name]"), 1)
                    self.assertNotIn("Output only", prompt)


if __name__ == "__main__":
    unittest.main()
