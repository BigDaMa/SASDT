from __future__ import annotations

from typing import Dict, List, Sequence

from LLMUtil.LLMInfo import LLMInfo
from LLMUtil.LLMPrompt import LLMPrompt
from LLMUtil.providers import (
    LLMUtil,
    OpenAIUtil,
    SplitQueryBudget,
    _budget_request,
    further_split_query,
    request_split_batch,
)
from LLMUtil.ResponseParser import RParser_Prompt
from Utils.BenchmarkCache import BenchmarkCache, cache_gpt_splits
from Utils.CacheUtil import cache_result, perfect_split_override_prompt
from Utils.config import Config
from gdt.models import resolve_openai_model


class OpenAIUtil_Prompt(LLMUtil):
    get_client = staticmethod(OpenAIUtil.get_client)

    @staticmethod
    def _cached_response(
        operation: str,
        query: str,
        instruction: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_retry: int | None = None,
    ) -> str:
        cache = BenchmarkCache.OpenAICache.setdefault(operation, {})
        model = resolve_openai_model(model or Config.llm_model)
        cache_key = (model, instruction, query)
        if Config.money_saver and not _budget_request.get() and cache_key in cache:
            return cache[cache_key]
        response = OpenAIUtil.create_response(
            query,
            instructions=instruction,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            num_retry=num_retry,
        )
        if Config.money_saver and not _budget_request.get():
            cache[cache_key] = response
        return response

    @staticmethod
    def get_semantic_components(
        query: str,
        file: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        num_retry: int = 3,
    ) -> str:
        del file, seed
        return OpenAIUtil_Prompt._cached_response(
            "OpenAIUtil_Prompt.get_semantic_components",
            query,
            LLMPrompt.gpt_label_component(original_strings=False),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            num_retry=num_retry,
        )

    @staticmethod
    def get_semantic_components_wrapper(
        queries: List[str],
        file: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> List[str]:
        return [
            OpenAIUtil_Prompt.get_semantic_components(
                query, file, model, temperature, max_tokens, seed
            )
            for query in queries
        ]

    get_further_split_query_from_components = staticmethod(
        lambda components: further_split_query(components)
    )

    @staticmethod
    def get_further_split(
        query_components: str,
        sem_types_str: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        num_retry: int = 3,
    ) -> str:
        del seed
        return OpenAIUtil_Prompt._cached_response(
            "OpenAIUtil_Prompt.get_further_split",
            query_components,
            LLMPrompt.gpt_label_refinement(
                semantic_types=sem_types_str,
                original_strings=False,
            ),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            num_retry=num_retry,
        )

    @staticmethod
    def get_final_split(
        query: str,
        sem_types: Sequence[str],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        num_retry: int = 3,
    ) -> str:
        del seed
        return OpenAIUtil_Prompt._cached_response(
            "OpenAIUtil_Prompt.get_final_split",
            query,
            LLMPrompt.gpt_label_final_component(
                semantic_types=", ".join(sem_types),
                original_strings=False,
            ),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            num_retry=num_retry,
        )

    @staticmethod
    @perfect_split_override_prompt()
    @cache_gpt_splits("labels", paired=True)
    @cache_result()
    def get_split_prompt(
        query_dict: Dict[int, str],
        file: str,
        input_idx: List[int],
        output_idx: List[int],
        prompt: str = "label",
        always_split_further: bool = False,
    ) -> Dict[int, List[str]]:
        if prompt != "label":
            raise ValueError("prompt-based clustering requires prompt='label'")
        LLMInfo.set_default_prompt(prompt)
        query_ids = [value_id for pair in zip(input_idx, output_idx, strict=True) for value_id in pair]
        if len(query_ids) != len(query_dict) or set(query_ids) != set(query_dict):
            raise ValueError("example compositions require one complete input/output pair per example")
        originals = [query_dict[value_id] for value_id in query_ids]
        budget = (
            SplitQueryBudget.for_strings(len(originals))
            if Config.query_individual_factor is not None else None
        )
        initial, initial_items = request_split_batch(
            lambda query: OpenAIUtil_Prompt.get_semantic_components(query, file),
            originals,
            instructions=LLMPrompt.gpt_label_component(original_strings=budget is not None),
            budget=budget,
            paired=True,
            item_is_valid=RParser_Prompt.is_composition,
            operation="initial semantic split",
        )
        if always_split_further:
            types = RParser_Prompt.get_types_from_split(initial)
            types_text = RParser_Prompt.parse_sem_types(types)
            _, _refined_items = request_split_batch(
                lambda query: OpenAIUtil_Prompt.get_further_split(
                    query,
                    types_text,
                ),
                originals,
                bodies=[LLMPrompt.composition_refinement_item(
                    original=original, composition=composition,
                    original_strings=budget is not None,
                ) for original, composition in zip(originals, initial_items)],
                instructions=LLMPrompt.gpt_label_refinement(
                    semantic_types=types_text,
                    original_strings=budget is not None,
                ),
                budget=budget,
                paired=True,
                item_is_valid=RParser_Prompt.is_composition,
                operation="refined semantic split",
            )
        # Discovery already supplies the compositions in either retry mode.
        # Do not query all examples again merely to repeat their discovered labels.
        final_items = ([refined or initial for refined, initial in zip(_refined_items, initial_items)]
                       if always_split_further else initial_items)
        parsed = RParser_Prompt.parse_component_str(final_items)
        return {value_id: parsed[index] for index, value_id in enumerate(query_ids)}

    @staticmethod
    @perfect_split_override_prompt()
    @cache_gpt_splits("labels_with_types")
    def get_labels_for_types(
        query_dict: Dict[int, str], file: str, type_list: List[str],
    ) -> Dict[int, List[str]]:
        """Label new source strings with GPT, reusing individual cached results."""
        _, items = request_split_batch(
            lambda query: OpenAIUtil.create_response(
                query,
                instructions=LLMPrompt.gpt_label_final_component(
                    semantic_types=", ".join(type_list),
                    original_strings=False,
                ),
                model=Config.llm_model,
            ),
            list(query_dict.values()),
            instructions=LLMPrompt.gpt_label_final_component(
                semantic_types=", ".join(type_list),
                original_strings=Config.query_individual_factor is not None,
            ),
            operation="source semantic labels",
            item_is_valid=RParser_Prompt.is_composition,
        )
        parsed = RParser_Prompt.parse_component_str(items)
        return {value_id: parsed[index] for index, value_id in enumerate(query_dict)}

    @staticmethod
    def get_sem_component_labels(
        prompts: List[str],
        model: str | None = None,
        temperature: float | None = None,
        num_retry: int = 3,
    ) -> List[str]:
        return [
            OpenAIUtil_Prompt._cached_response(
                "source_labels",
                prompt,
                LLMPrompt.gpt_label_component(original_strings=False),
                model=model or Config.llm_model,
                temperature=temperature,
                num_retry=num_retry,
            )
            for prompt in prompts
        ]
