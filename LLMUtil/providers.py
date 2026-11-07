from __future__ import annotations

from pathlib import Path
from collections import defaultdict, deque
from contextvars import ContextVar
from dataclasses import dataclass
import math
import re
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence

from gdt.diagnostics import capture_prompt, record_empty_results, record_exchange, record_prompt
from gdt.timing import exclude_output
from gdt.models import OPENAI_MODEL_ALIASES, is_gpt_model, resolve_openai_model
from gdt.pricing import pricing_active, track_request
from LLMUtil.LLMInfo import LLMInfo
from LLMUtil.LLMPrompt import LLMPrompt
from LLMUtil.ResponseParser import RParser
from Utils.CacheUtil import cache_result, perfect_split_override
from Utils.config import Config
from Utils.BenchmarkCache import cache_gpt_splits


_budget_request = ContextVar("split_budget_request", default=False)
_echo_originals = ContextVar("split_echo_originals", default=True)


def original_string_headers() -> bool:
    """Budgeted batches need identity headers; individual requests do not."""
    return _budget_request.get() and _echo_originals.get()


@dataclass
class SplitQueryBudget:
    limit: int
    used: int = 0

    @classmethod
    def for_strings(cls, count: int) -> SplitQueryBudget:
        factor = Config.query_individual_factor
        if factor is None or not math.isfinite(factor) or factor < 1:
            raise ValueError("query_individual_factor must be finite and at least 1")
        return cls(math.ceil(factor * count))


def request_split_batch(
    request: Callable[[str], str],
    originals: Sequence[str],
    *,
    instructions: str,
    operation: str,
    budget: SplitQueryBudget | None = None,
    bodies: Sequence[str] | None = None,
    one_at_a_time: bool = False,
    paired: bool = False,
    item_is_valid: Callable[[str, str], bool] | None = None,
    item_fallback: Callable[[str, str], str | None] | None = None,
) -> tuple[str, list[str]]:
    """Match echoed originals and retry only missing items within a shared budget.

    ``one_at_a_time`` keeps the initial pass isolated, then retries unresolved
    items only after every original has had one attempt.
    ``paired`` keeps adjacent example strings together, including on retries;
    both strings count against the budget, with one response block per string.
    """
    bodies = list(originals if bodies is None else bodies)
    if paired and (one_at_a_time or len(originals) % 2):
        raise ValueError("paired queries require complete pairs in batched requests")
    join_query = LLMPrompt.join_pairs if paired else LLMPrompt.join_items
    if not originals:
        return "", []
    if Config.query_individual_factor is None:
        if one_at_a_time:
            return _request_split_items_with_retry(
                request, originals, bodies, instructions, operation, item_is_valid, item_fallback
            )
        query = join_query([str(value) for value in bodies])
        return request_split_with_retry(
            lambda: request(query),
            prompt=format_prompt_for_diagnostics(query, instructions),
            expected_count=len(originals),
            operation=operation,
            originals=originals,
            item_is_valid=item_is_valid,
            item_fallback=item_fallback,
        )
    budget = budget if budget is not None else SplitQueryBudget.for_strings(len(originals))
    history = []
    found: dict[int, str] = {}
    fallbacks = {}
    last_response = ""
    query = ""
    group_size = 2 if paired else 1
    pending = deque(tuple(range(index, index + group_size))
                    for index in range(0, len(originals), group_size))
    # Build the matcher once, and never build it for individual requests.
    header_pattern = None if one_at_a_time else _original_header_pattern(originals)
    while pending and budget.limit - budget.used >= group_size:
        count = 1 if one_at_a_time else min(len(pending), (budget.limit - budget.used) // group_size)
        groups = [pending.popleft() for _ in range(count)]
        batch = [index for group in groups for index in group]
        query = bodies[batch[0]] if one_at_a_time else join_query([
            _split_batch_item(originals[index], bodies[index]) for index in batch
        ])
        budget.used += len(batch)
        token = _budget_request.set(True)
        echo_token = _echo_originals.set(not one_at_a_time)
        try:
            last_response, last_prompt, reason = _request_split_response(
                lambda: request(query), format_prompt_for_diagnostics(query, instructions),
            )
        finally:
            _echo_originals.reset(echo_token)
            _budget_request.reset(token)
        received = {}
        if reason is None:
            if one_at_a_time:
                index = batch[0]
                items = _single_response_items(last_response, originals[index])
                if len(items) == 1:
                    received[index] = items[0]
                reason = _split_items_error(items, 1, [originals[index]], item_is_valid)
                if not items and last_response.lstrip().startswith("ORIGINAL_STRING:"):
                    reason = "Output format error. Unknown ORIGINAL_STRING."
                if reason is None:
                    found[index] = items[0]
            else:
                matched = _match_batch_response(
                    last_response, originals, batch, header_pattern, item_is_valid,
                )
                reason = _batch_response_error(
                    last_response, originals, batch, header_pattern, matched,
                )
                found.update((index, item) for index, item in matched.items() if index not in found)
                if reason is not None and item_fallback is not None:
                    received = _match_batch_response(
                        last_response, originals, batch, header_pattern, None,
                    )
        exchange = record_exchange(history, last_prompt, last_response, reason, operation)
        _remember_fallbacks(fallbacks, received, originals, found, exchange, item_fallback)
        # Retries retain the partner as context but never overwrite an accepted split.
        pending.extend(group for group in groups if any(index not in found for index in group))
    _accept_fallbacks(found, fallbacks, originals)
    _fill_empty_results(found, originals, operation)
    items = [found[index] for index in range(len(originals))]
    return LLMPrompt.join_items(items), items


def _remember_fallbacks(fallbacks, received, originals, found, exchange, item_fallback):
    """Retain the latest usable partial split without ending its normal retries."""
    if exchange is None or item_fallback is None:
        return
    unresolved = defaultdict(deque)
    for index in received:
        if index not in found:
            unresolved[originals[index]].append(index)
    for index, item in received.items():
        available = unresolved[originals[index]]
        if available:
            cleaned = item_fallback(originals[index], item)
            if cleaned is not None:
                fallbacks[available.popleft()] = (cleaned, exchange)


def _accept_fallbacks(found, fallbacks, originals):
    for index, (cleaned, exchange) in fallbacks.items():
        if index not in found:
            found[index] = cleaned
            exchange.tolerated_inputs.append(originals[index])


def _fill_empty_results(found, originals, operation):
    missing = [index for index in range(len(originals)) if index not in found]
    record_empty_results([originals[index] for index in missing], operation)
    for index in missing:
        # Preserve one result per original so parsers retain ID alignment and
        # produce [] for compositions or {} for exact component mappings.
        found[index] = ""


def _request_split_response(request, fallback_prompt):
    """Keep diagnostics identical to the actual request, including random examples."""
    with capture_prompt() as captured:
        try:
            response = request()
            reason = None
            if not isinstance(response, str):
                response = repr(response)
                reason = "Output format error. Response is not text."
        except Exception as error:
            response = f"{type(error).__name__}: {error}"
            reason = f"Request failed. {type(error).__name__}."
    return response, captured[0] if captured else fallback_prompt, reason


def _split_items_error(items, expected_count, originals, item_is_valid):
    if len(items) != expected_count:
        return f"Number of split mismatch. {len(items)}/{expected_count} splits obtained."
    if item_is_valid is not None and (originals is None or any(
        not item_is_valid(original, item) for original, item in zip(originals, items)
    )):
        return "Output format/content error. Invalid split contents."
    return None


def _batch_response_error(response, originals, batch, header_pattern, matched):
    if len(matched) == len(batch):
        return None
    headers = list(header_pattern.finditer(response))
    if not headers:
        return "Output format error. No ORIGINAL_STRING found."
    unvalidated = _match_batch_response(response, originals, batch, header_pattern, None)
    if len(unvalidated) > len(matched):
        return ("Output format/content error. Invalid split contents. "
                f"{len(matched)}/{len(batch)} valid splits obtained.")
    if any(_echoed_original(header) not in {originals[index] for index in batch}
           for header in headers):
        return ("Output format error. Unknown ORIGINAL_STRING. "
                f"{len(matched)}/{len(batch)} splits obtained.")
    return f"Number of split mismatch. {len(matched)}/{len(batch)} splits obtained."


def _original_header_pattern(originals: Sequence[str]) -> re.Pattern[str]:
    # Longest-first matching preserves multiline originals and literal [ITEM] text.
    ordered = sorted(set(originals), key=len, reverse=True)
    alternatives = "|".join(re.escape(value) for value in ordered)
    bare_alternatives = "|".join(re.escape(value) for value in ordered if value) or r"(?!)"
    # An echo can be bare or prefixed, with its split on this line or the next.
    # Require a split boundary so an original is never matched inside a word.
    boundary = (r"(?:[ \t]*(?::[ \t]*)?(?:\r?\n|$)"
                r"|[^\w\[\]\r\n]*(?=\[)"
                r"|(?:[ \t]+|[ \t]*:[ \t]*)(?=\w+[ \t]*:))")
    start = r"(?:^[ \t]*|(?<=(?i:\[ITEM\]))[ \t]*)"
    return re.compile(
        rf"{start}ORIGINAL_STRING:[ \t]*(?P<original>{alternatives}){boundary}"
        rf"|{start}(?P<bare_original>{bare_alternatives}){boundary}(?=\s*\S)"
        rf"|{start}ORIGINAL_STRING:[^\n]*(?:\n|$)", re.MULTILINE,
    )


def _echoed_original(header: re.Match[str]) -> str | None:
    original = header.group("original")
    return original if original is not None else header.group("bare_original")


def _complete_split_body(body: str) -> str:
    """Keep the first result attached to a header, excluding orphan result blocks."""
    body = body.strip()
    if not body:
        return ""
    # Components may span contiguous lines. Blank lines or an item separator
    # end the result block; never attach a following orphan to it.
    body = re.sub(r"\s*\[ITEM\]\s*$", "", body, flags=re.IGNORECASE)
    return re.split(
        r"\r?\n[ \t]*\r?\n|^[ \t]*\[ITEM\][ \t]*(?:\r?\n|$)",
        body, maxsplit=1, flags=re.MULTILINE | re.IGNORECASE,
    )[0].strip()


def _match_batch_response(
    response: str,
    originals: Sequence[str],
    batch: Sequence[int],
    header_pattern: re.Pattern[str],
    item_is_valid: Callable[[str, str], bool] | None,
) -> dict[int, str]:
    indices = defaultdict(deque)
    for index in batch:
        indices[originals[index]].append(index)
    found = {}
    headers = list(header_pattern.finditer(response))
    for offset, header in enumerate(headers):
        original = _echoed_original(header)
        if original not in indices or not indices[original]:
            continue
        end = headers[offset + 1].start() if offset + 1 < len(headers) else len(response)
        body = _complete_split_body(response[header.end():end])
        if item_is_valid is None or item_is_valid(original, body):
            found[indices[original].popleft()] = body
    return found


def _single_response_items(response: str, original: str) -> list[str]:
    # Tolerate a correct unsolicited echo, but never require one for alignment.
    header = _original_header_pattern([original]).search(response)
    if header is not None and not response[:header.start()].strip():
        if _echoed_original(header) != original:
            return []
        response = response[header.end():]
    return RParser.split_items(response) if response.strip() else [""]


def _split_batch_item(original: str, body: str) -> str:
    """Attach the identity header without duplicating an unchanged input string."""

    return LLMPrompt.original_string_item(original=original, body=body)


def _request_split_items_with_retry(
    request: Callable[[str], str],
    originals: Sequence[str],
    bodies: Sequence[str],
    instructions: str,
    operation: str,
    item_is_valid: Callable[[str, str], bool] | None,
    item_fallback: Callable[[str, str], str | None] | None,
) -> tuple[str, list[str]]:
    """Use the legacy retry count while isolating each SLM input."""

    retries = Config.num_retry
    attempts = retries + 1
    found: dict[int, str] = {}
    fallbacks = {}
    pending = list(range(len(originals)))
    history = []
    last_response = ""
    last_query = ""
    for _attempt in range(attempts):
        unresolved = []
        for index in pending:
            last_query = bodies[index]
            last_response, last_prompt, reason = _request_split_response(
                lambda: request(last_query),
                format_prompt_for_diagnostics(last_query, instructions),
            )
            items = _single_response_items(last_response, originals[index]) if reason is None else []
            if reason is None:
                reason = _split_items_error(items, 1, [originals[index]], item_is_valid)
                if not items and last_response.lstrip().startswith("ORIGINAL_STRING:"):
                    reason = "Output format error. Unknown ORIGINAL_STRING."
            exchange = record_exchange(
                history, last_prompt, last_response, reason, operation,
            )
            if reason is None:
                found[index] = items[0]
            else:
                unresolved.append(index)
                received = {index: items[0]} if len(items) == 1 else {}
                _remember_fallbacks(fallbacks, received, originals, found, exchange, item_fallback)
        if not unresolved:
            items = [found[index] for index in range(len(originals))]
            return LLMPrompt.join_items(items), items
        pending = unresolved

    _accept_fallbacks(found, fallbacks, originals)
    _fill_empty_results(found, originals, operation)
    items = [found[index] for index in range(len(originals))]
    return LLMPrompt.join_items(items), items


class LLMUtil:
    """Marker base retained for compatibility with earlier experiments."""


class OpenAIUtil(LLMUtil):
    MODEL_ALIASES = OPENAI_MODEL_ALIASES
    get_openai_model = staticmethod(resolve_openai_model)

    @staticmethod
    def get_client(key_file: str | Path | None = None):
        import openai

        return openai.OpenAI(api_key=_read_api_key(key_file or LLMInfo.api_key))

    @staticmethod
    def create_response(
        input_text: str,
        *,
        instructions: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_retry: int | None = None,
    ) -> str:
        client = OpenAIUtil.get_client()
        if _budget_request.get() or pricing_active():
            client = client.with_options(max_retries=0)
        if original_string_headers():
            instructions = LLMPrompt.with_original_string_response_format(instructions)
        record_prompt(format_prompt_for_diagnostics(input_text, instructions))
        request: dict[str, Any] = {
            "model": resolve_openai_model(model or LLMInfo.DEFAULT_MODEL),
            "instructions": instructions,
            "input": input_text,
        }
        if max_tokens is not None:
            request["max_output_tokens"] = max_tokens
        if temperature is not None and str(request["model"]).startswith("gpt-4.1"):
            request["temperature"] = temperature
        def send():
            with track_request(request["model"], instructions, input_text) as usage:
                response = client.responses.create(**request)
                if usage is not None:
                    usage.record_output(response.output_text)
                return response.output_text.strip()

        return _retry(send, num_retry)

    @staticmethod
    def get_semantic_components(
        query: str,
        file: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        use_type: bool = False,
        type_list: List[str] | None = None,
    ) -> str:
        del file, seed
        return OpenAIUtil.create_response(
            query,
            instructions=_gpt_sem_component_instruction(
                use_type,
                type_list,
                original_strings=False,
            ),
            model=model or Config.extraction_model(),
            temperature=temperature,
            max_tokens=max_tokens,
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
            OpenAIUtil.get_semantic_components(
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
    ) -> str:
        del seed
        return OpenAIUtil.create_response(
            query_components,
            instructions=LLMPrompt.gpt_sem_refinement(
                semantic_types=sem_types_str,
                original_strings=False,
            ),
            model=model or Config.extraction_model(),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @staticmethod
    def get_final_split(
        query: str,
        sem_types: Sequence[str],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        del seed
        return OpenAIUtil.create_response(
            query,
            instructions=LLMPrompt.gpt_sem_final_component(
                semantic_types=", ".join(sem_types),
                original_strings=False,
            ),
            model=model or Config.extraction_model(),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @staticmethod
    @perfect_split_override()
    @cache_gpt_splits("values")
    @cache_result()
    def get_split(
        query_dict: Dict[int, str],
        file: str,
        input_idx: List[int],
        output_idx: List[int],
        prompt: str = "label+string",
        always_split_further: bool = False,
        use_type: bool = False,
        type_list: List[str] | None = None,
    ) -> Dict[int, Dict[str, str]]:
        return get_value_split(
            OpenAIUtil,
            query_dict,
            file,
            input_idx,
            output_idx,
            prompt,
            always_split_further,
            use_type,
            type_list,
        )


class NovitaUtil(LLMUtil):
    BASE_URL = "https://api.novita.ai/openai"

    @staticmethod
    def get_client(key_file: str | Path | None = None):
        import openai

        return openai.OpenAI(
            api_key=_read_api_key(key_file or LLMInfo.novita_api_key),
            base_url=NovitaUtil.BASE_URL,
        )

    @staticmethod
    def get_novita_model(modelname: str) -> str:
        return {
            "deepseek": "deepseek/deepseek-v3.2",
            "glm": "zai-org/glm-4.7",
        }.get(modelname, modelname)

    @staticmethod
    def create_response(
        input_text: str,
        *,
        instructions: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_retry: int | None = None,
    ) -> str:
        client = NovitaUtil.get_client()
        if _budget_request.get() or pricing_active():
            client = client.with_options(max_retries=0)
        if original_string_headers():
            instructions = LLMPrompt.with_original_string_response_format(instructions)
        record_prompt(format_prompt_for_diagnostics(input_text, instructions))
        request: dict[str, Any] = {
            "model": NovitaUtil.get_novita_model(model or "deepseek"),
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": input_text},
            ],
            "temperature": LLMInfo.DEFAULT_TEMPERATURE if temperature is None else temperature,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        def send():
            with track_request(request["model"], instructions, input_text) as usage:
                response = client.chat.completions.create(**request)
                text = response.choices[0].message.content or ""
                if usage is not None:
                    usage.record_output(text)
                return text.strip()

        return _retry(send, num_retry)

    @staticmethod
    def get_semantic_components(
        query: str,
        file: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        use_type: bool = False,
        type_list: List[str] | None = None,
    ) -> str:
        del file, seed
        return NovitaUtil.create_response(
            query,
            instructions=_gpt_sem_component_instruction(
                use_type,
                type_list,
                original_strings=False,
            ),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
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
            NovitaUtil.get_semantic_components(
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
    ) -> str:
        del seed
        return NovitaUtil.create_response(
            query_components,
            instructions=LLMPrompt.gpt_sem_refinement(
                semantic_types=sem_types_str,
                original_strings=False,
            ),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @staticmethod
    def get_final_split(
        query: str,
        sem_types: Sequence[str],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        del seed
        return NovitaUtil.create_response(
            query,
            instructions=LLMPrompt.gpt_sem_final_component(
                semantic_types=", ".join(sem_types),
                original_strings=False,
            ),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @staticmethod
    @perfect_split_override()
    @cache_result()
    def get_split(
        query_dict: Dict[int, str],
        file: str,
        input_idx: List[int],
        output_idx: List[int],
        prompt: str = "label+string",
        always_split_further: bool = False,
        use_type: bool = False,
        type_list: List[str] | None = None,
    ) -> Dict[int, Dict[str, str]]:
        return get_value_split(
            NovitaUtil,
            query_dict,
            file,
            input_idx,
            output_idx,
            prompt,
            always_split_further,
            use_type,
            type_list,
        )


def get_value_split(
    provider: type[OpenAIUtil] | type[NovitaUtil],
    query_dict: Dict[int, str],
    file: str,
    input_idx: List[int],
    output_idx: List[int],
    prompt: str,
    always_split_further: bool,
    use_type: bool,
    type_list: List[str] | None,
) -> Dict[int, Dict[str, str]]:
    if not query_dict:
        return {}
    LLMInfo.set_default_prompt(prompt)
    query_ids = list(query_dict)
    originals = list(query_dict.values())
    budget = (
        SplitQueryBudget.for_strings(len(originals))
        if Config.query_individual_factor is not None else None
    )
    use_supplied_types = use_type and bool(type_list)
    slm = _is_slm(provider)
    one_at_a_time = slm and not Config.language_model_batch
    initial, initial_items = request_split_batch(
        lambda query: provider.get_semantic_components(
            query,
            file,
            use_type=use_supplied_types,
            type_list=type_list,
        ),
        originals,
        instructions=(
            LLMPrompt.slm_sem_component_with_types(
                input_text="", semantic_types=RParser.parse_sem_types(type_list),
                original_strings=budget is not None and not one_at_a_time,
                batched=Config.language_model_batch,
            ) if slm and use_supplied_types else
            LLMPrompt.slm_sem_component(
                input_text="", original_strings=budget is not None and not one_at_a_time,
                batched=Config.language_model_batch,
            ) if slm else
            _gpt_sem_component_instruction(
                use_supplied_types, type_list, original_strings=budget is not None,
            )
        ),
        budget=budget,
        one_at_a_time=one_at_a_time,
        item_is_valid=_exact_component_item_is_valid(type_list if use_supplied_types else None),
        item_fallback=_exact_component_item_fallback(type_list if use_supplied_types else None),
        operation=(
            "semantic component extraction"
            if use_supplied_types
            else "initial semantic split"
        ),
    )
    if use_supplied_types or not any(initial_items):
        return _parse_value_split(query_ids, query_dict, initial_items)

    parsed_initial = RParser.parse_component_str(initial_items)
    local_by_id = {value_id: parsed_initial[index] for index, value_id in enumerate(query_ids)}
    inputs = [local_by_id[value_id] for value_id in query_ids if value_id in input_idx]
    outputs = [local_by_id[value_id] for value_id in query_ids if value_id in output_idx]
    semantic_types = RParser.get_types_from_split(initial)
    previous_items = initial_items
    if always_split_further or RParser.require_further_split(
        dict(enumerate(inputs)), dict(enumerate(outputs))
    ):
        semantic_types_text = RParser.parse_sem_types(semantic_types)
        refined, _refined_items = request_split_batch(
            lambda query: provider.get_further_split(query, semantic_types_text),
            originals,
            bodies=[further_split_query({0: parsed_initial[index]}) for index in parsed_initial],
            instructions=(
                LLMPrompt.slm_sem_refinement(
                    input_text="", semantic_types=semantic_types_text,
                    original_strings=budget is not None and not one_at_a_time,
                    batched=Config.language_model_batch,
                ) if slm else LLMPrompt.gpt_sem_refinement(
                    semantic_types=semantic_types_text, original_strings=budget is not None,
                )
            ),
            budget=budget,
            one_at_a_time=one_at_a_time,
            operation="refined semantic split",
            item_is_valid=_exact_component_item_is_valid(None),
            item_fallback=_exact_component_item_fallback(None),
        )
        semantic_types = RParser.get_types_from_split(refined)
        previous_items = _refined_items
    final, final_items = request_split_batch(
        lambda query: provider.get_final_split(query, semantic_types),
        originals,
        instructions=(
            LLMPrompt.slm_sem_final_component(
                input_text="", semantic_types=", ".join(semantic_types),
                original_strings=budget is not None and not one_at_a_time,
                batched=Config.language_model_batch,
            ) if slm else LLMPrompt.gpt_sem_final_component(
                semantic_types=", ".join(semantic_types), original_strings=budget is not None,
            )
        ),
        operation="final semantic split",
        one_at_a_time=one_at_a_time,
        budget=budget,
        item_is_valid=_exact_component_item_is_valid(semantic_types or None),
        item_fallback=_exact_component_item_fallback(semantic_types or None),
    )
    # Keep previously valid evidence when a later refinement/final request fails.
    final_items = [final or previous or initial for final, previous, initial
                   in zip(final_items, previous_items, initial_items)]
    return _parse_value_split(query_ids, query_dict, final_items)


def _parse_value_split(
    query_ids: Sequence[int],
    query_dict: Mapping[int, str],
    items: List[str],
) -> Dict[int, Dict[str, str]]:
    parsed = RParser.parse_component_str(items, escapespecchar=False)
    result: Dict[int, Dict[str, str]] = {}
    for index, value_id in enumerate(query_ids):
        result[value_id] = {}
        for raw_key, raw_value in parsed[index].items():
            key = re.sub(r"[^a-zA-Z0-9]", " ", raw_key).lower()
            key = re.sub(r"\s+", " ", key).strip()
            result[value_id][key] = RParser.remove_string_wrapper(
                raw_value, query_dict[value_id]
            )
    return result


def _is_slm(provider: type[LLMUtil]) -> bool:
    """OpenAI requests always use LLM batching, independently of the selected SLM."""

    if provider is OpenAIUtil:
        return False
    model = getattr(provider, "__MODEL__", Config.model)
    return not is_gpt_model(str(model))


def _exact_component_item_is_valid(
    semantic_types: Sequence[str] | None,
) -> Callable[[str, str], bool]:
    """Validate supplied labels and literal substrings, without semantic checks."""

    allowed_types = None if semantic_types is None else {
        RParser.key_normalize_with_underscore(_without_occurrence(value))
        for value in semantic_types
    }

    def validate(original: str, item: str) -> bool:
        if not item.strip():
            return True
        for line in item.splitlines():
            key, separator, value = line.partition(":")
            value = RParser.remove_string_wrapper(value.strip(), original)
            normalized_key = RParser.key_normalize_with_underscore(
                _without_occurrence(key)
            )
            if (
                not separator
                or not normalized_key
                or not value
                or value not in original
                or (allowed_types is not None and normalized_key not in allowed_types)
            ):
                return False
        return True

    return validate


def _exact_component_item_fallback(
    semantic_types: Sequence[str] | None,
) -> Callable[[str, str], str | None]:
    """Drop only empty values and standalone lines after normal retries fail."""
    validate = _exact_component_item_is_valid(semantic_types)

    def clean(original: str, item: str) -> str | None:
        kept = []
        discarded = False
        for line in item.splitlines():
            _key, separator, value = line.partition(":")
            value = RParser.remove_string_wrapper(value.strip(), original)
            if not separator or not value:
                discarded = True
            else:
                kept.append(line)
        cleaned = "\n".join(kept)
        return cleaned if discarded and validate(original, cleaned) else None

    return clean


def _without_occurrence(value: str) -> str:
    return re.sub(r"\s+\d+\s*$", "", value).strip()


def further_split_query(components: Dict[Any, Any]) -> str:
    items = []
    for value in components.values():
        if isinstance(value, dict):
            lines = [f"{key}: {item}" for key, item in value.items() if item]
        else:
            lines = [str(item) for item in value if item]
        if lines:
            items.append("\n".join(lines))
    return LLMPrompt.join_items(items)


def expect_item_count(response: str, count: int, operation: str) -> list[str]:
    items = RParser.split_items(response)
    if len(items) != count:
        raise ValueError(f"{operation} returned {len(items)} items; expected {count}")
    return items


class SplitCountError(ValueError):
    """Raised after malformed split responses exhaust their retry allowance."""


def request_split_with_retry(
    request: Callable[[], str],
    *,
    prompt: str,
    expected_count: int,
    operation: str,
    num_retry: int | None = None,
    originals: Sequence[str] | None = None,
    item_is_valid: Callable[[str, str], bool] | None = None,
    item_fallback: Callable[[str, str], str | None] | None = None,
) -> tuple[str, list[str]]:
    """Retry LLM responses whose ``[ITEM]`` count does not match the prompt."""

    retries = Config.num_retry if num_retry is None else num_retry
    if retries < 0:
        raise ValueError("num_retry must be non-negative")
    attempts = retries + 1
    history = []
    last_response = ""
    found = {}
    fallbacks = {}
    aligned_originals = list(originals) if originals is not None else [""] * expected_count
    header_pattern = _original_header_pattern(originals) if originals else None
    batch = list(range(expected_count))
    for _attempt in range(attempts):
        last_response, last_prompt, reason = _request_split_response(request, prompt)
        items = RParser.split_items(last_response)
        request_succeeded = reason is None
        matched = {}
        received = {}
        echoed = header_pattern is not None and header_pattern.search(last_response) is not None
        if request_succeeded:
            if echoed:
                matched = _match_batch_response(
                    last_response, aligned_originals, batch, header_pattern, item_is_valid,
                )
                reason = _batch_response_error(
                    last_response, aligned_originals, batch, header_pattern, matched,
                )
                received = _match_batch_response(
                    last_response, aligned_originals, batch, header_pattern, None,
                )
            else:
                reason = _split_items_error(items, expected_count, originals, item_is_valid)
                # Without echoes, only a complete batch can be aligned safely.
                if len(items) == expected_count:
                    received = dict(enumerate(items))
                    matched = {
                        index: item for index, item in received.items()
                        if item_is_valid is None or item_is_valid(aligned_originals[index], item)
                    }
        exchange = record_exchange(history, last_prompt, last_response, reason, operation)
        if reason is None:
            if echoed:
                items = [matched[index] for index in batch]
                return LLMPrompt.join_items(items), items
            return last_response, items
        found.update(matched)
        _remember_fallbacks(fallbacks, received, aligned_originals, found, exchange, item_fallback)
        if len(found) == expected_count:
            break

    _accept_fallbacks(found, fallbacks, aligned_originals)
    _fill_empty_results(found, aligned_originals, operation)
    items = [found[index] for index in range(expected_count)]
    return LLMPrompt.join_items(items), items


@exclude_output
def print_split_count_error(
    error: SplitCountError,
    prompt: str,
    response: str,
) -> None:
    separator = "!" * 96
    print(separator)
    print(f"!!! LLM SPLIT COUNT ERROR: {error} !!!")
    print("--- PROMPT START ---")
    print(prompt or "<empty prompt>")
    print("--- PROMPT END ---")
    print("--- RESPONSE START ---")
    print(response or "<empty response>")
    print("--- RESPONSE END ---")
    print(separator)


@exclude_output
def format_prompt_for_diagnostics(input_text: str, instructions: str) -> str:
    return LLMPrompt.diagnostic(instructions=instructions, input_text=input_text)


def _gpt_sem_component_instruction(
    use_type: bool,
    type_list: List[str] | None,
    *,
    original_strings: bool,
) -> str:
    if use_type and type_list:
        return LLMPrompt.gpt_sem_component_with_types(
            semantic_types=RParser.parse_sem_types(type_list),
            original_strings=original_strings,
        )
    return LLMPrompt.gpt_sem_component(original_strings=original_strings)


def _retry(operation, num_retry: int | None):
    attempts = 1 if _budget_request.get() else max(
        1, num_retry if num_retry is not None else Config.num_retry + 1
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:  # Provider SDKs expose several exception classes.
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 5))
    assert last_error is not None
    raise last_error


def _read_api_key(path: str | Path) -> str:
    key_path = Path(path)
    if not key_path.is_file():
        raise FileNotFoundError(
            f"API key file not found: {key_path}. Store the plaintext key in that file."
        )
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {key_path}")
    return key
