from __future__ import annotations

from copy import deepcopy
from functools import wraps
from inspect import signature
from pathlib import Path
from gdt.diagnostics import lm_failure_count
from gdt.models import resolve_openai_model


class BenchmarkCache:
    OpenAICache = {}
    StringCache = {}
    PairCache = {}

    @classmethod
    def clear(cls) -> None:
        cls.OpenAICache.clear()
        cls.StringCache.clear()
        cls.PairCache.clear()


def cache_gpt_splits(kind: str, *, paired: bool = False):
    """Reuse parsed splits; a cached type superset can serve a smaller vocabulary."""
    def decorate(function):
        parameters = signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            from gdt.config import Config
            from LLMUtil.ResponseParser import RParser

            if not Config.money_saver:
                return function(*args, **kwargs)
            bound = parameters.bind(*args, **kwargs)
            bound.apply_defaults()
            query = bound.arguments["query_dict"]
            model = Config.llm_model if kind.startswith("labels") else Config.extraction_model()
            model = resolve_openai_model(model)
            if model == "gpt":
                model = "gpt-5.1"
            if not kind.startswith("labels") and not model.lower().startswith("gpt-5.1"):
                return function(*args, **kwargs)
            vocabulary = bound.arguments.get("type_list")
            if kind == "values" and not bound.arguments.get("use_type"):
                vocabulary = None

            def normalize(value):
                return RParser.key_normalize(str(value).strip("[]"))

            wanted = frozenset(map(normalize, vocabulary)) if vocabulary else None
            scope = (str((Path(Config.data_directory) / bound.arguments["file"]).resolve()),
                     Config.num_examples,
                     model, "labels" if kind.startswith("labels") else kind,
                     bound.arguments.get("always_split_further", False),
                     Config.query_individual_factor is not None)
            cache = BenchmarkCache.StringCache.setdefault(scope, {})
            if paired:
                return _cached_paired_labels(function, bound, scope, cache)
            result = {}
            missing = {}
            for value_id, original in query.items():
                for known_types, value in cache.get(original, []):
                    if known_types == wanted or (
                        wanted is not None and known_types is not None and wanted <= known_types
                    ) or (
                        kind.startswith("labels") and known_types is None and wanted is not None
                    ):
                        value = deepcopy(value)
                        if wanted is not None:
                            if isinstance(value, dict):
                                value = {
                                    key: item for key, item in value.items()
                                    if normalize(key) in wanted
                                }
                            else:
                                value = [item for item in value if normalize(item) in wanted]
                        result[value_id] = value
                        break
                else:
                    missing[value_id] = original
            if missing:
                bound.arguments["query_dict"] = missing
                for name in ("input_idx", "output_idx"):
                    if name in bound.arguments:
                        bound.arguments[name] = [i for i in bound.arguments[name] if i in missing]
                failures_before = lm_failure_count()
                obtained = function(*bound.args, **bound.kwargs)
                clean_request = lm_failure_count() == failures_before
                for value_id, value in obtained.items():
                    if value_id in missing:
                        if clean_request and value:
                            cache.setdefault(missing[value_id], []).append((wanted, deepcopy(value)))
                        result[value_id] = value
            return {value_id: result[value_id] for value_id in query if value_id in result}

        return wrapped
    return decorate


def _cached_paired_labels(function, bound, scope, string_cache):
    query = bound.arguments["query_dict"]
    pairs = list(zip(bound.arguments["input_idx"], bound.arguments["output_idx"], strict=True))
    ids = [value_id for pair in pairs for value_id in pair]
    if len(ids) != len(query) or set(ids) != set(query):
        raise ValueError("example compositions require complete input/output pairs")
    cache = BenchmarkCache.PairCache.setdefault(scope, {})
    missing_pairs = []
    result = {}
    for left, right in pairs:
        key = (query[left], query[right])
        if key in cache:
            result[left], result[right] = deepcopy(cache[key])
        else:
            missing_pairs.append((left, right))
    if missing_pairs:
        bound.arguments["query_dict"] = {
            value_id: query[value_id] for pair in missing_pairs for value_id in pair
        }
        bound.arguments["input_idx"] = [left for left, _right in missing_pairs]
        bound.arguments["output_idx"] = [right for _left, right in missing_pairs]
        failures_before = lm_failure_count()
        result.update(function(*bound.args, **bound.kwargs))
        if lm_failure_count() == failures_before:
            for left, right in missing_pairs:
                if result.get(left) and result.get(right):
                    cache[(query[left], query[right])] = deepcopy((result[left], result[right]))

    # Only unambiguous per-string compositions may serve later source labeling.
    by_string = {}
    ambiguous = set()
    for pair, compositions in cache.items():
        for original, composition in zip(pair, compositions):
            if original in by_string and by_string[original] != composition:
                ambiguous.add(original)
            by_string[original] = composition
    for original, composition in by_string.items():
        if original in ambiguous:
            string_cache.pop(original, None)
        else:
            string_cache[original] = [(None, deepcopy(composition))]
    return {value_id: result[value_id] for value_id in query}
