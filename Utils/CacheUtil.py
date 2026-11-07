from __future__ import annotations

from gdt.timing import output_time
from gdt.diagnostics import lm_failure_count

import functools
import hashlib
from inspect import signature
import os
from pathlib import Path
import pickle
from typing import Any, Callable, TypeVar, cast

from Utils.config import Config
from gdt.ground_truth import provide_clusters, provide_operators, provide_splits


F = TypeVar("F", bound=Callable[..., Any])


def cache_result(
    cache_dir: str | Path | None = None, *, restore_features: bool = False,
    version: str | None = None,
) -> Callable[[F], F]:
    """Cache results and, when requested, the clustering feature registry."""

    def decorator(function: F) -> F:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not Config.use_cache and not Config.save_cache:
                return function(*args, **kwargs)
            root = Path(cache_dir if cache_dir is not None else Config.cache_dir)
            cache_path = root / _cache_filename(function, args, kwargs, version=version)
            if Config.use_cache and cache_path.is_file():
                with cache_path.open("rb") as handle:
                    result = pickle.load(handle)
                if restore_features:
                    from DataAccess.datafeature import DataFeature

                    result, features = result
                    DataFeature.clear()
                    DataFeature.data_features.update(features)
                if Config.verbose:
                    with output_time():
                        print(f"Loaded cache: {cache_path}")
                return result

            result = function(*args, **kwargs)
            should_save = Config.save_cache and not any(
                (Config.perfect_cluster, Config.perfect_split, Config.perfect_operator)
            ) and lm_failure_count() == 0
            if should_save:
                with output_time():
                    root.mkdir(parents=True, exist_ok=True)
                    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
                    cached_result = result
                    if restore_features:
                        from DataAccess.datafeature import DataFeature

                        cached_result = (result, DataFeature.data_features)
                    with temporary.open("wb") as handle:
                        pickle.dump(cached_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(temporary, cache_path)
                    if Config.verbose:
                        print(f"Saved cache: {cache_path}")
            return result

        return cast(F, wrapper)

    return decorator


def perfect_cluster_override() -> Callable[[F], F]:
    def decorator(function: F) -> F:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if Config.perfect_cluster and not Config.use_cache:
                file = _argument(args, kwargs, 0, "file")
                return provide_clusters(str(file))
            return function(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def perfect_split_override() -> Callable[[F], F]:
    return _perfect_split_decorator(prompt_only=False)


def perfect_split_override_prompt() -> Callable[[F], F]:
    return _perfect_split_decorator(prompt_only=True)


def _perfect_split_decorator(*, prompt_only: bool) -> Callable[[F], F]:
    def decorator(function: F) -> F:
        parameters = signature(function)

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if Config.perfect_split and not Config.use_cache:
                bound = parameters.bind(*args, **kwargs)
                query = bound.arguments["query_dict"]
                file = bound.arguments["file"]
                return provide_splits(str(file), query, prompt_only=prompt_only)
            return function(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def perfect_operator_override() -> Callable[[F], F]:
    def decorator(function: F) -> F:
        parameters = signature(function)

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if Config.perfect_operator and not Config.use_cache:
                bound = parameters.bind(*args, **kwargs)
                return provide_operators(str(bound.arguments["file"]))
            return function(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def clear_cache(cache_dir: str | Path = "cache") -> None:
    root = Path(cache_dir).resolve()
    if not root.is_dir():
        return
    for path in root.iterdir():
        if path.is_file() and path.suffix == ".pkl":
            path.unlink()


def clear_specific_cache(pattern: str, cache_dir: str | Path = "cache") -> None:
    root = Path(cache_dir).resolve()
    if not root.is_dir():
        return
    for path in root.iterdir():
        if path.is_file() and path.suffix == ".pkl" and pattern in path.name:
            path.unlink()


def _cache_filename(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    version: str | None = None,
) -> str:
    context = (
        "reserved-five-paired-compositions-v2",
        Config.data_directory,
        Config.num_examples,
        Config.model,
        Config.llm_model,
        Config.mode,
        Config.language_model_batch,
        Config.slm_split,
    )
    if version is not None:
        context += (version,)
    try:
        payload = pickle.dumps((context, args, sorted(kwargs.items())), protocol=4)
    except (pickle.PicklingError, TypeError, AttributeError):
        payload = repr((context, args, sorted(kwargs.items()))).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:24]
    qualified = f"{function.__module__}.{function.__qualname__}".replace("<", "_").replace(">", "_")
    return f"{qualified}-{digest}.pkl"


def _argument(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    index: int,
    name: str,
) -> Any:
    if len(args) > index:
        return args[index]
    if name in kwargs:
        return kwargs[name]
    raise TypeError(f"missing argument: {name}")
