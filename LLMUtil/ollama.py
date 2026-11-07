from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import random
import re
import sys
from typing import Any, List, Sequence

from LLMUtil.LLMInfo import LLMInfo
from LLMUtil.LLMPrompt import LLMPrompt
from LLMUtil.providers import LLMUtil, get_value_split, original_string_headers, request_split_batch
from LLMUtil.ResponseParser import RParser, RParser_Prompt
from Utils.CacheUtil import cache_result, perfect_split_override, perfect_split_override_prompt
from Utils.config import Config
from gdt.diagnostics import record_prompt
from gdt.pricing import track_request


_dataset_models: ContextVar[list[str] | None] = ContextVar("ollama_dataset_models", default=None)


class OLLAMAUtil(LLMUtil):
    """Use Ollama's default idle lifetime and explicitly unload after each dataset."""

    __BASE_URL__ = "http://localhost:11434"
    __MODE__ = "gen"
    __MODEL__ = "llama3.2-3b"

    STOP_WORDS = {
        "llama": ["<|end_of_text|>", "<|eot_id|>"],
        "qwen": ["<|im_end|>"],
        "granite": ["<|end_of_text|>"],
    }
    MODEL_ALIASES = {
        "qwen3-4b": "qwen3-4b",
        "llama3.2-3b": "llama3.2-3b-custom",
        "granite-1b": "granite-1b-custom",
        "granite-4b": "granite-4b-custom",
        "granite-micro": "granite-4b-custom",
        "phi4": "phi4",
        "qwen3.5-4b": "qwen3.5:4b",
        "granite4.2-3b": "granite4.2-3b",
        "ministral3-3b": "ministral3-3b",
    }

    @classmethod
    def set_model(cls, model: str) -> None:
        cls.__MODEL__ = model

    @classmethod
    def set_mode(cls, mode: str) -> None:
        if mode not in {"gen", "generate", "chat"}:
            raise ValueError(f"unknown Ollama mode: {mode}")
        cls.__MODE__ = mode

    @classmethod
    def make_url(cls) -> str:
        endpoint = "generate" if cls.__MODE__ in {"gen", "generate"} else "chat"
        return f"{cls.__BASE_URL__}/api/{endpoint}"

    @classmethod
    def get_stop_words(cls) -> List[str]:
        for prefix, values in cls.STOP_WORDS.items():
            if cls.__MODEL__.startswith(prefix):
                return values
        return ["<|end_of_text|>"]

    @classmethod
    def get_ollama_model_from_str(cls) -> str:
        for prefix, alias in cls.MODEL_ALIASES.items():
            if cls.__MODEL__.startswith(prefix):
                return alias
        return cls.__MODEL__

    @classmethod
    def load_model(cls) -> None:
        cls._model_lifecycle_request(unload=False)

    lode_model = load_model  # Historical typo.

    @classmethod
    def unload_model(cls, *, model: str | None = None) -> None:
        """Unload the current model, or an explicitly resolved Ollama model name."""
        cls._model_lifecycle_request(unload=True, model=model)

    @classmethod
    @contextmanager
    def dataset_session(cls):
        """Offload models used by this dataset, including failed requests.

        Enter this context before the runtime timer and dataset deadline so both
        have ended before cleanup begins. Cleanup failures must not hide results
        or the original dataset failure.
        """
        used_models: list[str] = []
        token = _dataset_models.set(used_models)
        try:
            yield
        finally:
            _dataset_models.reset(token)
            for model in used_models:
                try:
                    cls.unload_model(model=model)
                except Exception as error:
                    print(
                        f"WARNING: could not unload Ollama model {model}: "
                        f"{type(error).__name__}: {error}",
                        file=sys.stderr,
                    )

    @staticmethod
    def _record_model(model: str) -> None:
        used_models = _dataset_models.get()
        if used_models is not None and model not in used_models:
            used_models.append(model)

    @classmethod
    def _model_lifecycle_request(cls, *, unload: bool, model: str | None = None) -> None:
        import requests

        model = cls.get_ollama_model_from_str() if model is None else model
        # Keep-alive is omitted for loading/inference; only unloading overrides it.
        payload: dict[str, Any] = {"model": model, "stream": False, "think": False}
        if unload:
            payload["keep_alive"] = 0
        else:
            cls._record_model(model)
        # Lifecycle requests use generate even when predictions use chat.
        response = requests.post(
            f"{cls.__BASE_URL__}/api/generate",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

    @classmethod
    def get_response_batch(
        cls,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.05,
    ) -> List[str]:
        import requests

        responses = []
        for prompt in prompts:
            payload: dict[str, Any] = {
                "model": cls.get_ollama_model_from_str(),
                "stream": False,
                "think": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "stop": cls.get_stop_words(),
                    "min_p": 0.1,
                    "top_k": 1,
                    "top_p": 0.05,
                },
            }
            if cls.__MODE__ in {"gen", "generate"}:
                payload["prompt"] = prompt
            else:
                payload["messages"] = [{"role": "user", "content": prompt}]
            cls._record_model(payload["model"])
            with track_request(payload["model"], prompt) as usage:
                response = requests.post(cls.make_url(), json=payload, timeout=300)
                response.raise_for_status()
                body = response.json()
                # Only final answer text is parsed and counted; thinking is disabled.
                text = (
                    body.get("response", "")
                    if cls.__MODE__ in {"gen", "generate"}
                    else body.get("message", {}).get("content", "")
                )
                if usage is not None:
                    usage.record_output(text)
            responses.append(text.strip())
        return responses

    @classmethod
    def _split_response(cls, prompt: str) -> str:
        record_prompt(prompt)
        return cls.get_response_batch(
            [prompt],
            max_tokens=LLMInfo.DEFAULT_MAX_TOKENS, temperature=0.0,
        )[0]

    @classmethod
    def get_semantic_components(
        cls, query: str, file: str | None = None, *,
        use_type: bool = False, type_list: List[str] | None = None,
    ) -> str:
        if use_type and type_list:
            prompt = LLMPrompt.slm_sem_component_with_types(
                input_text=query,
                semantic_types=RParser.parse_sem_types(type_list),
                original_strings=original_string_headers(),
                batched=Config.language_model_batch,
            )
        else:
            prompt = LLMPrompt.slm_sem_component(
                input_text=query,
                original_strings=original_string_headers(),
                batched=Config.language_model_batch,
            )
        return cls._split_response(prompt)

    @classmethod
    def get_further_split(cls, query: str, sem_types_str: str) -> str:
        return cls._split_response(
            LLMPrompt.slm_sem_refinement(
                input_text=query,
                semantic_types=sem_types_str,
                original_strings=original_string_headers(),
                batched=Config.language_model_batch,
            )
        )

    @classmethod
    def get_final_split(cls, query: str, sem_types: Sequence[str]) -> str:
        return cls._split_response(
            LLMPrompt.slm_sem_final_component(
                input_text=query,
                semantic_types=", ".join(sem_types),
                original_strings=original_string_headers(),
                batched=Config.language_model_batch,
            )
        )

    @classmethod
    @perfect_split_override_prompt()
    def get_labels_for_types(
        cls, query_dict: dict[int, str], file: str, type_list: List[str],
        examples: Sequence[tuple[str, str]],
    ) -> dict[int, List[str]]:
        """Use the most-labeled example and a fresh random companion per request."""
        del file

        anchor = max(
            range(len(examples)),
            key=lambda index: len(re.findall(r"\[[^\]]+\]", examples[index][1])),
            default=None,
        )
        remaining_examples = [example for index, example in enumerate(examples) if index != anchor]

        def request(query: str) -> str:
            selected = [examples[anchor]] if anchor is not None else []
            if remaining_examples:
                selected.append(random.choice(remaining_examples))
            prompt = LLMPrompt.slm_cluster_label(
                input_text=query,
                semantic_types=", ".join(type_list),
                examples=selected,
                original_strings=original_string_headers(),
                batched=Config.language_model_batch,
            )
            record_prompt(prompt)
            return cls._split_response(prompt)

        _, items = request_split_batch(
            request,
            list(query_dict.values()),
            instructions="",
            one_at_a_time=not Config.language_model_batch,
            item_is_valid=RParser_Prompt.is_composition,
            operation="source semantic labels",
        )
        parsed = RParser_Prompt.parse_component_str(items)
        return {value_id: parsed[index] for index, value_id in enumerate(query_dict)}

    @staticmethod
    @perfect_split_override()
    @cache_result()
    def get_split(
        query_dict: dict[int, str], file: str, input_idx: List[int], output_idx: List[int],
        prompt: str = "label+string", always_split_further: bool = False,
        use_type: bool = False, type_list: List[str] | None = None,
    ) -> dict[int, dict[str, str]]:
        return get_value_split(
            OLLAMAUtil, query_dict, file, input_idx, output_idx, prompt,
            always_split_further, use_type, type_list,
        )
