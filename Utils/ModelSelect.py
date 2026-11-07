from __future__ import annotations

from functools import lru_cache
from typing import Any


class ModelSelect:
    SENTENCE_MODELS = {
        "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "all-minilm-l12-v2": "sentence-transformers/all-MiniLM-L12-v2",
        "all-minilm": "sentence-transformers/all-MiniLM-L12-v2",
        "qwen": "Qwen/Qwen3-Embedding-0.6B",
        "local-qwen-emb": "model/Qwen3-0.6B-Embedding-finetuned",
        "local-qwen-emb-peft": "model/Qwen3-0.6B-Embedding-finetuned-peft",
        "local-qwen-emb-prompt": "model/Qwen3-0.6B-Embedding-finetuned-peft-prompt",
    }
    CAUSAL_MODELS = {
        "local-qwen-prompt": "model/Qwen3-0.6B-finetuned-peft-prompt",
        "local-qwen-prompt-labelonly": "model/Qwen3-0.6B-finetuned-peft-prompt-labelonly",
        "local-qwen-1.7b-prompt-labelonly": "model/Qwen3-1.7B-finetuned-peft-prompt-labelonly",
        "local-qwen-4b-prompt-labelonly": "model/Qwen3-4B-finetuned-peft-prompt-labelonly",
    }

    @staticmethod
    @lru_cache(maxsize=None)
    def get_clustering_model_by_name(name: str = "all-MiniLM-L6-v2") -> Any:
        normalized = name.lower()
        if normalized in ModelSelect.CAUSAL_MODELS:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_pretrained(
                ModelSelect.CAUSAL_MODELS[normalized],
                torch_dtype="auto",
                device_map="auto",
            )
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(ModelSelect.SENTENCE_MODELS.get(normalized, name))

    @staticmethod
    @lru_cache(maxsize=None)
    def get_tokenizer_by_name(name: str = "local-qwen-prompt-labelonly") -> Any:
        normalized = name.lower()
        model_path = ModelSelect.CAUSAL_MODELS.get(normalized)
        if model_path is None:
            raise ValueError(f"tokenizer for model {name!r} is not registered")
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_path)
