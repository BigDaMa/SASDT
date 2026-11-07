from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple


class ModelUtil:
    @staticmethod
    def build_qa_pair(questions: Sequence[str], answers: Sequence[str]) -> str:
        return "".join(
            f"Q: {question} A: {answer}\n"
            for question, answer in zip(questions, answers)
        )

    @staticmethod
    def get_response(model, tokenizer, prompt: str, max_length: int = 512) -> str:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated = model.generate(**model_inputs, max_new_tokens=max_length)
        output_ids = generated[0][len(model_inputs.input_ids[0]) :].tolist()
        # Qwen uses 151668 as </think>; return only the answer after it when present.
        try:
            answer_start = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            answer_start = 0
        return tokenizer.decode(
            output_ids[answer_start:], skip_special_tokens=True
        ).strip()

    @staticmethod
    def get_semantic_component_labels(
        model,
        tokenizer,
        examples: List[Tuple[str, Sequence[str] | str]],
        queries: Dict[int, str],
        numshots: int = 3,
        *,
        seed: int = 42,
    ) -> Dict[int, str]:
        if not examples:
            raise ValueError("at least one example is required")
        random_state = random.Random(seed)
        selected = random_state.sample(examples, min(numshots, len(examples)))
        questions = [item[0] for item in selected]
        answers = [
            item[1]
            if isinstance(item[1], str)
            else "".join(f"<{label.replace(' ', '_')}>" for label in item[1])
            for item in selected
        ]
        prompt = ModelUtil.build_qa_pair(questions, answers)
        return {
            value_id: ModelUtil.get_response(
                model,
                tokenizer,
                f"{prompt}Q: {query} A: ",
            )
            for value_id, query in queries.items()
        }
