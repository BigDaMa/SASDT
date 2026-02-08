import random
from typing import Dict, List, Tuple
import time

class ModelUtil:

    @staticmethod
    def build_qa_pair(q: List[str], a: List[str]) -> str:
        qa_pair = ""
        for question, answer in zip(q, a):
            qa_pair += f"Q: {question} A: {answer}\n"
        return qa_pair
    
    @staticmethod
    def get_response(model, tokenizer, prompt: str, max_length: int = 512) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_length
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return response
    
    @staticmethod
    def get_semantic_component_labels(model, tokenizer, examples: List[Tuple[str, List[str]]], queries: Dict[int, str], numshots: int = 3) -> Dict[int, str]:
        q_all = [e[0] for e in examples]
        a_all = ["".join("<{s}>".format(s = "_".join(label.split())) for label in e[1]) for e in examples]
        q = []
        a = []
        for i in range(numshots):
            idx = random.randint(0, len(q_all) - 1)
            q.append(q_all[idx])
            a.append(a_all[idx])
        prompt = ModelUtil.build_qa_pair(q, a)
        results = {}
        starttime = time.time()
        for idx, query in queries.items():
            full_prompt = prompt + f"Q: {query} A: "
            response = ModelUtil.get_response(model, tokenizer, full_prompt, max_length=512)
            results[idx] = response
        endtime = time.time()
        return results
