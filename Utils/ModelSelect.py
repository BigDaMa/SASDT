from typing import Union
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelSelect:

    @staticmethod
    def get_clustering_model_by_name(name: str = "all-MiniLM-L6-v2") -> Union[SentenceTransformer, AutoModelForCausalLM]:
        if name.lower() == "all-minilm-l6-v2":
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        elif name.lower() == "all-minilm-l12-v2":
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        elif name.lower() == "all-minilm":
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        elif name.lower() == "qwen":
            model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        elif name.lower() == "local-qwen-emb":
            model = SentenceTransformer("model/Qwen3-0.6B-Embedding-finetuned")
        elif name.lower() == "local-qwen-emb-peft":
            model = SentenceTransformer("model/Qwen3-0.6B-Embedding-finetuned-peft")
        elif name.lower() == "local-qwen-emb-prompt":
            model = SentenceTransformer("model/Qwen3-0.6B-Embedding-finetuned-peft-prompt")
        elif name.lower() == "local-qwen-prompt":
            model = AutoModelForCausalLM.from_pretrained(
                "model/Qwen3-0.6B-finetuned-peft-prompt",
                torch_dtype="auto",
                device_map="auto"
            )
        elif name.lower() == "local-qwen-prompt-labelonly":
            model = AutoModelForCausalLM.from_pretrained(
                "model/Qwen3-0.6B-finetuned-peft-prompt-labelonly",
                torch_dtype="auto",
                device_map="auto"
            )
        elif name.lower() == "local-qwen-1.7b-prompt-labelonly":
            model = AutoModelForCausalLM.from_pretrained(
                "model/Qwen3-1.7B-finetuned-peft-prompt-labelonly",
                torch_dtype="auto",
                device_map="auto"
            )
        elif name.lower() == "local-qwen-4b-prompt-labelonly":
            model = AutoModelForCausalLM.from_pretrained(
                "model/Qwen3-4B-finetuned-peft-prompt-labelonly",
                torch_dtype="auto",
                device_map="auto"
            )
        else:
            try:
                model = SentenceTransformer(name)
            except Exception as e:
                print(e)
                raise ValueError(f"Model '{name}' could not be loaded. Please check the model name or path.") 
        return model
    
    @staticmethod
    def get_tokenizer_by_name(name: str = "local-qwen-prompt-labelonly") -> AutoTokenizer:
        if name.lower() == "local-qwen-prompt":
            tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B-finetuned-peft-prompt")
        elif name.lower() == "local-qwen-prompt-labelonly":
            tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B-finetuned-peft-prompt-labelonly")
        elif name.lower() == "local-qwen-1.7b-prompt-labelonly":
            tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-1.7B-finetuned-peft-prompt-labelonly")
        elif name.lower() == "local-qwen-4b-prompt-labelonly":
            tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-4B-finetuned-peft-prompt-labelonly")
        else:
            raise ValueError(f"Tokenizer for model '{name}' is not defined.")
        return tokenizer