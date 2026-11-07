import re

# Keys omit connectors; only these requested model names receive aliases.
OPENAI_MODEL_ALIASES = {
    "gpt5.1": "gpt-5.1",
    "gpt5.4": "gpt-5.4",
    "gpt6": "gpt-6-astra",
    "gpt6astra": "gpt-6-astra",
}


def resolve_openai_model(model: str) -> str:
    key = re.sub(r"[\s_\-\u2010-\u2015]+", "", model.casefold())
    return OPENAI_MODEL_ALIASES.get(key, model)


def is_gpt_model(model: str) -> bool:
    return resolve_openai_model(model).casefold().startswith("gpt-")


def report_model_name(model: str) -> str:
    canonical = resolve_openai_model(model)
    if canonical in {"gpt-5.1", "gpt-5.4", "gpt-6-astra"}:
        return {"gpt-5.1": "GPT5.1", "gpt-5.4": "GPT5.4", "gpt-6-astra": "GPT6"}[canonical]
    return canonical.replace(" ", "")
