__all__ = [
    "NovitaUtil",
    "OLLAMAUtil",
    "OpenAIUtil",
    "OpenAIUtil_Prompt",
]


def __getattr__(name):
    if name in __all__:
        from LLMUtil.LLMUtil import (
            NovitaUtil,
            OLLAMAUtil,
            OpenAIUtil,
            OpenAIUtil_Prompt,
        )

        return {
            "NovitaUtil": NovitaUtil,
            "OLLAMAUtil": OLLAMAUtil,
            "OpenAIUtil": OpenAIUtil,
            "OpenAIUtil_Prompt": OpenAIUtil_Prompt,
        }[name]
    raise AttributeError(name)
