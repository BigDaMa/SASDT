from __future__ import annotations

from gdt.models import resolve_openai_model
from gdt.results import RunResult


class DirectModelError(RuntimeError):
    """Stop the entire direct experiment with a concise user-facing message."""


class UnsupportedModelError(DirectModelError):
    def __init__(self, model: str, provider: str, *, result: RunResult | None = None):
        super().__init__(f"Model '{model}' is not supported by provider '{provider}'.")
        self.result = result


def provider_model(model: str, provider: str) -> str:
    if provider == "openai":
        return resolve_openai_model(model)
    if provider == "novita":
        from LLMUtil.providers import NovitaUtil

        return NovitaUtil.get_novita_model(model)
    raise ValueError(f"unknown provider: {provider}")


def check_models(models: list[str], provider: str) -> None:
    """Read the provider's catalog once, before touching datasets or report files."""
    from LLMUtil.providers import NovitaUtil, OpenAIUtil

    adapter = {"openai": OpenAIUtil, "novita": NovitaUtil}[provider]
    try:
        client = adapter.get_client().with_options(max_retries=0, timeout=30)
        available = {model.id for model in client.models.list()}
    except Exception as error:
        # Authentication/network failures are not evidence of an unsupported model.
        raise DirectModelError(
            f"Could not check models at provider '{provider}' ({type(error).__name__}). "
            "Experiment stopped."
        ) from error
    for model in models:
        if provider_model(model, provider) not in available:
            raise UnsupportedModelError(model, provider)


def is_unsupported_model(error: Exception) -> bool:
    """Handle model/endpoint rejection even if a model appeared in the catalog."""
    if isinstance(error, UnsupportedModelError):
        return True
    body = getattr(error, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    body = body if isinstance(body, dict) else {}
    code = str(body.get("code") or getattr(error, "code", "")).casefold()
    if code in {"model_not_found", "model_not_available", "unsupported_model", "invalid_model",
                "model_not_supported"}:
        return True
    if getattr(error, "status_code", None) not in {400, 403, 404, 422}:
        return False
    message = str(body.get("message") or error).casefold()
    return "model" in message and any(phrase in message for phrase in (
        "not supported", "unsupported", "not found", "does not exist", "not exist",
        "not available", "do not have access", "no access", "decommissioned", "retired",
        "does not support",
    ))
