from __future__ import annotations

import csv
import json
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal
from functools import cache
from pathlib import Path

from gdt.models import is_gpt_model, report_model_name, resolve_openai_model
from gdt.timing import exclude_output

PRICING_FILE = Path(__file__).resolve().parent.parent / "pricing.json"
LONG_CONTEXT_INPUT_TOKENS = 273_000
_active_costs: ContextVar[CostTracker | None] = ContextVar("model_costs", default=None)


@cache
def _encoding(model: str):
    import tiktoken

    try:
        return tiktoken.encoding_for_model(resolve_openai_model(model))
    except KeyError:
        # Ollama models (and new GPT IDs unknown to installed tiktoken versions)
        # use this fixed tiktoken encoding for comparable text-token estimates.
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, model: str) -> int:
    return len(_encoding(model).encode(text, disallowed_special=()))


def pricing_model(model: str) -> str:
    from LLMUtil.ollama import OLLAMAUtil

    canonical = resolve_openai_model(model).casefold()
    # Match both the configured Ollama name and its actual request model ID.
    for alias, api_model in OLLAMAUtil.MODEL_ALIASES.items():
        if canonical == api_model.casefold():
            return alias.casefold()
    return canonical


@dataclass
class RequestTokens:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @exclude_output
    def record_output(self, text: str) -> None:
        self.output_tokens += count_tokens(text, self.model)


@dataclass
class CostTracker:
    pricing_file: Path = field(default_factory=lambda: PRICING_FILE)
    costs: dict[str, Decimal | None] = field(
        default_factory=lambda: {"llm": Decimal(0), "slm": Decimal(0)},
    )
    tokens: dict[str, int] = field(default_factory=lambda: {
        "llm_input_tokens": 0, "llm_output_tokens": 0,
        "slm_input_tokens": 0, "slm_output_tokens": 0,
    })
    _prices: dict | None = field(default=None, init=False)
    _missing: set[str] = field(default_factory=set, init=False)

    @exclude_output
    def record(self, request: RequestTokens) -> None:
        family = "llm" if is_gpt_model(request.model) else "slm"
        self.tokens[f"{family}_input_tokens"] += request.input_tokens
        self.tokens[f"{family}_output_tokens"] += request.output_tokens
        if self._prices is None:
            with self.pricing_file.open(encoding="utf-8") as handle:
                self._prices = {
                    pricing_model(model): rates
                    for model, rates in json.load(handle, parse_float=Decimal).items()
                }
        model = pricing_model(request.model)
        rates = self._prices.get(model)
        if rates is None:
            self.costs[family] = None
            if model not in self._missing:
                warnings.warn(f"No pricing for {model} in {self.pricing_file}; cost is N/A.",
                              stacklevel=2)
                self._missing.add(model)
            return
        tier = ("long" if is_gpt_model(model)
                and request.input_tokens >= LONG_CONTEXT_INPUT_TOKENS else "short")
        cost = (request.input_tokens * Decimal(str(rates["input"][tier]))
                + request.output_tokens * Decimal(str(rates["output"][tier])))
        if self.costs[family] is not None:
            self.costs[family] += cost

    def result_fields(self) -> dict:
        llm, slm = self.costs["llm"], self.costs["slm"]
        return {
            **self.tokens,
            "cf_llm": float(llm) if llm is not None else None,
            "cf_slm": float(slm) if slm is not None else None,
            "cf_total": float(llm + slm) if llm is not None and slm is not None else None,
        }


def pricing_active() -> bool:
    return _active_costs.get() is not None


@contextmanager
def cost_session(tracker: CostTracker):
    token = _active_costs.set(tracker)
    try:
        yield tracker
    finally:
        _active_costs.reset(token)


@contextmanager
def track_request(model: str, *input_texts: str):
    """Count each submitted attempt, including rejected replies and request failures.

    Text fields are encoded separately. No provider usage counts, hidden reasoning,
    chat framing estimates, or lifecycle load/unload requests enter the calculation.
    """
    tracker = _active_costs.get()
    if tracker is None:
        yield None
        return
    request = RequestTokens(model)
    _count_input(request, input_texts)
    try:
        yield request
    finally:
        tracker.record(request)


@exclude_output
def _count_input(request: RequestTokens, texts: Sequence[str]) -> None:
    request.input_tokens = sum(count_tokens(text, request.model) for text in texts)


def write_pricing(
    results, output_directory: str | Path, *, filename: str = "pricing.csv",
) -> None:
    """Sum all datasets, repetitions, modes and failed runs within each setup."""
    totals: dict[str, dict[str, Decimal | None]] = {}
    for result in results:
        name = (f"{report_model_name(result.llm_model)}_"
                f"{report_model_name(result.model)}_{result.num_examples}")
        values = totals.setdefault(name, {"cf_llm": Decimal(0), "cf_slm": Decimal(0)})
        for field_name in values:
            value = getattr(result, field_name)
            if value is None:
                values[field_name] = None
            elif values[field_name] is not None:
                values[field_name] += Decimal(str(value))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    with (output / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Experiment", "CF-LLM", "CF-SLM", "CF-Total"])
        for name, values in totals.items():
            llm, slm = values["cf_llm"], values["cf_slm"]
            total = llm + slm if llm is not None and slm is not None else None
            writer.writerow([name, *(_format_cost(value) for value in (llm, slm, total))])


def _format_cost(value: Decimal | None) -> str:
    if value is None:
        return "N/A"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text
