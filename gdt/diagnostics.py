from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

from gdt.timing import exclude_output
from gdt.progress import progress_phase
from gdt.timeout import DatasetTimeoutError, _DatasetDeadlineExceeded


_phase = ContextVar("llm_diagnostic_phase", default=None)
_dataset = ContextVar("llm_diagnostic_dataset", default=None)
_prompt = ContextVar("llm_diagnostic_prompt", default=None)


@dataclass
class ResponseDiagnostic:
    prompt: str
    response: str
    reason: str
    phase: str
    tolerated_inputs: list[str] = field(default_factory=list)


@dataclass
class DatasetDiagnostics:
    exchanges: list[ResponseDiagnostic] = field(default_factory=list)
    empty_results: list[tuple[str, list[str]]] = field(default_factory=list)
    ground_truth_errors: list[tuple[str, str, str | None]] = field(default_factory=list)

    def result_fields(self):
        return {
            "has_lm_failures": bool(self.exchanges or self.empty_results),
            "lm_failed_responses": len(self.exchanges),
            "lm_empty_results": sum(len(inputs) for _phase, inputs in self.empty_results),
            "lm_tolerated_splits": sum(len(item.tolerated_inputs) for item in self.exchanges),
        }


@contextmanager
def dataset_diagnostics(output_directory, dataset, config):
    diagnostics = DatasetDiagnostics()
    token = _dataset.set(diagnostics)
    try:
        yield diagnostics
    finally:
        _dataset.reset(token)
        _write_dataset_exchanges(output_directory, dataset, config, diagnostics)


def record_empty_results(originals, operation):
    dataset = _dataset.get()
    if dataset is not None and originals:
        phase = _phase.get()
        dataset.empty_results.append((phase[0] if phase else operation, list(originals)))


@exclude_output
def record_ground_truth_error(path, reason, original=None):
    """Record recoverable truth errors separately from rejected LM responses."""
    dataset = _dataset.get()
    if dataset is not None:
        entry = (str(path), reason, original)
        if entry not in dataset.ground_truth_errors:
            dataset.ground_truth_errors.append(entry)


def lm_failure_count():
    """Let caches avoid treating results of failed requests as clean reusable data."""
    dataset = _dataset.get()
    if dataset is None:
        return 0
    return len(dataset.exchanges) + sum(len(inputs) for _phase, inputs in dataset.empty_results)


@contextmanager
def capture_prompt():
    captured = []
    token = _prompt.set(captured)
    try:
        yield captured
    finally:
        _prompt.reset(token)


def record_prompt(prompt):
    """Capture the rendered prompt at the provider boundary, before sending it."""
    captured = _prompt.get()
    if captured is not None:
        captured[:] = [prompt]


@contextmanager
def split_phase(name):
    progress_phase(name)
    token = _phase.set((name, []))
    try:
        yield
    except _DatasetDeadlineExceeded as error:
        if not hasattr(error, "phase"):
            error.phase = name
        raise
    finally:
        _phase.reset(token)


@exclude_output
def record_exchange(history, prompt, response, reason, operation):
    if not reason:
        return
    phase = _phase.get()
    exchange = ResponseDiagnostic(prompt, response, reason, phase[0] if phase else operation)
    history.append(exchange)
    if phase is not None:
        phase[1].append(exchange)
    dataset = _dataset.get()
    if dataset is not None:
        dataset.exchanges.append(exchange)
    return exchange


@exclude_output
def _write_dataset_exchanges(output_directory, dataset, config, diagnostics):
    if not (diagnostics.exchanges or diagnostics.empty_results or diagnostics.ground_truth_errors):
        return
    with _error_file(output_directory) as handle:
        for path, reason, original in diagnostics.ground_truth_errors:
            _write_header(handle, dataset, config, "ground truth")
            handle.write(f"GROUND TRUTH ERROR: {reason}\nFILE: {path}\n")
            if original is not None:
                handle.write(f"INPUT: {original}\n")
            handle.write("=" * 96 + "\n\n")
        for exchange in diagnostics.exchanges:
            _write_header(handle, dataset, config, exchange.phase)
            _write_exchanges(handle, [exchange])
            handle.write("=" * 96 + "\n\n")
        for phase, originals in diagnostics.empty_results:
            _write_header(handle, dataset, config, phase)
            handle.write(
                f"TOLERATED: EMPTY RESULT for {len(originals)} unresolved split(s) after retries. "
                "Continuing with successful splits.\n"
            )
            for original in originals:
                handle.write(f"INPUT: {original}\n")
            handle.write("=" * 96 + "\n\n")


@exclude_output
def attach_history(error, operation, history):
    phase = _phase.get()
    error.phase = phase[0] if phase is not None else operation
    error.exchanges = list(phase[1] if phase is not None else history)
    error.responses_logged = _dataset.get() is not None


def clear_errors(output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "errors.txt").write_text("", encoding="utf-8")


@exclude_output
def write_dataset_error(output_directory, dataset, config, error):
    from LLMUtil.providers import SplitCountError

    if not isinstance(error, (SplitCountError, DatasetTimeoutError)):
        return
    exchanges = getattr(error, "exchanges", [])
    if isinstance(error, SplitCountError) and exchanges and getattr(error, "responses_logged", False):
        return
    with _error_file(output_directory) as handle:
        _write_header(handle, dataset, config, getattr(error, "phase", "unknown"))
        handle.write(f"{type(error).__name__}: {error}\n")
        if not exchanges and isinstance(error, SplitCountError):
            handle.write("No rejected responses were recorded.\n")
        _write_exchanges(handle, exchanges)
        handle.write("=" * 96 + "\n\n")


def _error_file(output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    return (output / "errors.txt").open("a", encoding="utf-8")


def _write_header(handle, dataset, config, phase):
    handle.write("=" * 96 + "\n")
    handle.write(f"DATASET: {dataset} | llm={config.llm_model} | slm={config.model} | mode={config.mode} | "
                 f"examples={config.num_examples}\n")
    handle.write(f"PHASE: {phase}\n")


def _write_exchanges(handle, exchanges):
    for index, exchange in enumerate(exchanges, 1):
        handle.write("-" * 96 + "\n")
        handle.write(f"ERROR: {exchange.reason}\n")
        if exchange.tolerated_inputs:
            handle.write(
                f"TOLERATED: Accepted {len(exchange.tolerated_inputs)} split(s) after retries; "
                "discarded empty-valued labels/standalone lines.\n"
            )
        handle.write(f"QUERY {index}:\n{exchange.prompt}\nRESPONSE {index}:\n{exchange.response}\n")
