from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

from gdt.timing import exclude_output


_active_run = ContextVar("progress_run", default=None)


class ProgressLog:
    def __init__(self, output_directory, experiment, total):
        self.path = Path(output_directory) / "progress.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.total = total
        self.completed = 0
        self.write("EXPERIMENT", experiment=experiment, total=total)

    @exclude_output
    def write(self, event, **fields):
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        details = " | ".join(
            f"{key}={json.dumps(value, ensure_ascii=False, default=str)}"
            for key, value in fields.items()
        )
        # Closing each event also flushes it for readers in another terminal.
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {event} | {details}\n")

    @contextmanager
    def run(self, dataset, config, repetition=1, **fields):
        run = _Run(self, f"{self.completed + 1}/{self.total}", dataset)
        self.write(
            "START", run=run.identifier, dataset=dataset,
            config=config.as_display_dict(), repetition=repetition, **fields,
        )
        token = _active_run.set(run)
        started = perf_counter()
        try:
            yield run
        except BaseException as error:
            run.status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
            run.error = f"{type(error).__name__}: {error}"
            raise
        finally:
            _active_run.reset(token)
            self.completed += 1
            self.write(
                "END", run=run.identifier, dataset=dataset, status=run.status,
                elapsed_seconds=round(perf_counter() - started, 3),
                **run.details,
            )


class _Run:
    def __init__(self, log, identifier, dataset):
        self.log = log
        self.identifier = identifier
        self.dataset = dataset
        self.status = "failed"
        self.error = None
        self.result = None

    @property
    def details(self):
        if self.error:
            return {"error": self.error}
        if self.result is None:
            return {"error": "Run ended without a result"}
        return {
            "runtime_seconds": round(self.result.duration_seconds, 3),
            "accuracy": self.result.evaluation.accuracy if self.result.evaluation else None,
            "error": self.result.error,
            "has_lm_failures": self.result.has_lm_failures,
            "lm_failed_responses": self.result.lm_failed_responses,
            "lm_empty_results": self.result.lm_empty_results,
            "lm_tolerated_splits": self.result.lm_tolerated_splits,
        }

    def finish(self, result):
        self.result = result
        if result.succeeded:
            self.status = "success_with_lm_failures" if result.has_lm_failures else "success"
        else:
            self.status = "failed"


def progress_phase(name):
    run = _active_run.get()
    if run is not None:
        run.log.write("PHASE", run=run.identifier, dataset=run.dataset, phase=name)
