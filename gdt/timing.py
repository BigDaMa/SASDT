from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from functools import wraps
import sys
import time


_active_timer = ContextVar("pipeline_timer", default=None)


class PipelineTimer:
    def __enter__(self):
        self.started = time.perf_counter()
        self.excluded = 0.0
        self.depth = 0
        self.finished = None
        self.token = _active_timer.set(self)
        self.stdout = redirect_stdout(_OutputStream(sys.stdout))
        self.stderr = redirect_stderr(_OutputStream(sys.stderr))
        self.stdout.__enter__()
        self.stderr.__enter__()
        return self

    def stop(self):
        if self.finished is None:
            self.finished = time.perf_counter()

    @property
    def elapsed(self):
        end = self.finished if self.finished is not None else time.perf_counter()
        return max(0.0, end - self.started - self.excluded)

    def __exit__(self, *exc):
        self.stop()
        self.stderr.__exit__(*exc)
        self.stdout.__exit__(*exc)
        _active_timer.reset(self.token)


def stop_pipeline_timer():
    timer = _active_timer.get()
    if timer is not None:
        timer.stop()


@contextmanager
def output_time():
    timer = _active_timer.get()
    if timer is None or timer.finished is not None:
        yield
        return
    outermost = timer.depth == 0
    started = time.perf_counter() if outermost else None
    timer.depth += 1
    try:
        yield
    finally:
        timer.depth -= 1
        if outermost:
            end = timer.finished if timer.finished is not None else time.perf_counter()
            timer.excluded += end - started


def exclude_output(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        with output_time():
            return function(*args, **kwargs)
    return wrapper


class _OutputStream:
    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    @exclude_output
    def write(self, text):
        return self.stream.write(text)

    @exclude_output
    def flush(self):
        return self.stream.flush()
