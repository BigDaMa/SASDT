from contextlib import contextmanager
import signal
import sys
from typing import Iterator


class DatasetTimeoutError(TimeoutError):
    """The dataset exhausted its total time allowance, including model waits."""


class _DatasetDeadlineExceeded(BaseException):
    """Escape provider/SDK ``except Exception`` handlers without being retried."""


@contextmanager
def dataset_time_limit(seconds: int) -> Iterator[None]:
    """Interrupt a Linux main-thread run; convert cancellation at its boundary.

    This alarm is independent of the measured runtime, which excludes output.
    Keep the cancellation outside Exception until all model retry layers unwind.
    """
    if not sys.platform.startswith("linux"):
        yield
        return

    def handle_timeout(_signal: int, _frame: object) -> None:
        raise _DatasetDeadlineExceeded()

    previous = signal.signal(signal.SIGALRM, handle_timeout)
    try:
        signal.alarm(seconds)
        try:
            yield
        except _DatasetDeadlineExceeded as error:
            timeout = DatasetTimeoutError(f"dataset exceeded {seconds} seconds")
            timeout.phase = getattr(error, "phase", "unknown")
            raise timeout from None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
