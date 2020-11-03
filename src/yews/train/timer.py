import time
from contextlib import ContextDecorator
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Optional


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


def _print(msg, *args):
    print(msg % args)


@dataclass
class Timer(ContextDecorator):
    """Time your code using a class, context manager, or decorator.

    Note: Disable print by setting logger=None.

    """

    timers: ClassVar[Dict[str, float]] = {}
    calls: ClassVar[Dict[str, int]] = {}
    name: Optional[str] = None
    text: str = "Elapsed time: %.4f seconds"
    logger: Optional[Callable[[str], None]] = _print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialization: add timer to dict of timers"""
        if self.name is not None:
            self.timers.setdefault(self.name, 0)
            self.calls.setdefault(self.name, 0)

    @classmethod
    def reset(cls):
        """Reset all exist timers."""
        cls.timers = {}
        cls.calls = {}

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text, elapsed_time)
        if self.name is not None:
            self.timers[self.name] += elapsed_time
            self.calls[self.name] += 1

        return elapsed_time

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()
