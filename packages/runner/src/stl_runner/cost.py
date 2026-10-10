"""Wall-clock time and peak resident memory for a block of code."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psutil


@dataclass
class Measurement:
    seconds: float = 0.0
    peak_mb: float = 0.0
    start_mb: float = 0.0  # resident memory when the block started


@contextmanager
def measure(poll_seconds: float = 0.05) -> Iterator[Measurement]:
    proc = psutil.Process()
    peak = [proc.memory_info().rss]
    stop = threading.Event()

    def poll() -> None:
        while not stop.wait(poll_seconds):
            peak[0] = max(peak[0], proc.memory_info().rss)

    thread = threading.Thread(target=poll, daemon=True)
    m = Measurement(start_mb=peak[0] / 2**20)
    thread.start()
    start = time.perf_counter()
    try:
        yield m
    finally:
        m.seconds = time.perf_counter() - start
        stop.set()
        thread.join()
        m.peak_mb = max(peak[0], proc.memory_info().rss) / 2**20
