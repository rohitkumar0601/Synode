import time

import numpy as np
from stl_runner.cost import measure


def test_measure_time_and_memory():
    with measure() as m:
        a = np.ones((2000, 2000))
        time.sleep(0.1)
        del a
    assert m.seconds >= 0.1
    assert m.peak_mb > 10


def test_measure_reports_starting_rss():
    with measure() as m:
        a = np.ones((2000, 2000))
        del a
    assert 0 < m.start_mb <= m.peak_mb
