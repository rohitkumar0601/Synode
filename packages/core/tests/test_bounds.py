import math

import pytest
from stl_core.bounds import bound_value_problems


def test_valid_bounds_have_no_problems():
    assert bound_value_problems({"a": [0, 1], "b": (-2.5, 7)}, ["a"]) == []


@pytest.mark.parametrize("bound, needle", [
    ("05", "not a [lower, upper] pair"),  # a string is iterable: must not parse as [0, 5]
    (5, "not a [lower, upper] pair"),
    ([0, 1, 2], "not a [lower, upper] pair"),
    ([0], "not a [lower, upper] pair"),
    (["a", 1], "not numeric"),
    ([None, 1], "not numeric"),
    ([0, math.inf], "finite"),
    ([math.nan, 1], "finite"),
    ([3, 3], "lower < upper"),
    ([5, -5], "lower < upper"),
])
def test_bad_bound_values(bound, needle):
    problems = bound_value_problems({"x": bound}, [])
    assert len(problems) == 1 and problems[0].startswith("'x'") and needle in problems[0]


def test_log_scale_needs_nonnegative_lower():
    assert bound_value_problems({"x": [-1, 5]}, []) == []
    (msg,) = bound_value_problems({"x": [-1, 5]}, ["x"])
    assert "nonnegative" in msg
