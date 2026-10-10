import numpy as np
import pandas as pd
import pytest
from stl_core.metadata import ColumnType, TableMetadata
from stl_core.split import split

MD = TableMetadata(
    columns={"x1": ColumnType.numerical, "x2": ColumnType.numerical,
             "c": ColumnType.categorical, "y": ColumnType.categorical},
    target="y",
)


def draw(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 2 * x1 + rng.normal(scale=0.3, size=n)
    c = np.where(x1 + rng.normal(scale=0.5, size=n) > 0, "hi", "lo")
    c = np.where(rng.random(n) < 0.1, "mid", c)
    y = np.where(x1 + 0.5 * (c == "hi") + rng.normal(scale=0.5, size=n) > 0.3, "pos", "neg")
    return pd.DataFrame({"x1": x1, "x2": x2, "c": c.astype(object), "y": y.astype(object)})


@pytest.fixture(scope="session")
def toy():
    s = split(draw(1500, 0), MD, seed=0)
    return s.train, s.control, s.test, MD


@pytest.fixture(scope="session")
def fresh():
    return draw(900, 99)
