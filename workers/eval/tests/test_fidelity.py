import numpy as np
from stl_eval.fidelity import quality


def test_quality_high_for_same_distribution(toy, fresh):
    train, _, _, md = toy
    overall, shapes, pairs = quality(train, fresh, md)
    assert overall > 0.85 and shapes > 0.85 and pairs > 0.85


def test_quality_pairs_drop_when_columns_shuffled(toy, fresh):
    train, _, _, md = toy
    shuffled = fresh.copy()
    rng = np.random.default_rng(0)
    for col in shuffled:
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
    _, _, pairs_good = quality(train, fresh, md)
    _, _, pairs_bad = quality(train, shuffled, md)
    assert pairs_bad < pairs_good - 0.1
