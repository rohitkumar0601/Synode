"""Deterministic OpenML-CC18 dataset selection for the RQ3 meta-benchmark."""
from __future__ import annotations

from pathlib import Path

import yaml

SUITE_ID = 99  # OpenML-CC18
MIN_ROWS, MAX_ROWS, MAX_FEATURES, MAX_CLASSES = 500, 20_000, 60, 20
# 60/20/20 split: the smallest class keeps >= 5 rows in the smallest (20%) part
MIN_CLASS_SIZE = 25


def _reason(q: dict) -> str | None:
    n, f = q["NumberOfInstances"], q["NumberOfFeatures"] - 1  # NumberOfFeatures counts the target
    if not MIN_ROWS <= n <= MAX_ROWS:
        return f"instances {int(n)} outside [{MIN_ROWS}, {MAX_ROWS}]"
    if f > MAX_FEATURES:
        return f"features {int(f)} > {MAX_FEATURES}"
    if not 2 <= q["NumberOfClasses"] <= MAX_CLASSES:
        return f"classes {int(q['NumberOfClasses'])} outside [2, {MAX_CLASSES}]"
    if q["MinorityClassSize"] < MIN_CLASS_SIZE:
        return (f"min_class {int(q['MinorityClassSize'])} < {MIN_CLASS_SIZE} "
                "(needs >=5 rows after 60/20/20 split)")
    return None


def select_from_table(table: dict[int, dict], max_n: int = 25) -> tuple[list[dict], list[dict]]:
    """table: {dataset_id: {name, target, NumberOf* qualities}} -> (selected, excluded)."""
    selected: list[dict] = []
    excluded: list[dict] = []
    for did in sorted(table):
        q = table[did]
        why = _reason(q)
        if why is None and len(selected) >= max_n:
            why = "cap: beyond max_n"
        if why:
            excluded.append({"id": int(did), "name": q["name"], "reason": why})
            continue
        selected.append({
            "id": int(did), "name": q["name"], "target": q["target"],
            "n_instances": int(q["NumberOfInstances"]),
            "n_features": int(q["NumberOfFeatures"]) - 1,
            "n_classes": int(q["NumberOfClasses"]),
        })
    return selected, excluded


def fetch_cc18_table() -> dict[int, dict]:
    import openml

    suite = openml.study.get_suite(SUITE_ID)
    ids = list(suite.data)
    df = openml.datasets.list_datasets(data_id=ids, size=len(ids) + 10)
    table: dict[int, dict] = {}
    for did, row in df.iterrows() if df.index.name == "did" else df.set_index("did").iterrows():
        ds = openml.datasets.get_dataset(
            int(did), download_data=False, download_qualities=False,
            download_features_meta_data=False)
        table[int(did)] = {
            "name": row["name"], "target": ds.default_target_attribute,
            **{k: float(row[k]) for k in
               ("NumberOfInstances", "NumberOfFeatures", "NumberOfClasses", "MinorityClassSize")},
        }
    return table


def freeze(path: str | Path, selected: list[dict], excluded: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        yaml.safe_dump({"suite": SUITE_ID, "datasets": selected, "excluded": excluded},
                       sort_keys=False),
        encoding="utf-8")


def select_cc18(max_n: int = 25, path: str | Path | None = None) -> list[dict]:
    selected, excluded = select_from_table(fetch_cc18_table(), max_n)
    if path is not None:
        freeze(path, selected, excluded)
    return selected


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[4] / "experiments/conf/meta/datasets.yaml"
    sel = select_cc18(25, out)
    print(len(sel), "selected ->", out)
