import yaml
from stl_meta.select import freeze, select_from_table

ROW = dict(NumberOfInstances=1000, NumberOfFeatures=11, NumberOfClasses=2, MinorityClassSize=100)


def table():
    return {
        1: {"name": "ok1", "target": "y", **ROW},
        2: {"name": "tiny", "target": "y", **ROW, "NumberOfInstances": 499},
        3: {"name": "huge", "target": "y", **ROW, "NumberOfInstances": 20001},
        4: {"name": "wide", "target": "y", **ROW, "NumberOfFeatures": 62},
        5: {"name": "manyclass", "target": "y", **ROW, "NumberOfClasses": 21},
        6: {"name": "rare", "target": "y", **ROW, "MinorityClassSize": 24},
        7: {"name": "ok2", "target": "t", **ROW, "NumberOfFeatures": 61},
        8: {"name": "ok3", "target": "y", **ROW},
    }


def test_criteria_filter_and_cap():
    sel, exc = select_from_table(table(), max_n=2)
    assert [d["id"] for d in sel] == [1, 7]
    reasons = {e["id"]: e["reason"] for e in exc}
    assert reasons[2].startswith("instances") and reasons[3].startswith("instances")
    assert reasons[4].startswith("features") and reasons[5].startswith("classes")
    assert reasons[6].startswith("min_class") and reasons[8] == "cap: beyond max_n"
    assert sel[1]["n_features"] == 60


def test_yaml_round_trip(tmp_path):
    sel, exc = select_from_table(table(), max_n=25)
    p = tmp_path / "d.yaml"
    freeze(p, sel, exc)
    got = yaml.safe_load(p.read_text())
    assert got["datasets"] == sel and got["excluded"] == exc
    assert set(got["datasets"][0]) == {
        "id", "name", "target", "n_instances", "n_features", "n_classes"}
