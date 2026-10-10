import zipfile

import numpy as np
import pandas as pd
import pytest
from stl_core.metadata import ColumnType, TableMetadata
from stl_core.schemas import MetricReport
from stl_runner import keys
from stl_runner.bundle import export_bundle, import_bundle
from stl_runner.pipeline import generate_job, load_jobs, plan_run, register_dataset
from stl_runner.storage import ArtifactStore

MD = TableMetadata(columns={"x": ColumnType.numerical, "c": ColumnType.categorical,
                            "y": ColumnType.categorical}, target="y")


def _df(n=500):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": rng.normal(size=n), "c": rng.choice(["a", "b"], n),
                         "y": rng.choice(["p", "n"], n)})


def _src(tmp_path, max_rows=400):
    store = ArtifactStore(str(tmp_path / "src"))
    register_dataset(store, "toy", _df(), MD, max_rows=max_rows)
    jobs = plan_run(store, "r1", "toy", ["train_copy", "independent_marginals"], [0])
    for job in jobs:
        generate_job(store, job)
        rep = MetricReport(job_id=job.job_id, run_id="r1", dataset_id="toy",
                           generator=job.generator, seed=job.seed, status="done")
        store.write_json(keys.report("r1", job.job_id), rep)
    return store, jobs


def _dst(tmp_path, max_rows=400):
    store = ArtifactStore(str(tmp_path / "dst"))
    register_dataset(store, "toy", _df(), MD, max_rows=max_rows)
    return store


def test_export_lists_jobs(tmp_path):
    store, jobs = _src(tmp_path)
    out = tmp_path / "b.zip"
    manifest = export_bundle(store, "r1", str(out))
    assert out.exists()
    assert [j.job_id for j in manifest.jobs] == [j.job_id for j in jobs]
    assert manifest.include_synthetic is False
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
    assert "bundle.json" in names
    assert "jobs/train_copy__s0/report.json" in names
    assert not any(n.endswith("synthetic.parquet") for n in names)


def test_export_with_synthetic(tmp_path):
    store, _ = _src(tmp_path)
    export_bundle(store, "r1", str(tmp_path / "b.zip"), include_synthetic=True)
    with zipfile.ZipFile(tmp_path / "b.zip") as z:
        assert "jobs/train_copy__s0/synthetic.parquet" in z.namelist()


def test_import_roundtrip_merges_manifest(tmp_path):
    store, jobs = _src(tmp_path)
    export_bundle(store, "r1", str(tmp_path / "b.zip"), include_synthetic=True)
    dst = _dst(tmp_path)
    plan_run(dst, "r1", "toy", ["tvae"], [0])  # pre-existing, different job
    reports = import_bundle(dst, str(tmp_path / "b.zip"))
    assert {r.job_id for r in reports} == {j.job_id for j in jobs}
    assert dst.exists(keys.report("r1", "train_copy__s0"))
    assert dst.exists(keys.synthetic("r1", "train_copy__s0"))
    assert {j.job_id for j in load_jobs(dst, "r1")} == {"tvae__s0", *[j.job_id for j in jobs]}


def test_import_fingerprint_mismatch(tmp_path):
    store, _ = _src(tmp_path)
    export_bundle(store, "r1", str(tmp_path / "b.zip"))
    dst = _dst(tmp_path, max_rows=300)
    with pytest.raises(ValueError, match="--force"):
        import_bundle(dst, str(tmp_path / "b.zip"))
    assert len(import_bundle(dst, str(tmp_path / "b.zip"), force=True)) == 2


def _tamper(src, dst, edit):
    """Copy a bundle, applying edit(manifest_dict, members_dict) in place."""
    import json
    with zipfile.ZipFile(src) as z:
        members = {n: z.read(n) for n in z.namelist()}
    manifest = json.loads(members.pop("bundle.json"))
    edit(manifest, members)
    with zipfile.ZipFile(dst, "w") as z:
        z.writestr("bundle.json", json.dumps(manifest))
        for n, d in members.items():
            z.writestr(n, d)


def _files(store):
    return store.list("")


def test_import_rejects_path_traversal_job_id(tmp_path):
    store, _ = _src(tmp_path)
    export_bundle(store, "r1", str(tmp_path / "b.zip"))

    def edit(m, members):
        m["jobs"][0]["job_id"] = "../../../evil"
        members["jobs/../../../evil/report.json"] = members["jobs/train_copy__s0/report.json"]

    _tamper(tmp_path / "b.zip", tmp_path / "bad.zip", edit)
    dst = _dst(tmp_path)
    before = _files(dst)
    with pytest.raises(ValueError, match="evil"):
        import_bundle(dst, str(tmp_path / "bad.zip"))
    assert _files(dst) == before


def test_import_rejects_bad_run_id(tmp_path):
    store, _ = _src(tmp_path)
    export_bundle(store, "r1", str(tmp_path / "b.zip"))
    _tamper(tmp_path / "b.zip", tmp_path / "bad.zip", lambda m, _: m.update(run_id="a/b"))
    dst = _dst(tmp_path)
    before = _files(dst)
    with pytest.raises(ValueError, match="run_id"):
        import_bundle(dst, str(tmp_path / "bad.zip"))
    assert _files(dst) == before


def test_import_rejects_mismatched_job_run_id(tmp_path):
    store, _ = _src(tmp_path)
    export_bundle(store, "r1", str(tmp_path / "b.zip"))
    _tamper(tmp_path / "b.zip", tmp_path / "bad.zip",
            lambda m, _: m["jobs"][0].update(run_id="other"))
    dst = _dst(tmp_path)
    before = _files(dst)
    with pytest.raises(ValueError, match="other"):
        import_bundle(dst, str(tmp_path / "bad.zip"))
    assert _files(dst) == before


def test_import_is_atomic_when_report_missing(tmp_path):
    store, _ = _src(tmp_path)
    export_bundle(store, "r1", str(tmp_path / "b.zip"))
    _tamper(tmp_path / "b.zip", tmp_path / "bad.zip",
            lambda m, mem: mem.pop("jobs/independent_marginals__s0/report.json"))
    dst = _dst(tmp_path)
    before = _files(dst)
    with pytest.raises(ValueError, match="missing"):
        import_bundle(dst, str(tmp_path / "bad.zip"))
    assert _files(dst) == before


def test_import_into_run_of_other_dataset_needs_force(tmp_path):
    store, _ = _src(tmp_path)
    export_bundle(store, "r1", str(tmp_path / "b.zip"))
    dst = _dst(tmp_path)
    plan_run(dst, "r1", "elsewhere", ["tvae"], [0])
    with pytest.raises(ValueError, match="--force"):
        import_bundle(dst, str(tmp_path / "b.zip"))
    assert len(import_bundle(dst, str(tmp_path / "b.zip"), force=True)) == 2


def test_import_warns_when_dataset_not_registered(tmp_path):
    store, _ = _src(tmp_path)
    export_bundle(store, "r1", str(tmp_path / "b.zip"))
    dst = ArtifactStore(str(tmp_path / "empty"))
    notes: list[str] = []
    assert len(import_bundle(dst, str(tmp_path / "b.zip"), warnings=notes)) == 2
    assert notes == ["dataset toy not registered locally; fingerprint not verified"]


def test_import_twice_is_idempotent(tmp_path):
    store, _ = _src(tmp_path)
    export_bundle(store, "r1", str(tmp_path / "b.zip"), include_synthetic=True)
    dst = _dst(tmp_path)
    import_bundle(dst, str(tmp_path / "b.zip"))
    files, jobs = _files(dst), load_jobs(dst, "r1")
    # re-importing over existing reports needs force (G5); the result is unchanged
    import_bundle(dst, str(tmp_path / "b.zip"), force=True)
    assert _files(dst) == files and load_jobs(dst, "r1") == jobs


def test_validate_component_rejects_trailing_newline():
    with pytest.raises(ValueError, match="invalid run_id"):
        keys.validate_component("r1\n", "run_id")
    assert keys.validate_component("r1", "run_id") == "r1"


def test_import_rejects_unknown_generator(tmp_path):
    store, _ = _src(tmp_path)
    export_bundle(store, "r1", str(tmp_path / "b.zip"))
    _tamper(tmp_path / "b.zip", tmp_path / "bad.zip",
            lambda m, _: m["jobs"][0].update(generator="evil_gen"))
    dst = _dst(tmp_path)
    before = _files(dst)
    with pytest.raises(ValueError, match="evil_gen"):
        import_bundle(dst, str(tmp_path / "bad.zip"))
    assert _files(dst) == before


@pytest.mark.parametrize("field,value", [("generator", "independent_marginals"),
                                         ("seed", 7), ("dataset_id", "other")])
def test_import_rejects_report_not_matching_job(tmp_path, field, value):
    import json
    store, _ = _src(tmp_path)
    export_bundle(store, "r1", str(tmp_path / "b.zip"))

    def edit(m, members):
        name = "jobs/train_copy__s0/report.json"
        rep = json.loads(members[name])
        rep[field] = value
        members[name] = json.dumps(rep).encode()

    _tamper(tmp_path / "b.zip", tmp_path / "bad.zip", edit)
    dst = _dst(tmp_path)
    before = _files(dst)
    with pytest.raises(ValueError, match=field):
        import_bundle(dst, str(tmp_path / "bad.zip"))
    assert _files(dst) == before


def test_import_refuses_to_overwrite_local_report_without_force(tmp_path):
    store, _ = _src(tmp_path)
    export_bundle(store, "r1", str(tmp_path / "b.zip"))
    dst = _dst(tmp_path)
    local = MetricReport(job_id="train_copy__s0", run_id="r1", dataset_id="toy",
                         generator="train_copy", seed=0, status="failed", error="local")
    dst.write_json(keys.report("r1", "train_copy__s0"), local)
    before = _files(dst)
    with pytest.raises(ValueError, match="train_copy__s0"):
        import_bundle(dst, str(tmp_path / "b.zip"))
    assert _files(dst) == before
    assert import_bundle(dst, str(tmp_path / "b.zip"), force=True)
    assert dst.read_json(keys.report("r1", "train_copy__s0"))["status"] == "done"


def test_import_caps_member_and_total_size(tmp_path, monkeypatch):
    from stl_runner import bundle
    store, _ = _src(tmp_path)
    export_bundle(store, "r1", str(tmp_path / "b.zip"))
    assert bundle.MAX_MEMBER_BYTES == 512 * 1024**2 and bundle.MAX_TOTAL_BYTES == 2 * 1024**3
    dst = _dst(tmp_path)
    before = _files(dst)
    with zipfile.ZipFile(tmp_path / "b.zip") as z:
        sizes = [i.file_size for i in z.infolist()]
    monkeypatch.setattr(bundle, "MAX_MEMBER_BYTES", max(sizes) - 1)
    with pytest.raises(ValueError, match="uncompressed"):
        import_bundle(dst, str(tmp_path / "b.zip"))
    monkeypatch.setattr(bundle, "MAX_MEMBER_BYTES", max(sizes))
    monkeypatch.setattr(bundle, "MAX_TOTAL_BYTES", sum(sizes) - 1)
    with pytest.raises(ValueError, match="uncompressed"):
        import_bundle(dst, str(tmp_path / "b.zip"))
    assert _files(dst) == before


PRIVACY = {"preprocessor": 0.0, "synthesizer": 1.0, "bounds": "public", "bounds_sha256": "ab"}


def test_privacy_json_is_optional_bundle_member(tmp_path):
    store, _ = _src(tmp_path)
    store.write_json(keys.privacy("r1", "train_copy__s0"), PRIVACY)
    export_bundle(store, "r1", str(tmp_path / "b.zip"))
    with zipfile.ZipFile(tmp_path / "b.zip") as z:
        names = set(z.namelist())
    assert "jobs/train_copy__s0/privacy.json" in names
    assert "jobs/independent_marginals__s0/privacy.json" not in names  # absent: skipped
    dst = _dst(tmp_path)
    import_bundle(dst, str(tmp_path / "b.zip"))
    assert dst.read_json(keys.privacy("r1", "train_copy__s0")) == PRIVACY
    assert not dst.exists(keys.privacy("r1", "independent_marginals__s0"))


def test_import_rejects_privacy_json_that_is_not_an_object(tmp_path):
    store, _ = _src(tmp_path)
    export_bundle(store, "r1", str(tmp_path / "b.zip"))

    def edit(manifest, members):
        members["jobs/train_copy__s0/privacy.json"] = b"[1, 2]"

    _tamper(tmp_path / "b.zip", tmp_path / "bad.zip", edit)
    dst = _dst(tmp_path)
    before = _files(dst)
    with pytest.raises(ValueError, match="privacy.json is not a JSON object"):
        import_bundle(dst, str(tmp_path / "bad.zip"))
    assert _files(dst) == before  # nothing written
