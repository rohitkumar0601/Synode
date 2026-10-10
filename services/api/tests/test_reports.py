import io
import json
import zipfile

from sqlmodel import Session, select
from stl_api import db
from stl_api.models import Dataset, Job, Run
from stl_core.schemas import MetricReport, PrivacyMetrics
from stl_runner import keys
from stl_runner.bundle import export_bundle
from stl_runner.pipeline import load_jobs, set_state
from stl_runner.storage import store_from_env
from test_runs import make_run, registered_dataset


def metric_report(run_id, job_id, generator, seed, dataset_id, leaky):
    risk = 0.9 if leaky else 0.02
    return MetricReport(
        job_id=job_id, run_id=run_id, dataset_id=dataset_id, generator=generator, seed=seed,
        status="done",
        fidelity={"quality_score": 0.99 if leaky else 0.8, "detection_auc": 0.5,
                  "alpha_precision": 0.9, "beta_recall": 0.9},
        utility={"tstr": {"auc": 0.9 if leaky else 0.7}, "importance_spearman": 0.9},
        privacy=PrivacyMetrics(
            dcr_share=1.0 if leaky else 0.5, singling_out_risk=risk, linkability_risk=risk,
            inference_risk=risk, mia_advantage=0.0, singling_out_status="measured"))


def finish(client, did, generators=("train_copy", "independent_marginals"), seeds=(0, 1),
           only=None):
    run_id = make_run(client, did, generators=generators, seeds=seeds).json()["run_id"]
    store = store_from_env()
    for job in client.get(f"/runs/{run_id}").json()["jobs"]:
        if only is not None and job["job_id"] not in only:
            continue
        rep = metric_report(run_id, job["job_id"], job["generator"], job["seed"], did,
                            job["generator"] == "train_copy")
        store.write_json(keys.report(run_id, job["job_id"]), rep)
        spec = next(j for j in load_jobs(store, run_id) if j.job_id == job["job_id"])
        set_state(store, spec, "done")
    return run_id


def test_report_gate_and_winner(client):
    did = registered_dataset(client)
    run_id = finish(client, did)
    r = client.get(f"/runs/{run_id}/report")
    assert r.status_code == 200
    body = r.json()
    assert body["jobs_reported"] == body["jobs_total"] == 4
    rec = body["recommendation"]
    assert rec["winner"] == "independent_marginals"
    ranked = {c["generator"]: c for c in rec["ranking"]}
    assert ranked["train_copy"]["gate_passed"] is False
    assert {a["generator"] for a in body["aggregated"]} == {"train_copy", "independent_marginals"}
    pareto = {p["generator"]: p for p in body["pareto"]}
    assert pareto["train_copy"]["gate_passed"] is False
    assert pareto["independent_marginals"]["gate_passed"] is True
    assert pareto["independent_marginals"]["on_front"] is True
    assert set(pareto["train_copy"]) == {"generator", "fidelity", "utility", "privacy",
                                         "on_front", "gate_passed"}
    assert len(body["per_job"]) == 4
    assert {j["status"] for j in body["per_job"]} == {"done"}


def test_report_custom_weights_and_preset(client):
    did = registered_dataset(client)
    run_id = finish(client, did)
    r = client.get(f"/runs/{run_id}/report", params={"weights": "1,1,2"})
    assert r.status_code == 200 and r.json()["recommendation"]["preset"] == "custom"
    r = client.get(f"/runs/{run_id}/report", params={"preset": "privacy_first"})
    assert r.status_code == 200 and r.json()["recommendation"]["preset"] == "privacy_first"


def test_report_partial_and_empty(client):
    did = registered_dataset(client)
    empty = make_run(client, did).json()["run_id"]
    body = client.get(f"/runs/{empty}/report").json()
    assert body["jobs_reported"] == 0 and body["jobs_total"] == 4
    assert body["recommendation"] is None and body["aggregated"] == []
    assert {j["status"] for j in body["per_job"]} == {"queued"}
    run_id = finish(client, did, only={"independent_marginals__s0"})
    body = client.get(f"/runs/{run_id}/report").json()
    assert body["jobs_reported"] == 1 < body["jobs_total"] == 4
    assert body["recommendation"] is not None


def test_report_errors(client):
    did = registered_dataset(client)
    run_id = finish(client, did)
    for w in ("1,1", "0,0,0", "a,b,c", "-1,1,1", "nan,1,1"):
        assert client.get(f"/runs/{run_id}/report", params={"weights": w}).status_code == 422, w
    assert client.get(f"/runs/{run_id}/report", params={"preset": "nope"}).status_code == 422
    assert client.get("/runs/missing/report").status_code == 404


def _delete_run_rows(run_id):
    with Session(db.get_engine()) as s:
        for j in s.exec(select(Job).where(Job.run_id == run_id)):
            s.delete(j)
        s.delete(s.get(Run, run_id))
        s.commit()


def test_import_bundle(client, tmp_path, monkeypatch):
    did = registered_dataset(client)
    run_id = finish(client, did)
    zpath = tmp_path / "b.zip"
    export_bundle(store_from_env(), run_id, str(zpath))
    _delete_run_rows(run_id)
    monkeypatch.setenv("STL_STORAGE_ROOT", str(tmp_path / "store2"))
    r = client.post("/import-results", files={"file": ("b.zip", zpath.read_bytes())})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_id"] == run_id and body["imported"] == 4
    assert any("not registered locally" in w for w in body["warnings"])
    runs = client.get("/runs").json()
    assert [x["run_id"] for x in runs] == [run_id] and runs[0]["counts"] == {"done": 4}
    assert client.get(f"/runs/{run_id}/report").json()["recommendation"]["winner"] == \
        "independent_marginals"
    again = client.post("/import-results", files={"file": ("b.zip", zpath.read_bytes())})
    assert again.status_code == 422 and "--force" in again.json()["detail"]
    forced = client.post("/import-results", data={"force": "true"},
                         files={"file": ("b.zip", zpath.read_bytes())})
    assert forced.status_code == 200 and forced.json()["imported"] == 4


def test_import_bad_bundles_422(client, tmp_path):
    assert client.post("/import-results",
                       files={"file": ("b.zip", b"not a zip")}).status_code == 422
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("bundle.json", json.dumps({
            "format_version": 1, "run_id": "../evil", "dataset_id": "d", "created_at": "x",
            "dataset_fingerprint": "f", "jobs": [], "include_synthetic": False}))
    r = client.post("/import-results", files={"file": ("b.zip", buf.getvalue())})
    assert r.status_code == 422 and "run_id" in r.json()["detail"]
    assert client.get("/runs").json() == []
    assert not (tmp_path / "store" / "runs").exists()


def _adult_conf(tmp_path, monkeypatch, name="toy.csv", df=None):
    from test_datasets import make_df
    csv = tmp_path / name
    (make_df() if df is None else df).to_csv(csv, index=False)
    conf = tmp_path / "conf" / "datasets"
    conf.mkdir(parents=True, exist_ok=True)
    (conf / "adult.yaml").write_text(f"source: file\npath: {csv.as_posix()}\ntarget: y\n")
    monkeypatch.setenv("STL_CONF_DIR", str(tmp_path / "conf"))


def _adult_bundle(client, tmp_path, monkeypatch):
    _adult_conf(tmp_path, monkeypatch)
    assert client.post("/datasets/builtin/adult", json={"max_rows": 80}).status_code == 201
    run_id = finish(client, "adult")
    zpath = tmp_path / "adult.zip"
    export_bundle(store_from_env(), run_id, str(zpath))
    _delete_run_rows(run_id)
    with Session(db.get_engine()) as s:
        s.delete(s.get(Dataset, "adult"))
        s.commit()
    monkeypatch.setenv("STL_STORAGE_ROOT", str(tmp_path / "store2"))
    r = client.post("/import-results", files={"file": ("b.zip", zpath.read_bytes())})
    assert r.status_code == 200, r.text
    return run_id


def test_import_placeholder_promoted_by_builtin(client, tmp_path, monkeypatch):
    _adult_bundle(client, tmp_path, monkeypatch)
    assert client.get("/datasets/adult").json()["source"] == "import"
    r = client.post("/datasets/builtin/adult", json={"max_rows": 80})
    assert r.status_code == 201, r.text
    assert "warnings" not in r.json() and r.json()["registered"] is True
    got = client.get("/datasets/adult").json()
    assert got["source"] == "builtin" and got["metadata"]["columns"]
    # now a real dataset with a run: replacing it is blocked again
    assert client.post("/datasets/builtin/adult", json={}).status_code == 409


def test_import_placeholder_promotion_warns_on_fingerprint_change(client, tmp_path, monkeypatch):
    from test_datasets import make_df
    _adult_bundle(client, tmp_path, monkeypatch)
    _adult_conf(tmp_path, monkeypatch, "other.csv", make_df(n=60).assign(age=1.5))
    r = client.post("/datasets/builtin/adult", json={"max_rows": 80})
    assert r.status_code == 201, r.text
    assert any("fingerprint differs" in w and "not comparable" in w for w in r.json()["warnings"])
    assert client.get("/datasets/adult").json()["source"] == "builtin"


def test_report_store_failure_503(client, monkeypatch):
    did = registered_dataset(client)
    run_id = finish(client, did)
    store_cls = type(store_from_env())

    def boom(self, key):
        raise ConnectionError("s3 down")

    monkeypatch.setattr(store_cls, "exists", boom)
    r = client.get(f"/runs/{run_id}/report")
    assert r.status_code == 503 and r.json()["detail"].startswith("artifact store unavailable:")


def test_report_missing_manifest_409(client):
    did = registered_dataset(client)
    run_id = finish(client, did)
    with Session(db.get_engine()) as s:
        s.add(Run(id="ghost", dataset_id=did, generators_json="[]", seeds_json="[]",
                  preset="balanced"))
        s.commit()
    assert client.get("/runs/ghost/report").status_code == 409
    assert client.get(f"/runs/{run_id}/report").status_code == 200


def test_report_corrupt_report_warns(client):
    did = registered_dataset(client)
    run_id = finish(client, did)
    store_from_env().write_bytes(keys.report(run_id, "train_copy__s0"), b"{not json")
    body = client.get(f"/runs/{run_id}/report").json()
    assert body["jobs_reported"] == 3
    bad = next(j for j in body["per_job"] if j["job_id"] == "train_copy__s0")
    assert bad["warnings"][0].startswith("report unreadable: ")


def test_clean_nonfinite_to_null():
    from stl_api.routes.reports import _clean
    out = _clean({"a": float("nan"), "b": [float("inf"), 1.0, (float("-inf"),)], "c": "x"})
    assert out == {"a": None, "b": [None, 1.0, [None]], "c": "x"}
    json.dumps(out, allow_nan=False)


def test_forced_import_syncs_run_rows(client, tmp_path):
    did = registered_dataset(client)
    run_id = finish(client, did)
    zpath = tmp_path / "b.zip"
    export_bundle(store_from_env(), run_id, str(zpath))
    # simulate DB drift
    with Session(db.get_engine()) as s:
        run = s.get(Run, run_id)
        run.generators_json, run.seeds_json = "[]", "[]"
        s.add(run)
        s.commit()
    r = client.post("/import-results", data={"force": "true"},
                    files={"file": ("b.zip", zpath.read_bytes())})
    assert r.status_code == 200
    got = client.get(f"/runs/{run_id}").json()
    assert sorted(got["generators"]) == ["independent_marginals", "train_copy"]
    assert got["seeds"] == [0, 1] and got["counts"] == {"done": 4}


def test_import_store_write_error_503_and_tempfile_removed(client, tmp_path, monkeypatch):
    import tempfile as tf

    from stl_api.routes import reports
    made = []
    real = tf.mkstemp

    def spy(*a, **k):
        fd, path = real(*a, **k)
        made.append(path)
        return fd, path

    monkeypatch.setattr(reports.tempfile, "mkstemp", spy)
    did = registered_dataset(client)
    run_id = finish(client, did)
    zpath = tmp_path / "b.zip"
    export_bundle(store_from_env(), run_id, str(zpath))
    data = zpath.read_bytes()
    ok = client.post("/import-results", data={"force": "true"}, files={"file": ("b.zip", data)})
    bad = client.post("/import-results", files={"file": ("b.zip", b"nope")})
    assert ok.status_code == 200 and bad.status_code == 422

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(reports, "import_bundle", boom)
    err = client.post("/import-results", files={"file": ("b.zip", data)})
    assert err.status_code == 503 and "disk full" in err.json()["detail"]
    import os
    assert len(made) == 3 and not any(os.path.exists(p) for p in made)
