import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session
from stl_api import db
from stl_api.main import create_app
from stl_api.models import Run
from stl_runner import keys
from stl_runner.storage import store_from_env


def make_df(n=100, rare=False):
    rng = np.random.default_rng(0)
    y = ["a", "b"] * (n // 2)
    if rare:
        y[0] = "rare"
    return pd.DataFrame({
        "age": rng.integers(18, 90, n) + rng.random(n),
        "income": rng.random(n) * 1000,
        "city": rng.choice(["x", "y", "z"], n),
        "y": y[:n],
    })


def csv_bytes(df):
    return df.to_csv(index=False).encode()


def upload(client, df=None, name="My Data", target="y", filename="d.csv"):
    df = make_df() if df is None else df
    return client.post("/datasets", data={"name": name, "target": target},
                       files={"file": (filename, csv_bytes(df), "text/csv")})


def test_upload_csv(client):
    r = upload(client)
    assert r.status_code == 201
    body = r.json()
    assert body["id"].startswith("my-data-") and len(body["id"]) == len("my-data-") + 6
    assert body["n_rows"] == 100 and body["target"] == "y" and body["registered"] is False
    assert body["metadata"]["columns"]["age"] == "numerical"
    assert body["metadata"]["columns"]["city"] == "categorical"
    got = client.get(f"/datasets/{body['id']}").json()
    assert len(got["preview"]) == 5 and got["source"] == "upload"
    assert [d["id"] for d in client.get("/datasets").json()] == [body["id"]]


def test_upload_parquet(client):
    buf = io.BytesIO()
    make_df().to_parquet(buf, index=False)
    r = client.post("/datasets", data={"name": "pq", "target": "y"},
                    files={"file": ("d.parquet", buf.getvalue())})
    assert r.status_code == 201 and r.json()["n_rows"] == 100


def test_bad_target_422(client):
    r = upload(client, target="nope")
    assert r.status_code == 422 and "nope" in r.json()["detail"]


def test_oversize_413(tmp_path, monkeypatch, client):
    monkeypatch.setenv("STL_MAX_UPLOAD_MB", "1")
    big = pd.DataFrame({"a": np.arange(300_000), "y": ["a", "b"] * 150_000})
    assert len(csv_bytes(big)) > 1024 * 1024
    with TestClient(create_app()) as c:
        assert upload(c, big).status_code == 413


def test_oversize_chunked_no_content_length_413(monkeypatch, client):
    monkeypatch.setenv("STL_MAX_UPLOAD_MB", "1")

    def body():
        for _ in range(3):
            yield b"x" * (512 * 1024)

    with TestClient(create_app()) as c:
        r = c.post("/datasets/", content=body(),
                   headers={"content-type": "multipart/form-data; boundary=b"})
    assert r.status_code == 413
    assert r.json() == {"detail": "upload exceeds STL_MAX_UPLOAD_MB=1"}


def test_oversize_understated_content_length_413(monkeypatch, client):
    import asyncio
    monkeypatch.setenv("STL_MAX_UPLOAD_MB", "1")
    app = create_app()
    crlf = bytes([13, 10])
    disposition = b'Content-Disposition: form-data; name="file"; filename="d.csv"'
    head = crlf.join([b"--b", disposition, b"", b""])
    chunks = [head + b"x" * (512 * 1024)] + [b"x" * (512 * 1024)] * 2
    sent = []

    async def receive():
        if chunks:
            return {"type": "http.request", "body": chunks.pop(0), "more_body": bool(chunks)}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/datasets", "raw_path": b"/datasets",
             "query_string": b"", "scheme": "http", "server": ("t", 80), "root_path": "",
             "headers": [(b"content-type", b"multipart/form-data; boundary=b"),
                         (b"content-length", b"10")]}
    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 413
    assert b"STL_MAX_UPLOAD_MB=1" in sent[1]["body"]


def test_metadata_edit(client):
    did = upload(client).json()["id"]
    cols = {"age": "numerical", "income": "numerical", "city": "categorical", "y": "categorical"}
    r = client.put(f"/datasets/{did}/metadata", json={"columns": {**cols, "city": "id"},
                                                      "target": "y"})
    assert r.status_code == 200 and r.json()["metadata"]["columns"]["city"] == "id"
    assert r.json()["registered"] is False
    bad = client.put(f"/datasets/{did}/metadata", json={"columns": {**cols, "ghost": "numerical"},
                                                        "target": "y"})
    assert bad.status_code == 422 and "ghost" in bad.json()["detail"]
    bad = client.put(f"/datasets/{did}/metadata", json={"columns": cols, "target": "zzz"})
    assert bad.status_code == 422
    bad = client.put(f"/datasets/{did}/metadata", json={"columns": {**cols, "y": "id"},
                                                        "target": "y"})
    assert bad.status_code == 422


def test_register(client, tmp_path):
    did = upload(client).json()["id"]
    r = client.post(f"/datasets/{did}/register",
                    json={"max_rows": 60, "public_bounds": {"age": [0, 120]},
                          "public_log_scale": []})
    assert r.status_code == 200, r.text
    assert r.json()["registered"] is True and r.json()["n_rows"] == 60
    got = client.get(f"/datasets/{did}").json()
    assert got["fingerprint"]
    assert (tmp_path / "store" / "datasets" / did / "public.json").exists()
    assert (tmp_path / "store" / "datasets" / did / "splits" / "0" / "train.parquet").exists()
    assert client.app.state.dispatcher.meta_calls == [did]
    # editing metadata un-registers
    cols = got["metadata"]["columns"]
    assert client.put(f"/datasets/{did}/metadata",
                      json={"columns": cols, "target": "y"}).json()["registered"] is False


def test_register_bad_bounds_422(client):
    did = upload(client).json()["id"]
    r = client.post(f"/datasets/{did}/register", json={"public_bounds": {"city": [0, 1]}})
    assert r.status_code == 422 and "city" in r.json()["detail"]
    r = client.post(f"/datasets/{did}/register", json={"public_bounds": {"age": [5, 1]}})
    assert r.status_code == 422


def test_register_rare_class_422(client):
    did = upload(client, make_df(rare=True)).json()["id"]
    r = client.post(f"/datasets/{did}/register", json={})
    assert r.status_code == 422 and "rare" in r.json()["detail"]
    assert client.get(f"/datasets/{did}").json()["registered"] is False


def test_unknown_dataset_404(client):
    assert client.get("/datasets/nope").status_code == 404
    assert client.post("/datasets/nope/register", json={}).status_code == 404
    assert client.delete("/datasets/nope").status_code == 404
    assert client.post("/datasets/builtin/nope", json={}).status_code == 404


def test_delete_and_run_conflict(client):
    did = upload(client).json()["id"]
    with Session(db.get_engine()) as s:
        s.add(Run(id="r1", dataset_id=did, generators_json="[]", seeds_json="[0]", preset="p"))
        s.commit()
    assert client.delete(f"/datasets/{did}").status_code == 409
    with Session(db.get_engine()) as s:
        s.delete(s.get(Run, "r1"))
        s.commit()
    assert client.delete(f"/datasets/{did}").status_code == 204
    assert client.get(f"/datasets/{did}").status_code == 404


def test_builtin_local_yaml(client, tmp_path, monkeypatch):
    """The built-in path with a local file source (no network)."""
    csv = tmp_path / "toy.csv"
    make_df().assign(junk=1).to_csv(csv, index=False)
    conf = tmp_path / "conf" / "datasets"
    conf.mkdir(parents=True)
    (conf / "adult.yaml").write_text(
        f"source: file\npath: {csv.as_posix()}\ntarget: y\ndrop: [junk]\n"
        "column_types: {city: categorical}\npublic_bounds: {age: [0, 120]}\n"
        "public_log_scale: []\n")
    monkeypatch.setenv("STL_CONF_DIR", str(tmp_path / "conf"))
    r = client.post("/datasets/builtin/adult", json={"max_rows": 80})
    assert r.status_code == 201, r.text
    assert r.json()["id"] == "adult" and r.json()["registered"] is True
    assert "junk" not in r.json()["metadata"]["columns"]
    assert (tmp_path / "store" / "datasets" / "adult" / "public.json").exists()
    assert client.app.state.dispatcher.meta_calls == ["adult"]


FIXTURES = Path(__file__).parent / "fixtures"


def _registered(client):
    did = upload(client).json()["id"]
    assert client.post(f"/datasets/{did}/register", json={}).status_code == 200
    return did


def _suggestion(client, did):
    r = client.get(f"/datasets/{did}/suggestion")
    assert r.status_code == 200, r.text
    return r.json()


def test_suggestion_unknown_dataset_404(client):
    assert client.get("/datasets/nope/suggestion").status_code == 404


def test_suggestion_unregistered_dataset(client):
    did = upload(client).json()["id"]
    assert _suggestion(client, did) == {"available": False,
                                        "reason": "dataset is not registered"}


def test_suggestion_from_fixture_file(client):
    did = _registered(client)
    store_from_env().write_bytes(keys.dataset_suggestion(did),
                                 (FIXTURES / "suggestion.json").read_bytes())
    got = _suggestion(client, did)
    assert got == json.loads((FIXTURES / "suggestion.json").read_text())


def test_suggestion_reasons_when_missing(client):
    did = _registered(client)
    store = store_from_env()
    # no predictor, nothing computed yet
    got = _suggestion(client, did)
    assert got["available"] is False and "no meta-predictor" in got["reason"]
    # a predictor exists but the worker has not run yet
    store.write_bytes(keys.meta_predictor(), b"model")
    got = _suggestion(client, did)
    assert got["available"] is False and "not been computed yet" in got["reason"]
    # extraction failed: the stored error is the reason
    store.write_json(keys.dataset_metafeatures(did), {"error": "pymfe failed: boom"})
    got = _suggestion(client, did)
    assert got == {"available": False,
                   "reason": "meta-feature extraction failed: pymfe failed: boom"}


@pytest.mark.parametrize("content", [
    b"{not json", b"[1, 2]", b'{"available": true}', b'{"available": "yes"}',
    b'{"available": true, "generator": "ctgan", "probabilities": [0.5]}',
    bytes([0xFF, 0xFE]),
])
def test_suggestion_corrupt_file_is_not_500(client, content):
    did = _registered(client)
    store_from_env().write_bytes(keys.dataset_suggestion(did), content)
    assert _suggestion(client, did) == {"available": False,
                                        "reason": "stored suggestion is unreadable"}


def test_register_clears_stale_suggestion(client):
    did = _registered(client)
    store = store_from_env()
    store.write_bytes(keys.dataset_suggestion(did), (FIXTURES / "suggestion.json").read_bytes())
    store.write_json(keys.dataset_metafeatures(did), {"features": {"nr_inst": 1.0}})
    assert client.post(f"/datasets/{did}/register", json={}).status_code == 200
    assert not store.exists(keys.dataset_suggestion(did))
    assert not store.exists(keys.dataset_metafeatures(did))
    assert client.app.state.dispatcher.meta_calls == [did, did]


class _BrokenDispatcher:
    def enqueue(self, job, allow_gpu_only):
        raise ConnectionError("redis down")

    def enqueue_metafeatures(self, dataset_id):
        raise ConnectionError("redis down")


def test_enqueue_failure_is_not_fatal(client, tmp_path, monkeypatch, caplog):
    client.app.state.dispatcher = _BrokenDispatcher()
    did = upload(client).json()["id"]
    r = client.post(f"/datasets/{did}/register", json={})
    assert r.status_code == 200 and r.json()["registered"] is True
    csv = tmp_path / "toy.csv"
    make_df().to_csv(csv, index=False)
    conf = tmp_path / "conf" / "datasets"
    conf.mkdir(parents=True)
    (conf / "adult.yaml").write_text(f"source: file\npath: {csv.as_posix()}\ntarget: y\n")
    monkeypatch.setenv("STL_CONF_DIR", str(tmp_path / "conf"))
    assert client.post("/datasets/builtin/adult", json={}).status_code == 201
    assert caplog.text.count("could not enqueue meta-feature task") == 2


@pytest.mark.network
def test_builtin_adult(client):
    r = client.post("/datasets/builtin/adult", json={"max_rows": 500})
    assert r.status_code == 201, r.text
    assert r.json()["registered"] is True and r.json()["n_rows"] <= 500


def _with_run(did):
    with Session(db.get_engine()) as s:
        s.add(Run(id="r1", dataset_id=did, generators_json="[]", seeds_json="[0]", preset="p"))
        s.commit()


def test_mutations_blocked_when_runs_reference_dataset(client):
    did = upload(client).json()["id"]
    cols = client.get(f"/datasets/{did}").json()["metadata"]["columns"]
    _with_run(did)
    msg = f"dataset {did} is used by 1 run(s); create a new dataset instead"
    r = client.post(f"/datasets/{did}/register", json={})
    assert r.status_code == 409 and r.json()["detail"] == msg
    r = client.put(f"/datasets/{did}/metadata", json={"columns": cols, "target": "y"})
    assert r.status_code == 409 and r.json()["detail"] == msg


def test_builtin_replace_blocked_when_runs_reference_dataset(client, tmp_path, monkeypatch):
    conf = tmp_path / "conf" / "datasets"
    conf.mkdir(parents=True)
    csv = tmp_path / "toy.csv"
    make_df().to_csv(csv, index=False)
    (conf / "adult.yaml").write_text(f"source: file\npath: {csv.as_posix()}\ntarget: y\n")
    monkeypatch.setenv("STL_CONF_DIR", str(tmp_path / "conf"))
    assert client.post("/datasets/builtin/adult", json={}).status_code == 201
    _with_run("adult")
    r = client.post("/datasets/builtin/adult", json={})
    assert r.status_code == 409 and "used by 1 run(s)" in r.json()["detail"]


def test_failed_reregister_clears_registration(client):
    did = upload(client).json()["id"]
    assert client.post(f"/datasets/{did}/register", json={}).status_code == 200
    # make a high-cardinality categorical target: every class is rare
    cols = client.get(f"/datasets/{did}").json()["metadata"]["columns"]
    cols["city"] = "categorical"
    r = client.put(f"/datasets/{did}/metadata", json={"columns": cols, "target": "city"})
    assert r.status_code == 200
    # city has 3 classes of ~33 rows; shrink the table so classes become rare
    r = client.post(f"/datasets/{did}/register", json={"max_rows": 5})
    assert r.status_code == 422
    got = client.get(f"/datasets/{did}").json()
    assert got["registered"] is False and got["fingerprint"] is None


def test_builtin_missing_yaml_404(client, tmp_path, monkeypatch):
    monkeypatch.setenv("STL_CONF_DIR", str(tmp_path / "empty"))
    assert client.post("/datasets/builtin/adult", json={}).status_code == 404


def test_builtin_bad_target_422(client, tmp_path, monkeypatch):
    conf = tmp_path / "conf" / "datasets"
    conf.mkdir(parents=True)
    csv = tmp_path / "toy.csv"
    make_df().to_csv(csv, index=False)
    (conf / "adult.yaml").write_text(f"source: file\npath: {csv.as_posix()}\ntarget: nope\n")
    monkeypatch.setenv("STL_CONF_DIR", str(tmp_path / "conf"))
    r = client.post("/datasets/builtin/adult", json={})
    assert r.status_code == 422 and "nope" in r.json()["detail"]


BASE_FIELDS = {"id", "name", "source", "target", "n_rows", "metadata", "registered",
               "fingerprint"}


def test_every_dataset_response_has_base_fields(client, tmp_path, monkeypatch):
    created = upload(client)
    assert BASE_FIELDS <= created.json().keys()
    assert created.json()["source"] == "upload" and created.json()["fingerprint"] is None
    did = created.json()["id"]
    cols = created.json()["metadata"]["columns"]
    put = client.put(f"/datasets/{did}/metadata", json={"columns": cols, "target": "y"})
    assert BASE_FIELDS <= put.json().keys()
    reg = client.post(f"/datasets/{did}/register", json={"max_rows": 60})
    assert reg.status_code == 200, reg.text
    assert BASE_FIELDS <= reg.json().keys() and reg.json()["fingerprint"]
    csv = tmp_path / "toy.csv"
    make_df().to_csv(csv, index=False)
    conf = tmp_path / "conf" / "datasets"
    conf.mkdir(parents=True)
    (conf / "adult.yaml").write_text(
        f"source: file\npath: {csv.as_posix()}\ntarget: y\ncolumn_types: {{city: categorical}}\n"
        "public_bounds: {age: [0, 120]}\npublic_log_scale: []\n")
    monkeypatch.setenv("STL_CONF_DIR", str(tmp_path / "conf"))
    b = client.post("/datasets/builtin/adult", json={"max_rows": 80})
    assert b.status_code == 201, b.text
    assert BASE_FIELDS <= b.json().keys() and b.json()["source"] == "builtin"
    for item in client.get("/datasets").json():
        assert BASE_FIELDS <= item.keys()
    got = client.get(f"/datasets/{did}").json()
    assert BASE_FIELDS <= got.keys() and "preview" in got


def test_oversize_413_carries_cors_header(monkeypatch, client):
    monkeypatch.setenv("STL_MAX_UPLOAD_MB", "1")
    big = pd.DataFrame({"a": np.arange(300_000), "y": ["a", "b"] * 150_000})
    with TestClient(create_app()) as c:
        r = c.post("/datasets", data={"name": "x", "target": "y"},
                   files={"file": ("d.csv", csv_bytes(big), "text/csv")},
                   headers={"Origin": "http://localhost:3010"})
    assert r.status_code == 413
    assert r.headers["access-control-allow-origin"] == "http://localhost:3010"


def test_unhandled_error_is_json_500_with_cors(client):
    app = create_app()

    @app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom", headers={"Origin": "http://localhost:3010"})
    assert r.status_code == 500
    assert r.json() == {"detail": "internal server error"}
    assert r.headers["access-control-allow-origin"] == "http://localhost:3010"
