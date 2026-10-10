import json

from stl_api.dispatch import CeleryDispatcher, NullDispatcher, job_chain
from stl_core.catalog import CATALOG
from stl_core.schemas import JobSpec
from stl_runner.pipeline import set_state
from stl_runner.storage import store_from_env
from test_datasets import upload


def registered_dataset(client):
    did = upload(client).json()["id"]
    assert client.post(f"/datasets/{did}/register").status_code == 200
    return did


def make_run(client, did, generators=("train_copy", "mst_eps1"), seeds=(0, 1), **extra):
    return client.post("/runs", json={"dataset_id": did, "generators": list(generators),
                                      "seeds": list(seeds), **extra})


def test_generators_listing(client):
    body = client.get("/generators").json()
    assert {g["name"] for g in body} == set(CATALOG)
    by = {g["name"]: g for g in body}
    assert by["gaussian_copula"]["queue"] == "q.sdv" and by["tabddpm"]["cpu_ok"] is False
    assert by["privbayes_eps1"]["audit"] is True and by["ctgan"]["audit"] is False
    assert by["mst_eps1"]["params"] == {"epsilon": 1.0}
    assert set(by["ctgan"]) == {"name", "family", "queue", "cpu_ok", "params", "audit"}


def test_create_run_enqueues_every_job(client):
    did = registered_dataset(client)
    disp = client.app.state.dispatcher
    assert isinstance(disp, NullDispatcher)
    r = make_run(client, did, allow_gpu_only=True)
    assert r.status_code == 201
    body = r.json()
    assert body["run_id"].startswith(f"{did}-") and len(body["jobs"]) == 4
    assert {j["state"] for j in body["jobs"]} == {"queued"}
    assert len(disp.calls) == 4
    assert all(allow for _, allow in disp.calls)
    queues = {job.job_id: CATALOG[job.generator].queue for job, _ in disp.calls}
    assert queues == {"train_copy__s0": "q.sdv", "train_copy__s1": "q.sdv",
                      "mst_eps1__s0": "q.smartnoise", "mst_eps1__s1": "q.smartnoise"}
    got = client.get(f"/runs/{body['run_id']}").json()
    assert got["counts"] == {"queued": 4} and got["done"] is False
    assert got["generators"] == ["train_copy", "mst_eps1"] and got["preset"] == "balanced"
    assert [r["run_id"] for r in client.get("/runs").json()] == [body["run_id"]]


def test_job_time_cap_default(client):
    did = registered_dataset(client)
    make_run(client, did, generators=["train_copy"], seeds=[0])
    job, _ = client.app.state.dispatcher.calls[0]
    assert job.time_cap_seconds == 7200


def test_unknown_generator_422(client):
    did = registered_dataset(client)
    r = make_run(client, did, generators=["nope"])
    assert r.status_code == 422 and "nope" in json.dumps(r.json()["detail"])
    assert client.app.state.dispatcher.calls == []


def test_unregistered_dataset_409(client):
    did = upload(client).json()["id"]
    assert make_run(client, did).status_code == 409
    assert make_run(client, "missing").status_code == 409


def test_bad_weights_422(client):
    did = registered_dataset(client)
    assert make_run(client, did, weights=[1, 1]).status_code == 422
    assert make_run(client, did, weights=[0, 0, 0]).status_code == 422
    assert make_run(client, did, preset="nope").status_code == 422
    ok = make_run(client, did, weights=[1, 1, 2])
    assert ok.status_code == 201


def test_status_refresh_from_store(client):
    did = registered_dataset(client)
    run_id = make_run(client, did).json()["run_id"]
    job = client.app.state.dispatcher.calls[0][0]
    set_state(store_from_env(), job, "generated")
    got = client.get(f"/runs/{run_id}").json()
    states = {j["job_id"]: j["state"] for j in got["jobs"]}
    assert states[job.job_id] == "generated"
    assert got["counts"] == {"generated": 1, "queued": 3}
    # persisted in the DB too
    assert client.get("/runs").json()[0]["counts"] == {"generated": 1, "queued": 3}


def test_done_when_all_terminal(client):
    did = registered_dataset(client)
    run_id = make_run(client, did, generators=["train_copy"], seeds=[0, 1]).json()["run_id"]
    store = store_from_env()
    for job, _ in client.app.state.dispatcher.calls:
        set_state(store, job, "done")
    assert client.get(f"/runs/{run_id}").json()["done"] is True


def test_retry_requeues_failed_only(client):
    did = registered_dataset(client)
    run_id = make_run(client, did).json()["run_id"]
    disp = client.app.state.dispatcher
    failed = disp.calls[0][0]
    set_state(store_from_env(), failed, "failed", "boom")
    disp.calls.clear()
    r = client.post(f"/runs/{run_id}/retry")
    assert r.status_code == 200 and r.json()["retried"] == [failed.job_id]
    assert [j.job_id for j, _ in disp.calls] == [failed.job_id]
    errs = {j["job_id"]: j["error"] for j in client.get(f"/runs/{run_id}").json()["jobs"]}
    assert errs[failed.job_id] is None
    states = {j["job_id"]: j["state"] for j in client.get(f"/runs/{run_id}").json()["jobs"]}
    assert states[failed.job_id] == "queued"
    # a second retry finds nothing failed, so nothing is enqueued twice
    disp.calls.clear()
    assert client.post(f"/runs/{run_id}/retry").json()["retried"] == []
    assert disp.calls == []
    assert client.post("/runs/nope/retry").status_code == 404
    assert client.get("/runs/nope").status_code == 404


def test_run_blocks_dataset_changes(client):
    did = registered_dataset(client)
    make_run(client, did)
    assert client.post(f"/datasets/{did}/register").status_code == 409


def test_job_chain_structure():
    job = JobSpec(job_id="mst_eps1__s0", run_id="r", dataset_id="d", generator="mst_eps1", seed=0)
    c = job_chain(job, allow_gpu_only=True)
    gen, metrics, ev = c.tasks
    assert [t.task for t in c.tasks] == ["stl.generate", "stl.metrics", "stl.evaluate"]
    assert [t.options["queue"] for t in c.tasks] == ["q.smartnoise", "q.metrics", "q.eval"]
    assert all(t.immutable for t in c.tasks)
    job_json = job.model_dump_json()
    assert json.loads(gen.args[0])["job_id"] == "mst_eps1__s0" and gen.args == (job_json,)
    assert gen.kwargs == {"allow_gpu_only": True}
    assert metrics.args == (job_json,) and ev.args == (job_json,)
    sdv = job_chain(JobSpec(job_id="j", run_id="r", dataset_id="d",
                            generator="gaussian_copula", seed=0))
    assert sdv.tasks[0].options["queue"] == "q.sdv"
    assert sdv.tasks[0].kwargs == {"allow_gpu_only": False}


class FailingDispatcher:
    def __init__(self, fail_on=None):
        self.fail_on = fail_on  # None = fail everything
        self.calls = []

    def enqueue(self, job, allow_gpu_only):
        if self.fail_on is None or job.generator in self.fail_on:
            raise ConnectionError("broker down")
        self.calls.append((job, allow_gpu_only))


def test_all_dispatch_failures_503(client):
    did = registered_dataset(client)
    client.app.state.dispatcher = FailingDispatcher()
    r = make_run(client, did)
    body = r.json()
    assert r.status_code == 503 and set(body) == {"detail", "run_id", "jobs"}
    assert "broker down" in body["detail"]
    assert body["detail"].startswith(f"run {body['run_id']} created")
    assert len(body["jobs"]) == 4
    assert all(j["state"] == "failed" and "broker down" in j["error"] for j in body["jobs"])
    assert set(body["jobs"][0]) == {"job_id", "generator", "seed", "state", "error"}
    run = client.get("/runs").json()[0]
    assert run["run_id"] == body["run_id"]
    assert run["counts"] == {"failed": 4} and run["done"] is True
    detail = client.get(f"/runs/{run['run_id']}").json()
    assert detail["jobs"][0]["error"] == "dispatch failed: ConnectionError: broker down"


def test_partial_dispatch_failure_201(client):
    did = registered_dataset(client)
    client.app.state.dispatcher = FailingDispatcher(fail_on={"mst_eps1"})
    r = make_run(client, did)
    assert r.status_code == 201
    states = {j["job_id"]: j["state"] for j in r.json()["jobs"]}
    assert states == {"train_copy__s0": "queued", "train_copy__s1": "queued",
                      "mst_eps1__s0": "failed", "mst_eps1__s1": "failed"}
    store = store_from_env()
    st = store.read_json(f"runs/{r.json()['run_id']}/mst_eps1__s0/status.json")
    assert st["state"] == "failed" and "broker down" in st["error"]


def test_validation_422s(client):
    did = registered_dataset(client)
    for extra in ({"seeds": [0, 0]}, {"seeds": [-1]}, {"seeds": [True]}, {"seeds": []},
                  {"seeds": [1.5]}, {"params": {"ctgan": {}}},
                  {"generators": ["train_copy", "train_copy"]}, {"generators": []}):
        r = make_run(client, did, **{"generators": ["train_copy"], **extra})
        assert r.status_code == 422, extra
    r = client.post("/runs", json={"dataset_id": "../x", "generators": ["train_copy"]})
    assert r.status_code == 422
    assert client.get("/runs").json() == []
    assert make_run(client, did, generators=["train_copy"],
                    params={"train_copy": {}}).status_code == 201


class BrokenStore:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        if name in ("exists", "read_json"):
            raise OSError("store unreachable")
        return getattr(self.inner, name)


def test_store_unreachable_returns_stale_db_view(client):
    from stl_api.deps import get_store
    did = registered_dataset(client)
    run_id = make_run(client, did).json()["run_id"]
    real = store_from_env()
    client.app.dependency_overrides[get_store] = lambda: BrokenStore(real)
    try:
        r = client.get(f"/runs/{run_id}")
        assert r.status_code == 200 and r.json()["stale"] is True
        assert r.json()["counts"] == {"queued": 4}
        assert client.get("/runs").json()[0]["stale"] is True
    finally:
        client.app.dependency_overrides.clear()
    assert client.get(f"/runs/{run_id}").json()["stale"] is False


def test_terminal_runs_skip_store(client):
    from stl_api.deps import get_store
    did = registered_dataset(client)
    run_id = make_run(client, did, generators=["train_copy"], seeds=[0]).json()["run_id"]
    real = store_from_env()
    set_state(real, client.app.state.dispatcher.calls[0][0], "done")
    assert client.get(f"/runs/{run_id}").json()["done"] is True
    client.app.dependency_overrides[get_store] = lambda: BrokenStore(real)
    try:
        r = client.get(f"/runs/{run_id}").json()
        assert r["stale"] is False and r["done"] is True
    finally:
        client.app.dependency_overrides.clear()


def test_retry_uses_stored_allow_gpu_only_and_override(client):
    did = registered_dataset(client)
    run_id = make_run(client, did, generators=["train_copy"], seeds=[0],
                      allow_gpu_only=True).json()["run_id"]
    disp = client.app.state.dispatcher
    job = disp.calls[0][0]
    store = store_from_env()
    set_state(store, job, "failed", "x")
    disp.calls.clear()
    client.post(f"/runs/{run_id}/retry")
    assert [a for _, a in disp.calls] == [True]
    set_state(store, job, "failed", "x")
    client.get(f"/runs/{run_id}")
    disp.calls.clear()
    client.post(f"/runs/{run_id}/retry", json={"allow_gpu_only": False})
    assert [a for _, a in disp.calls] == [False]


def test_retry_missing_manifest_409(client):
    did = registered_dataset(client)
    run_id = make_run(client, did, generators=["train_copy"], seeds=[0]).json()["run_id"]
    store = store_from_env()
    set_state(store, client.app.state.dispatcher.calls[0][0], "failed", "x")
    import os
    os.remove(os.path.join(os.environ["STL_STORAGE_ROOT"], "runs", run_id, "manifest.json"))
    assert client.post(f"/runs/{run_id}/retry").status_code == 409


def test_celery_dispatcher_publishes_to_memory_broker():
    job = JobSpec(job_id="gaussian_copula__s0", run_id="r", dataset_id="d",
                  generator="gaussian_copula", seed=0)
    disp = CeleryDispatcher("memory://")
    disp.enqueue(job, True)  # must not raise
    with disp.app.connection_for_read() as conn:
        q = conn.SimpleQueue("q.sdv")
        msg = q.get(timeout=2)
        assert msg.headers["task"] == "stl.generate"
        assert msg.delivery_info["routing_key"] == "q.sdv"
        args, kwargs, embed = msg.decode()
        assert json.loads(args[0])["job_id"] == "gaussian_copula__s0"
        assert kwargs == {"allow_gpu_only": True}
        assert [t["task"] for t in embed["chain"]][::-1] == ["stl.metrics", "stl.evaluate"]
        msg.ack()
        q.close()


def test_retry_all_dispatch_failures_503_and_partial_200(client):
    did = registered_dataset(client)
    run_id = make_run(client, did).json()["run_id"]
    store = store_from_env()
    specs = [j for j, _ in client.app.state.dispatcher.calls]
    for spec in specs:
        set_state(store, spec, "failed", "boom")
    # partial failure: only the first generator's jobs fail to dispatch -> 200
    first = specs[0].generator
    client.app.state.dispatcher = FailingDispatcher(fail_on={first})
    r = client.post(f"/runs/{run_id}/retry")
    assert r.status_code == 200
    states = {j["job_id"]: j["state"] for j in client.get(f"/runs/{run_id}").json()["jobs"]}
    still_failed = {k for k, v in states.items() if v == "failed"}
    assert {s.job_id for s in specs if s.generator == first} <= still_failed
    # total failure: the still-failed jobs are retried and none dispatches -> 503
    client.app.state.dispatcher = FailingDispatcher()
    r = client.post(f"/runs/{run_id}/retry")
    body = r.json()
    assert r.status_code == 503 and set(body) == {"detail", "run_id", "retried"}
    assert body["run_id"] == run_id and "broker down" in body["detail"]
    assert sorted(body["retried"]) == sorted(s.job_id for s in specs if s.generator == first)


def test_run_weights_exposed(client):
    did = registered_dataset(client)
    rid = make_run(client, did, weights=[1, 1, 2]).json()["run_id"]
    assert client.get(f"/runs/{rid}").json()["weights"] == [0.25, 0.25, 0.5]
    assert next(r for r in client.get("/runs").json() if r["run_id"] == rid)["weights"] == [
        0.25, 0.25, 0.5]
    pid = make_run(client, did, preset="balanced").json()["run_id"]
    assert client.get(f"/runs/{pid}").json()["weights"] is None
    assert next(r for r in client.get("/runs").json() if r["run_id"] == pid)["weights"] is None


# --- cancellation, resume, q.metrics -------------------------------------------------------

import os  # noqa: E402
import time  # noqa: E402

from stl_api.dispatch import METRICS_QUEUE, eval_chain, parse_pending  # noqa: E402
from stl_runner import keys  # noqa: E402
from stl_runner.control import is_cancelled, request_cancel  # noqa: E402
from stl_runner.pipeline import job_state  # noqa: E402


def _states(client, run_id):
    return {j["job_id"]: j["state"] for j in client.get(f"/runs/{run_id}").json()["jobs"]}


def test_metrics_queue_and_eval_chain():
    assert METRICS_QUEUE == "q.metrics"
    job = JobSpec(job_id="ctgan__s0", run_id="r", dataset_id="d", generator="ctgan", seed=0)
    c = eval_chain(job)
    assert [t.task for t in c.tasks] == ["stl.metrics", "stl.evaluate"]
    assert [t.options["queue"] for t in c.tasks] == ["q.metrics", "q.eval"]
    assert all(t.immutable and t.args == (job.model_dump_json(),) for t in c.tasks)


def test_cancel_marks_unfinished_jobs_and_is_idempotent(client):
    did = registered_dataset(client)
    run_id = make_run(client, did).json()["run_id"]
    store = store_from_env()
    specs = {j.job_id: j for j, _ in client.app.state.dispatcher.calls}
    set_state(store, specs["train_copy__s0"], "done")
    set_state(store, specs["train_copy__s1"], "fitting")
    r = client.post(f"/runs/{run_id}/cancel")
    assert r.status_code == 202 and r.json() == {"run_id": run_id, "cancelled": 3}
    assert is_cancelled(store, run_id)
    assert _states(client, run_id) == {"train_copy__s0": "done", "train_copy__s1": "cancelled",
                                       "mst_eps1__s0": "cancelled", "mst_eps1__s1": "cancelled"}
    assert job_state(store, specs["mst_eps1__s0"])["state"] == "cancelled"  # store too
    got = client.get(f"/runs/{run_id}").json()
    assert got["done"] is True and got["cancelled"] is True
    assert got["counts"] == {"done": 1, "cancelled": 3}
    assert client.get("/runs").json()[0]["cancelled"] is True
    again = client.post(f"/runs/{run_id}/cancel")
    assert again.status_code == 202 and again.json()["cancelled"] == 0
    assert client.post("/runs/nope/cancel").status_code == 404


def test_refresh_does_not_resurrect_cancelled_jobs(client):
    did = registered_dataset(client)
    run_id = make_run(client, did, generators=["train_copy"], seeds=[0]).json()["run_id"]
    store = store_from_env()
    job = client.app.state.dispatcher.calls[0][0]
    request_cancel(store, run_id)  # e.g. a worker wrote "sampling" after the cancel
    set_state(store, job, "sampling")
    got = client.get(f"/runs/{run_id}").json()
    assert got["jobs"][0]["state"] == "cancelled" and got["done"] is True


def test_not_cancelled_flag_false(client):
    did = registered_dataset(client)
    run_id = make_run(client, did).json()["run_id"]
    assert client.get(f"/runs/{run_id}").json()["cancelled"] is False


def test_retry_cancelled_run_clears_flag_and_redispatches(client):
    did = registered_dataset(client)
    run_id = make_run(client, did).json()["run_id"]
    disp = client.app.state.dispatcher
    store = store_from_env()
    specs = {j.job_id: j for j, _ in disp.calls}
    set_state(store, specs["train_copy__s0"], "done")
    set_state(store, specs["train_copy__s1"], "failed", "boom")
    client.post(f"/runs/{run_id}/cancel")
    disp.calls.clear()
    r = client.post(f"/runs/{run_id}/retry")
    assert r.status_code == 200
    assert sorted(r.json()["retried"]) == ["mst_eps1__s0", "mst_eps1__s1", "train_copy__s1"]
    assert sorted(j.job_id for j, _ in disp.calls) == sorted(r.json()["retried"])
    assert not is_cancelled(store, run_id)
    got = client.get(f"/runs/{run_id}").json()
    assert got["cancelled"] is False and got["done"] is False
    assert _states(client, run_id)["mst_eps1__s0"] == "queued"


def _age(job, seconds):
    path = os.path.join(os.environ["STL_STORAGE_ROOT"], keys.status(job.run_id, job.job_id))
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_resume_reenqueues_lost_jobs(client):
    did = registered_dataset(client)
    run_id = make_run(client, did, generators=["train_copy", "mst_eps1", "gaussian_copula"],
                      seeds=[0, 1]).json()["run_id"]
    disp = client.app.state.dispatcher
    store = store_from_env()
    specs = {j.job_id: j for j, _ in disp.calls}
    disp.calls.clear()
    set_state(store, specs["train_copy__s0"], "done")
    set_state(store, specs["train_copy__s1"], "generated")       # lost continuation
    set_state(store, specs["mst_eps1__s0"], "fitting")           # recently active: untouched
    set_state(store, specs["mst_eps1__s1"], "fitting")
    _age(specs["mst_eps1__s1"], 7200 + 601)                      # stalled past cap + 600
    set_state(store, specs["gaussian_copula__s0"], "evaluating")
    _age(specs["gaussian_copula__s0"], 10_000)                   # stalled evaluation
    # gaussian_copula__s1 stays queued -> full chain
    r = client.post(f"/runs/{run_id}/resume")
    assert r.status_code == 200
    assert sorted(r.json()["resumed"]) == ["gaussian_copula__s0", "gaussian_copula__s1",
                                           "mst_eps1__s1", "train_copy__s1"]
    assert sorted(j.job_id for j, _ in disp.calls) == ["gaussian_copula__s1", "mst_eps1__s1"]
    assert sorted(j.job_id for j in disp.eval_calls) == ["gaussian_copula__s0",
                                                         "train_copy__s1"]
    assert client.post("/runs/nope/resume").status_code == 404


def test_resume_skips_jobs_pending_in_broker_and_honours_stall_env(client, monkeypatch):
    monkeypatch.setenv("STL_STALL_SECONDS", "5")
    did = registered_dataset(client)
    run_id = make_run(client, did, generators=["train_copy"], seeds=[0, 1, 2]).json()["run_id"]
    disp = client.app.state.dispatcher
    store = store_from_env()
    specs = {j.job_id: j for j, _ in disp.calls}
    disp.calls.clear()
    set_state(store, specs["train_copy__s0"], "generated")
    set_state(store, specs["train_copy__s2"], "sampling")
    _age(specs["train_copy__s2"], 10)
    disp.pending = {(run_id, "train_copy__s0"): {("stl.metrics", "q.metrics")},
                    (run_id, "train_copy__s1"): {("stl.generate", "q.sdv")}}
    r = client.post(f"/runs/{run_id}/resume")
    assert r.json()["resumed"] == ["train_copy__s2"]
    # a metrics message stuck behind training on the old q.synthcity queue is not "pending"
    disp.pending = {(run_id, "train_copy__s0"): {("stl.metrics", "q.synthcity")}}
    disp.calls.clear()
    disp.eval_calls.clear()
    r = client.post(f"/runs/{run_id}/resume")
    assert "train_copy__s0" in r.json()["resumed"]
    assert [j.job_id for j in disp.eval_calls] == ["train_copy__s0"]


def test_resume_cancelled_run_409(client):
    did = registered_dataset(client)
    run_id = make_run(client, did, generators=["train_copy"], seeds=[0]).json()["run_id"]
    request_cancel(store_from_env(), run_id)
    assert client.post(f"/runs/{run_id}/resume").status_code == 409


class _BrokenResumeDispatcher(FailingDispatcher):
    def enqueue_eval(self, job):
        raise ConnectionError("broker down")

    def pending_jobs(self):
        return None


def test_resume_dispatch_failure_503(client):
    did = registered_dataset(client)
    run_id = make_run(client, did, generators=["train_copy"], seeds=[0, 1]).json()["run_id"]
    store = store_from_env()
    specs = [j for j, _ in client.app.state.dispatcher.calls]
    set_state(store, specs[0], "generated")
    client.app.state.dispatcher = _BrokenResumeDispatcher()
    r = client.post(f"/runs/{run_id}/resume")
    assert r.status_code == 503 and "broker down" in r.json()["detail"]
    assert r.json()["resumed"] == []
    assert _states(client, run_id) == {"train_copy__s0": "generated", "train_copy__s1": "queued"}


def test_parse_pending_reads_celery_envelopes():
    import base64

    def envelope(task, job):
        body = json.dumps([[job.model_dump_json()], {}, {"chain": None}]).encode()
        return json.dumps({"body": base64.b64encode(body).decode(), "headers": {"task": task},
                           "properties": {"body_encoding": "base64"}})

    job = JobSpec(job_id="ctgan__s0", run_id="r1", dataset_id="d", generator="ctgan", seed=0)
    got = parse_pending([("q.synthcity", envelope("stl.metrics", job)),
                         ("q.synthcity", envelope("stl.generate", job)),
                         ("q.meta", envelope("stl.metafeatures", job)),
                         ("q.eval", "not json")])
    assert got == {("r1", "ctgan__s0"): {("stl.metrics", "q.synthcity"),
                                         ("stl.generate", "q.synthcity")}}


def test_celery_dispatcher_eval_chain_and_pending_on_memory_broker():
    job = JobSpec(job_id="ctgan__s0", run_id="r", dataset_id="d", generator="ctgan", seed=0)
    disp = CeleryDispatcher("memory://")
    disp.enqueue_eval(job)
    with disp.app.connection_for_read() as conn:
        q = conn.SimpleQueue("q.metrics")
        msg = q.get(timeout=2)
        assert msg.headers["task"] == "stl.metrics"
        _, _, embed = msg.decode()
        assert [t["task"] for t in embed["chain"]] == ["stl.evaluate"]
        msg.ack()
        q.close()
    assert disp.pending_jobs() is None  # only the redis transport can be inspected


def test_stall_seconds_setting(monkeypatch):
    from stl_api.settings import Settings
    monkeypatch.delenv("STL_STALL_SECONDS", raising=False)
    monkeypatch.setenv("STL_TIME_CAP_SECONDS", "100")
    assert Settings.from_env().stall_seconds == 700
    monkeypatch.setenv("STL_STALL_SECONDS", "30")
    assert Settings.from_env().stall_seconds == 30
