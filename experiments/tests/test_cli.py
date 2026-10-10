import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stl_runner import keys
from stl_runner.pipeline import job_state, load_jobs, set_state
from stl_runner.storage import ArtifactStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run  # noqa: E402

# The default dataset group is adult: a file-sourced toy must drop adult's public bounds.
TOY_DS = ["~dataset.public_bounds", "~dataset.public_log_scale"]


def _toy_csv(path: Path) -> Path:
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=1000)
    df = pd.DataFrame({"x1": x1, "x2": 2 * x1 + rng.normal(scale=0.3, size=1000),
                       "c": np.where(x1 > 0, "hi", "lo"),
                       "y": np.where(x1 + rng.normal(scale=0.5, size=1000) > 0, "p", "n")})
    p = path / "toy.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    monkeypatch.delenv("STL_STORAGE_ROOT", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("STL_ALLOW_GPU_ONLY", raising=False)


SC = {"alpha_precision": 0.9, "beta_recall": 0.5, "authenticity": 0.45, "detection_auc": 0.52,
      "mia_auc": 0.51, "mia_domias_advantage": 0.02}


@pytest.fixture
def fake_synthcity(monkeypatch):
    """Stand-in for stl_synthcity.metrics (the synthcity env is not the eval env): the
    `metrics` stage then writes fixed values, or a result chosen per job."""
    calls: list[str] = []
    results: dict[str, dict] = {}

    def compute_stored_job(store, job, max_rows=20_000):
        calls.append(job.job_id)
        out = {"metrics_version": 2, **results.get(job.job_id, SC)}
        store.write_json(keys.synthcity_metrics(job.run_id, job.job_id), out)
        return out

    stage = types.SimpleNamespace(compute_stored_job=compute_stored_job, METRICS_VERSION=2)
    monkeypatch.setattr(run, "_metrics_stage", lambda: stage)
    return calls, results


def _args(tmp_path, generators, extra=()):
    csv = _toy_csv(tmp_path).as_posix()
    store = (tmp_path / "store").as_posix()
    overrides = [*TOY_DS, "dataset.name=toy", "dataset.source=file", f"dataset.path='{csv}'",
                 "dataset.target=y", "seeds=[0]", f"storage_root='{store}'", "tracking_uri=''",
                 f"generators_to_run=[{generators}]", "budget.time_cap_seconds=null", *extra]
    return [a for o in overrides for a in ("-o", o)], ArtifactStore(store)


def test_full_local_run(tmp_path, capsys, fake_synthcity):
    csv = _toy_csv(tmp_path).as_posix()
    store = (tmp_path / "store").as_posix()
    overrides = [
        *TOY_DS,
        "dataset.name=toy", "dataset.source=file", f"dataset.path='{csv}'", "dataset.target=y",
        "seeds=[0]", "max_rows=1000", f"storage_root='{store}'", "tracking_uri=''",
        "generators_to_run=[train_copy,independent_marginals]",
        "eval.n_attacks=100", "eval.singling_out_mode=univariate",
        # Max-over-all-secrets inference at 100 attacks is noise-dominated on this 1k-row toy
        # (binary "c" alone can reach ~0.13); "all" is covered in workers/eval/tests.
        "eval.inference_secrets=target",
        "budget.time_cap_seconds=null",
    ]
    args = [a for o in overrides for a in ("-o", o)]
    assert run.main([*args, "prepare"]) == 0
    assert run.main([*args, "generate"]) == 0
    assert run.main([*args, "metrics"]) == 0
    assert run.main([*args, "evaluate"]) == 0
    capsys.readouterr()
    assert run.main([*args, "report", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["jobs_reported"] == 2 and out["jobs_total"] == 2
    rec = out["recommendation"]
    assert rec["winner"] == "independent_marginals"
    tc = next(c for c in rec["ranking"] if c["generator"] == "train_copy")
    assert tc["gate_passed"] is False
    assert {a["generator"] for a in out["aggregated"]} == {"train_copy", "independent_marginals"}


def test_generate_skips_unavailable(tmp_path, capsys):
    csv = _toy_csv(tmp_path).as_posix()
    store = (tmp_path / "store").as_posix()
    overrides = [*TOY_DS, "dataset.name=toy", "dataset.source=file", f"dataset.path='{csv}'",
                 "dataset.target=y", "seeds=[0]", f"storage_root='{store}'", "tracking_uri=''",
                 "generators_to_run=[ctgan]", "budget.time_cap_seconds=null"]
    args = [a for o in overrides for a in ("-o", o)]
    run.main([*args, "prepare"])
    assert run.main([*args, "generate"]) == 0
    assert "skipped ctgan__s0" in capsys.readouterr().out
    st = ArtifactStore(store)
    job = load_jobs(st, "toy-main")[0]
    assert job_state(st, job)["state"] == "queued"
    assert run.main([*args, "evaluate"]) == 0
    assert "ctgan__s0: skipped (state queued)" in capsys.readouterr().out
    assert not st.exists(keys.report(job.run_id, job.job_id))
    assert job_state(st, job)["state"] == "queued"


def test_evaluate_failed_job_without_report(tmp_path):
    args, st = _args(tmp_path, "train_copy")
    assert run.main([*args, "prepare"]) == 0
    job = load_jobs(st, "toy-main")[0]
    set_state(st, job, "failed", "boom")
    assert run.main([*args, "evaluate"]) == 1  # a failed report is a failure exit
    rep = st.read_json(keys.report(job.run_id, job.job_id))
    assert rep["status"] == "failed"
    assert rep["error"] == "boom"


def test_report_without_reports(tmp_path, capsys):
    args, _ = _args(tmp_path, "train_copy")
    assert run.main([*args, "prepare"]) == 0
    capsys.readouterr()
    assert run.main([*args, "report"]) == 1
    assert "no reports yet for run toy-main; run evaluate first" in capsys.readouterr().out


def test_generate_exit_code_on_failure(tmp_path, monkeypatch):
    args, st = _args(tmp_path, "train_copy")
    assert run.main([*args, "prepare"]) == 0

    def fail(store, job, **_):
        set_state(store, job, "failed", "boom")
        return "failed"

    monkeypatch.setattr(run, "run_generate_with_cap", fail)
    assert run.main([*args, "generate"]) == 1


def test_evaluate_reruns_interrupted_job(tmp_path, capsys):
    args, st = _args(tmp_path, "train_copy",
                     ["eval.n_attacks=50", "eval.min_attacks=50",
                      "eval.singling_out_mode=univariate",
                      "eval.inference_secrets=target"])
    assert run.main([*args, "prepare"]) == 0
    assert run.main([*args, "generate"]) == 0
    job = load_jobs(st, "toy-main")[0]
    set_state(st, job, "evaluating")  # a killed evaluation leaves this behind
    capsys.readouterr()
    assert run.main([*args, "evaluate"]) == 0
    out = capsys.readouterr().out
    assert "train_copy__s0: was interrupted; re-evaluating" in out
    assert job_state(st, job)["state"] == "done"


def test_evaluate_reports_skipped_in_progress_jobs(tmp_path, capsys):
    args, st = _args(tmp_path, "train_copy")
    assert run.main([*args, "prepare"]) == 0
    job = load_jobs(st, "toy-main")[0]
    set_state(st, job, "fitting")
    capsys.readouterr()
    assert run.main([*args, "evaluate"]) == 0
    assert "train_copy__s0: skipped (state fitting)" in capsys.readouterr().out


def test_report_counts_partial_runs(tmp_path, capsys):
    args, st = _args(tmp_path, "train_copy,ctgan",
                     ["eval.n_attacks=50", "eval.min_attacks=50",
                      "eval.singling_out_mode=univariate",
                      "eval.inference_secrets=target"])
    for cmd in ("prepare", "generate", "evaluate"):
        assert run.main([*args, cmd]) == 0
    capsys.readouterr()
    assert run.main([*args, "report"]) == 0
    assert "1 of 2 jobs reported" in capsys.readouterr().out
    assert run.main([*args, "report", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["jobs_reported"], out["jobs_total"]) == (1, 2)


def test_report_invalid_weights_exits_2(tmp_path, capsys):
    args, _ = _args(tmp_path, "train_copy",
                    ["eval.n_attacks=50", "eval.min_attacks=50",
                     "eval.singling_out_mode=univariate",
                     "eval.inference_secrets=target"])
    for cmd in ("prepare", "generate", "evaluate"):
        run.main([*args, cmd])
    capsys.readouterr()
    assert run.main([*args, "report", "--weights", "1,-1,1"]) == 2
    out = capsys.readouterr()
    assert "custom weights must be exactly 3 finite non-negative numbers" in out.out + out.err


def test_evaluate_invalid_eval_config_exits_2(tmp_path, capsys):
    args, st = _args(tmp_path, "train_copy", ["eval.inference_secrets=[nope]"])
    assert run.main([*args, "prepare"]) == 0
    assert run.main([*args, "generate"]) == 0
    capsys.readouterr()
    assert run.main([*args, "evaluate"]) == 2
    out = capsys.readouterr()
    assert "nope" in out.out + out.err
    job = load_jobs(st, "toy-main")[0]
    assert job_state(st, job)["state"] == "generated"


def test_dataset_configs():
    credit = run.load_config(["datasets=credit"])["dataset"]
    assert credit["openml_id"] == 46373 and credit["target"] == "y"
    diabetes = run.load_config(["datasets=diabetes"])["dataset"]
    assert diabetes["openml_id"] == 4541 and diabetes["target"] == "readmitted"
    assert set(diabetes["drop"]) == {"encounter_id", "patient_nbr"}


def test_generate_skips_gpu_only_without_cuda(tmp_path, capsys):
    args, st = _args(tmp_path, "tabddpm")
    assert run.main([*args, "prepare"]) == 0
    capsys.readouterr()
    assert run.main([*args, "generate"]) == 0
    assert "skipped tabddpm__s0: GPU-only generator" in capsys.readouterr().out
    assert job_state(st, load_jobs(st, "toy-main")[0])["state"] == "queued"


def test_export_import_results_roundtrip(tmp_path, capsys, fake_synthcity):
    args, _ = _args(tmp_path, "train_copy,independent_marginals",
                    ["eval.n_attacks=50", "eval.min_attacks=50",
                     "eval.singling_out_mode=univariate",
                     "eval.inference_secrets=target"])
    for cmd in ("prepare", "generate", "metrics", "evaluate"):
        assert run.main([*args, cmd]) == 0
    bundle = (tmp_path / "bundle.zip").as_posix()
    assert run.main([*args, "export-results", "--run", "toy-main", "--out", bundle]) == 0
    other = (tmp_path / "other").as_posix()
    args2 = [f"storage_root='{other}'" if a.startswith("storage_root=") else a for a in args]
    assert run.main([*args2, "prepare"]) == 0
    capsys.readouterr()
    assert run.main([*args2, "import-results", bundle]) == 0
    assert "imported 2 reports into run toy-main" in capsys.readouterr().out
    assert run.main([*args2, "report", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {a["generator"] for a in out["aggregated"]} == {"train_copy", "independent_marginals"}


def test_import_results_missing_bundle_exits_2(tmp_path):
    args, _ = _args(tmp_path, "train_copy")
    assert run.main([*args, "import-results", (tmp_path / "nope.zip").as_posix()]) == 2


def test_prepare_rejects_unknown_column_types_key(tmp_path, capsys):
    args, st = _args(tmp_path, "train_copy", ["+dataset.column_types.nope=categorical"])
    assert run.main([*args, "prepare"]) == 2
    err = capsys.readouterr().err
    assert "column_types" in err and "'nope'" in err
    assert not st.exists(keys.run_manifest("toy-main"))


def test_default_lineup_sweeps_mst_not_privbayes():
    cfg = run.load_config([])
    lineup = cfg["generators_to_run"]
    assert {"mst_eps0.1", "mst_eps1", "mst_eps10"} <= set(lineup)
    assert not any(g.startswith("privbayes") for g in lineup)
    assert {"mst_eps0.1", "mst_eps1", "mst_eps10"} <= set(cfg["generator_params"])


@pytest.mark.parametrize("name", ["adult", "credit", "diabetes"])
def test_dataset_public_bounds_are_well_formed(name):
    ds = run.load_config([f"datasets={name}"])["dataset"]
    bounds = ds["public_bounds"]
    assert bounds and all(lo < hi for lo, hi in bounds.values())
    assert set(ds["public_log_scale"]) <= set(bounds)
    assert all(bounds[c][0] >= 0 for c in ds["public_log_scale"])


def test_prepare_stores_public_bounds(tmp_path):
    args, st = _args(tmp_path, "train_copy", ["+dataset.public_bounds={x1:[-5,5]}",
                                              "+dataset.public_log_scale=[]"])
    assert run.main([*args, "prepare"]) == 0
    assert st.read_json(keys.dataset_public("toy")) == {
        "public_bounds": {"x1": [-5, 5]}, "public_log_scale": []}


@pytest.mark.parametrize("extra, bad", [
    (["+dataset.public_bounds={c:[0,1]}"], "'c'"),
    (["+dataset.public_bounds={nope:[0,1]}"], "'nope'"),
    (["+dataset.public_bounds={x1:[-5,5]}", "+dataset.public_log_scale=[x2]"], "'x2'"),
    (["+dataset.public_bounds={x1:[5,-5]}"], "'x1'"),  # lower >= upper
    (["+dataset.public_bounds={x1:[3,3]}"], "'x1'"),
    (["+dataset.public_bounds={x1:[0,1,2]}"], "'x1'"),  # not a [lower, upper] pair
    (["+dataset.public_bounds={x1:[a,1]}"], "'x1'"),  # not numeric
    (["+dataset.public_bounds={x1:'[0,5]'}"], "'x1'"),  # a string, not a list
    (["+dataset.public_bounds={x1:[0,inf]}"], "'x1'"),  # not finite
    (["+dataset.public_bounds={x1:[-1,5]}", "+dataset.public_log_scale=[x1]"], "'x1'"),
])
def test_prepare_rejects_invalid_public_bounds(tmp_path, capsys, extra, bad):
    args, st = _args(tmp_path, "train_copy", extra)
    assert run.main([*args, "prepare"]) == 2
    err = capsys.readouterr().err
    assert "public" in err and bad in err
    assert not st.exists(keys.run_manifest("toy-main"))


def test_metrics_stage_runs_generated_jobs_once(tmp_path, capsys, fake_synthcity):
    calls, results = fake_synthcity
    args, st = _args(tmp_path, "train_copy,independent_marginals,ctgan")
    assert run.main([*args, "prepare"]) == 0
    assert run.main([*args, "generate"]) == 0  # ctgan unavailable here: stays queued
    capsys.readouterr()
    assert run.main([*args, "metrics"]) == 0
    assert calls == ["train_copy__s0", "independent_marginals__s0"]
    assert "train_copy__s0: synthcity metrics done" in capsys.readouterr().out
    assert st.read_json(keys.synthcity_metrics("toy-main", "train_copy__s0")) == {
        "metrics_version": 2, **SC}
    assert run.main([*args, "metrics"]) == 0  # existing files are kept
    assert len(calls) == 2


def test_metrics_stage_failure_exit_code(tmp_path, capsys, fake_synthcity):
    calls, results = fake_synthcity
    args, st = _args(tmp_path, "train_copy,independent_marginals")
    results["train_copy__s0"] = {"error": "ValueError: boom"}
    results["independent_marginals__s0"] = {**SC, "mia_domias_advantage": None,
                                            "errors": {"domias": "LinAlgError: singular"}}
    assert run.main([*args, "prepare"]) == 0
    assert run.main([*args, "generate"]) == 0
    capsys.readouterr()
    assert run.main([*args, "metrics"]) == 1
    out = capsys.readouterr().out
    assert "train_copy__s0: synthcity metrics failed: ValueError: boom" in out
    assert ("independent_marginals__s0: synthcity metrics done with errors: "
            "domias: LinAlgError: singular") in out


def test_metrics_stage_outside_synthcity_env_exits_2(tmp_path, capsys):
    # stl_synthcity is not installed in the eval env this test runs in
    args, _ = _args(tmp_path, "train_copy")
    assert run.main([*args, "prepare"]) == 0
    capsys.readouterr()
    assert run.main([*args, "metrics"]) == 2
    assert "synthcity env" in capsys.readouterr().err


def test_metrics_stage_import_error_is_precise(tmp_path, capsys, monkeypatch):
    def broken():
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(run, "_metrics_stage", broken)
    args, _ = _args(tmp_path, "train_copy")
    assert run.main([*args, "prepare"]) == 0
    capsys.readouterr()
    assert run.main([*args, "metrics"]) == 2
    err = capsys.readouterr().err
    assert "importing the synthcity metrics failed: ModuleNotFoundError: No module named " \
           "'torch'" in err


def test_metrics_recompute_reruns_error_and_stale_files(tmp_path, capsys, fake_synthcity):
    calls, _ = fake_synthcity
    args, st = _args(tmp_path, "train_copy,independent_marginals,ctgan",
                     ["eval.n_attacks=50", "eval.min_attacks=50",
                      "eval.singling_out_mode=univariate", "eval.inference_secrets=target"])
    assert run.main([*args, "prepare"]) == 0
    assert run.main([*args, "generate"]) == 0
    assert run.main([*args, "metrics"]) == 0
    assert run.main([*args, "evaluate"]) == 0
    tc, im = (keys.synthcity_metrics("toy-main", j) for j in
              ("train_copy__s0", "independent_marginals__s0"))
    st.write_json(tc, {"error": "boom"})  # failed earlier
    st.write_json(im, {"metrics_version": 1, **SC})  # stale version
    calls.clear()
    assert run.main([*args, "metrics"]) == 0
    assert calls == []  # without --recompute existing files are kept
    capsys.readouterr()
    assert run.main([*args, "metrics", "--recompute"]) == 0
    assert calls == ["train_copy__s0", "independent_marginals__s0"]
    out = capsys.readouterr().out
    assert "train_copy__s0: recomputing synthcity metrics; report reset for re-evaluation" in out
    for job in load_jobs(st, "toy-main")[:2]:  # evaluated jobs go back to "generated"
        assert job_state(st, job)["state"] == "generated"
        assert not st.exists(keys.report(job.run_id, job.job_id))
        assert st.read_json(keys.synthcity_metrics(job.run_id, job.job_id))[
            "metrics_version"] == 2
    calls.clear()
    assert run.main([*args, "metrics", "--recompute"]) == 0
    assert calls == []  # current files are not recomputed


def test_report_warns_on_mixed_metric_versions(tmp_path, capsys, fake_synthcity):
    args, st = _args(tmp_path, "train_copy,independent_marginals",
                     ["eval.n_attacks=50", "eval.min_attacks=50",
                      "eval.singling_out_mode=univariate", "eval.inference_secrets=target"])
    for cmd in ("prepare", "generate", "metrics", "evaluate"):
        assert run.main([*args, cmd]) == 0
    capsys.readouterr()
    assert run.main([*args, "report"]) == 0
    assert "metric versions" not in capsys.readouterr().err
    key = keys.report("toy-main", "train_copy__s0")
    old = st.read_json(key)
    old.pop("metrics_version")  # a report written before versioning
    st.write_json(key, old)
    assert run.main([*args, "report"]) == 0
    assert ("warning: run toy-main mixes metric versions (None, 2): re-run metrics "
            "--recompute and evaluate") in capsys.readouterr().err


DEEP = ("ctgan", "tvae", "tabddpm", "great")


def test_default_budget_is_library_defaults():
    cfg = run.load_config([])
    assert cfg["budget"] == {"profile": "library", "epochs": None, "time_cap_seconds": 86400}
    assert all(cfg["generator_params"][g] == {"epochs": None} for g in DEEP)


@pytest.mark.parametrize("profile", ["library", "equal50", "smoke"])
def test_budget_profiles_compose_and_match_core(profile):
    from stl_core.budget import BUDGET_PROFILES

    cfg = run.load_config([f"budget={profile}"])
    assert cfg["budget"] == {"profile": profile, **BUDGET_PROFILES[profile]}
    epochs = BUDGET_PROFILES[profile]["epochs"]
    assert all(cfg["generator_params"][g] == {"epochs": epochs} for g in DEEP)


def test_budget_profile_fields_can_still_be_overridden():
    cfg = run.load_config(["budget=equal50", "budget.epochs=2", "budget.time_cap_seconds=60"])
    assert cfg["budget"] == {"profile": "equal50", "epochs": 2, "time_cap_seconds": 60}
    assert cfg["generator_params"]["ctgan"] == {"epochs": 2}


def test_unknown_budget_profile_is_rejected():
    from hydra.errors import MissingConfigException

    with pytest.raises(MissingConfigException):
        run.load_config(["budget=fast"])


@pytest.mark.parametrize("profile", ["library", "smoke"])
def test_prepare_records_budget_profile(tmp_path, profile):
    args, st = _args(tmp_path, "train_copy", [f"budget={profile}",
                                              "budget.time_cap_seconds=null"])
    assert run.main([*args, "prepare"]) == 0
    assert st.read_json(keys.run_manifest("toy-main"))["budget_profile"] == profile


def test_generate_and_evaluate_record_generator_params(tmp_path, fake_synthcity):
    args, st = _args(tmp_path, "independent_marginals",
                     ["budget=smoke", "budget.time_cap_seconds=null", "eval.n_attacks=100",
                      "eval.singling_out_mode=univariate", "eval.inference_secrets=target"])
    for cmd in ("prepare", "generate", "metrics", "evaluate"):
        assert run.main([*args, cmd]) == 0, cmd
    gp = st.read_json(keys.generator_params("toy-main", "independent_marginals__s0"))
    assert gp["library"] == "stl_core" and gp["profile"] == "smoke"
    report = st.read_json(keys.report("toy-main", "independent_marginals__s0"))
    assert report["generator_params"] == gp
