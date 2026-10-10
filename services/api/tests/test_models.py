from sqlmodel import Session
from stl_api import db
from stl_api.models import Dataset, Job, Run


def test_models_round_trip(client):
    with Session(db.get_engine()) as s:
        s.add(Dataset(id="d1", name="adult", source="builtin", target="income"))
        s.add(Run(id="r1", dataset_id="d1", generators_json='["ctgan"]', seeds_json="[0]",
                  preset="fast"))
        s.add(Job(id="r1/j1", run_id="r1", job_id="j1", generator="ctgan", seed=0))
        s.commit()
        job = s.get(Job, "r1/j1")
        assert job.state == "queued" and job.error is None
        assert s.get(Dataset, "d1").registered is False
        assert s.get(Run, "r1").weights_json is None
