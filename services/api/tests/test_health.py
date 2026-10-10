def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "db": True, "store": True}


def test_health_reports_failures(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr("stl_api.main.get_engine", boom)
    monkeypatch.setattr("stl_api.main.store_from_env", boom)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "db": False, "store": False}
