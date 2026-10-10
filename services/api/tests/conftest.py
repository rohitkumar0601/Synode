import pytest
from fastapi.testclient import TestClient
from stl_api import db


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'api.db').as_posix()}")
    monkeypatch.setenv("STL_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("STL_DISPATCH", "none")
    db.get_engine.cache_clear()
    from stl_api.main import create_app

    with TestClient(create_app()) as c:
        yield c
    db.get_engine.cache_clear()
