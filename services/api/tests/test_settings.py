import pytest
from stl_api.settings import Settings


def test_dispatch_default_and_valid(monkeypatch):
    monkeypatch.delenv("STL_DISPATCH", raising=False)
    assert Settings.from_env().dispatch == "celery"
    monkeypatch.setenv("STL_DISPATCH", "none")
    assert Settings.from_env().dispatch == "none"


def test_dispatch_invalid_raises(monkeypatch):
    monkeypatch.setenv("STL_DISPATCH", "bogus")
    with pytest.raises(ValueError, match="STL_DISPATCH"):
        Settings.from_env()


def test_time_cap_setting(monkeypatch):
    monkeypatch.delenv("STL_TIME_CAP_SECONDS", raising=False)
    assert Settings.from_env().time_cap_seconds == 7200.0
    monkeypatch.setenv("STL_TIME_CAP_SECONDS", "60")
    assert Settings.from_env().time_cap_seconds == 60.0
    for bad in ("abc", "0", "-5", "inf", "nan"):
        monkeypatch.setenv("STL_TIME_CAP_SECONDS", bad)
        with pytest.raises(ValueError, match="STL_TIME_CAP_SECONDS"):
            Settings.from_env()
