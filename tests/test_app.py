"""The dashboard renders every page on synthetic data, without raising.

A smoke test, not a test of numbers: every number on a page comes from a
``src`` function with its own tests.  What breaks in a dashboard restructure
is the wiring -- a renamed helper, a column a page expects and no longer gets,
a section that assumes a saved model exists -- and those fail only when the
page actually runs.  Streamlit's ``AppTest`` runs a page script in-process, so
the paths it reads are redirected by patching the module constants the app
looked them up from.  No network: the synthetic season's remaining calendar
is read from its own results.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.app_fixture import write_app_data  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = ["Home.py", *sorted(p.relative_to(APP).as_posix()
                            for p in (APP / "pages").glob("*.py"))]
TIMEOUT = 180


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("app_data")
    write_app_data(directory)
    return directory


def _redirect_defaults(module, old: Path, new: Path, monkeypatch) -> None:
    """Repoint every function default in ``module`` that captured ``old``.

    ``monitor`` and ``store`` bind their paths as default arguments at import,
    so patching the module constant alone leaves ``read_log()`` reading the
    real ``Reports/model_log.csv`` -- which made an earlier version of this
    test pass on the repository's own log rather than the scratch one.
    """
    import inspect

    for _, fn in inspect.getmembers(module, inspect.isfunction):
        if fn.__defaults__ and old in fn.__defaults__:
            monkeypatch.setattr(fn, "__defaults__",
                                tuple(new if d == old else d for d in fn.__defaults__))
        if fn.__kwdefaults__ and old in fn.__kwdefaults__.values():
            monkeypatch.setattr(fn, "__kwdefaults__",
                                {k: new if v == old else v
                                 for k, v in fn.__kwdefaults__.items()})


@pytest.fixture
def app_paths(data_dir: Path, tmp_path: Path, monkeypatch) -> Path:
    """Point every path the app reads at the synthetic data and a scratch dir."""
    import streamlit as st

    from src import config
    from src.models import monitor, store

    monkeypatch.setattr(config, "DNF_DATASET_PATH", data_dir / "dnf_dataset.parquet")
    monkeypatch.setattr(config, "RACE_RESULTS_PATH", data_dir / "race_results.parquet")
    monkeypatch.setattr(config, "CIRCUIT_PROFILE_PATH", data_dir / "circuit_profiles.parquet")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    for module, name, file in (
        (store, "MODEL_PATH", "dnf_model.joblib"),
        (store, "MANIFEST_PATH", "dnf_model.json"),
        (store, "PENDING_PATH", "pending.json"),
        (monitor, "LOG_PATH", "model_log.csv"),
        (monitor, "PREDICTIONS_PATH", "predictions.csv"),
    ):
        _redirect_defaults(module, getattr(module, name), tmp_path / file, monkeypatch)
        monkeypatch.setattr(module, name, tmp_path / file)
    # Caches outlive a test; one test's model must not leak into the next.
    st.cache_data.clear()
    st.cache_resource.clear()
    return tmp_path


def _save_attrition_model() -> None:
    import pandas as pd

    from src import config
    from src.models import predict, store

    dataset = pd.read_parquet(config.DNF_DATASET_PATH)
    estimator, manifest, _ = predict.fit_current(dataset, stage="post_quali")
    store.save_model(estimator, manifest)


def _run(page: str, **widgets) -> AppTest:
    at = AppTest.from_file(str(APP / page), default_timeout=TIMEOUT)
    at.run()
    for key, value in widgets.items():
        at.radio(key=key).set_value(value)
        at.run()
    return at


def _errors(at: AppTest) -> list[str]:
    return [str(e.value) for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_a_saved_model(app_paths, page) -> None:
    at = _run(page)
    assert not _errors(at), _errors(at)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_a_saved_model(app_paths, page) -> None:
    _save_attrition_model()
    at = _run(page)
    assert not _errors(at), _errors(at)


def test_the_race_page_backtests_a_completed_race(app_paths) -> None:
    _save_attrition_model()
    at = _run("pages/2_Race.py", race_mode="A completed race (backtest)")
    assert not _errors(at), _errors(at)
    assert any("Attrition stage" in s.value for s in at.subheader)


def test_the_track_record_reads_the_scratch_log_not_the_repository_one(app_paths) -> None:
    at = _run("pages/4_Track_record.py")
    assert not _errors(at), _errors(at)
    assert any("The log is empty" in i.value for i in at.info)


def test_the_sidebar_card_lists_every_stage(app_paths) -> None:
    _save_attrition_model()
    at = _run("Home.py")
    card = " ".join(c.value for c in at.sidebar.caption)
    for name in ("Attrition (DNF)", "Finishing order", "pre_weekend",
                 "post_quali", "Season noise"):
        assert name in card
