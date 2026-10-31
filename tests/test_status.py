import json
import os
import time

import numpy as np

import robotics.status as st


def test_reporter_writes_state_preview_and_lists_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "STATUS_DIR", tmp_path)
    rep = st.StatusReporter("train", "unit-run", total=100, min_interval=0.0)
    rep.update(progress=40, loss=0.5)
    rep.series("loss", 0.5, x=40)
    rep.preview(np.full((16, 32, 3), 120, np.uint8), min_interval=0.0)
    runs = st.list_runs()
    assert len(runs) == 1
    r = runs[0]
    assert r["alive"] and r["status"] == "running" and r["progress"] == 40 and r["metrics"]["loss"] == 0.5
    assert r["eta"] is not None and r["preview_t"] and (tmp_path / "unit-run.jpg").exists()
    rep.finish("done", final=1.0)
    r = st.list_runs()[0]
    assert not r["alive"] and r["status"] == "done" and r["progress"] == 100 and r["eta"] is None


def test_run_of_a_dead_process_is_reported_as_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "STATUS_DIR", tmp_path)
    now = time.time()
    state = {"kind": "collect", "name": "ghost", "pid": 2 ** 22 + 12345, "started": now - 30, "updated": now - 20, "finished": None,
             "status": "running", "progress": 5.0, "total": 10, "metrics": {}, "series": {}, "meta": {}, "preview_t": None}
    (tmp_path / "ghost.json").write_text(json.dumps(state))
    r = st.list_runs()[0]
    assert not r["alive"] and r["status"] == "stopped"


def test_reporter_never_raises_when_the_directory_is_unusable(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(st, "STATUS_DIR", blocker / "sub")  # cannot be created
    rep = st.StatusReporter("eval", "broken")
    rep.update(progress=1)
    rep.preview(np.zeros((4, 4, 3), np.uint8), min_interval=0.0)
    rep.finish()
