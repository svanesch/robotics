"""Heartbeat files for long-running jobs, shown live by scripts/dashboard.py (`pixi run dashboard`).

A job creates a StatusReporter, calls update() with its progress and numbers, preview() with a camera frame now and then,
and finish() at the end. Everything is written as small files under runs/status/ (atomically, throttled). Reporting never
raises: a broken dashboard must not break a run.
"""
from __future__ import annotations

import atexit
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

from robotics.robots.spec import REPO_ROOT

STATUS_DIR = Path(os.environ.get("ROBOTICS_STATUS_DIR", REPO_ROOT / "runs" / "status"))
SERIES_MAX = 300  # points kept per curve (older points are thinned out)
FINISHED_KEEP = 24  # finished runs the dashboard lists


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name).strip(".") or "run"


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class StatusReporter:
    def __init__(self, kind: str, name: str, total: float | None = None, meta: dict | None = None, min_interval: float = 0.5):
        self.name, self.min_interval = _safe(name), min_interval
        now = time.time()
        self.state = {"kind": kind, "name": self.name, "pid": os.getpid(), "started": now, "updated": now, "finished": None,
                      "status": "running", "progress": 0.0, "progress0": 0.0, "total": total, "metrics": {}, "series": {}, "meta": meta or {},
                      "preview_t": None, "previews": {}, "loop_t": None}
        self._last_write = 0.0
        self._last_preview: dict[str, float] = {}
        self._last_snap = 0.0
        self._ring: deque = deque(maxlen=40)  # the last few seconds of the live view, replayed in a loop when the feed stops
        self._last_loop = 0.0
        self._ok = True
        try:
            STATUS_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._ok = False
        self._flush(force=True)
        atexit.register(self._on_exit)

    # ------------------------------------------------------------------ api
    def update(self, progress: float | None = None, total: float | None = None, **metrics) -> None:
        if progress is not None:
            self.state["progress"] = float(progress)
        if total is not None:
            self.state["total"] = float(total)
        self.state["metrics"].update({k: v for k, v in metrics.items() if v is not None})
        self._flush()

    def series(self, key: str, value: float, x: float | None = None) -> None:
        """Append a point to a curve (e.g. the training loss)."""
        s = self.state["series"].setdefault(key, [])
        s.append([float(self.state["progress"] if x is None else x), float(value)])
        if len(s) > 2 * SERIES_MAX:
            self.state["series"][key] = s[::2]

    def preview(self, image: np.ndarray, min_interval: float = 0.2, key: str = "live") -> None:
        """image: (H,W,3) uint8. key "live" is the streaming camera view; other keys ("train", "val", ...) are extra figures."""
        now = time.time()
        if not self._ok or now - self._last_preview.get(key, 0.0) < min_interval:
            return
        self._last_preview[key] = now
        try:
            import imageio.v2 as imageio

            fname = f"{self.name}.jpg" if key == "live" else f"{self.name}.{_safe(key)}.jpg"
            tmp = STATUS_DIR / f"{self.name}.{_safe(key)}.tmp.jpg"
            imageio.imwrite(tmp, np.ascontiguousarray(image), quality=88)
            os.replace(tmp, STATUS_DIR / fname)
            self.state["previews"][key] = now
            if key == "live":
                h, w = image.shape[:2]
                self._ring.append(np.ascontiguousarray(image[: h - h % 2, : w - w % 2]).copy())
                self.state["preview_t"] = now
                if now - self._last_loop > 20.0 and len(self._ring) >= 8:
                    self._write_loop()
            self._flush(force=True)  # publish the new frame time right away
        except Exception:
            pass

    def snapshot(self, model, data, min_interval: float = 0.05) -> None:
        """Publish the simulated world (poses + the per-episode model parameters) so `pixi run watch` can mirror it in 3D."""
        now = time.time()
        if not self._ok or now - self._last_snap < min_interval:
            return
        self._last_snap = now
        try:
            tmp = STATUS_DIR / f"{self.name}.tmp.npz"
            np.savez(tmp, qpos=data.qpos, geom_size=model.geom_size, geom_rgba=model.geom_rgba, geom_rbound=model.geom_rbound,
                     cam_pos=model.cam_pos, cam_quat=model.cam_quat, light_pos=model.light_pos, light_diffuse=model.light_diffuse)
            os.replace(tmp, STATUS_DIR / f"{self.name}.state.npz")
        except Exception:
            pass

    def _write_loop(self) -> None:
        """Short mp4 of the last frames; the dashboard plays it in a loop once the live feed has stopped."""
        self._last_loop = time.time()
        try:
            import imageio.v2 as imageio

            tmp = STATUS_DIR / f"{self.name}.loop.tmp.mp4"
            imageio.mimsave(tmp, list(self._ring), fps=8, codec="libx264", macro_block_size=1, quality=6, format="mp4")
            os.replace(tmp, STATUS_DIR / f"{self.name}.loop.mp4")
            self.state["loop_t"] = self._last_loop
        except Exception:
            pass

    def finish(self, status: str = "done", **metrics) -> None:
        self.state["metrics"].update({k: v for k, v in metrics.items() if v is not None})
        if len(self._ring) >= 4:
            self._write_loop()
        self.state["status"], self.state["finished"] = status, time.time()
        if self.state["total"] and status == "done":
            self.state["progress"] = self.state["total"]
        self._flush(force=True)

    # ------------------------------------------------------------- internals
    def _on_exit(self) -> None:
        if self.state["status"] == "running":
            self.finish("stopped")

    def _flush(self, force: bool = False) -> None:
        now = time.time()
        if not self._ok or (not force and now - self._last_write < self.min_interval):
            return
        self._last_write = now
        self.state["updated"] = now
        try:
            _write_atomic(STATUS_DIR / f"{self.name}.json", json.dumps(self.state, default=_jsonable).encode())
        except Exception:
            pass


# --------------------------------------------------------------------------- reading
def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def list_runs(now: float | None = None) -> list[dict]:
    """All known runs, running ones first. Adds `alive`, `age` (s since the last heartbeat), `rate` and `eta` (s)."""
    now = now or time.time()
    runs = []
    for p in STATUS_DIR.glob("*.json"):
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        r["alive"] = r["status"] == "running" and _alive(r["pid"])
        r["has_state"] = (STATUS_DIR / f"{r['name']}.state.npz").exists()
        if r["status"] == "running" and not r["alive"]:
            r["status"] = "stopped"
        r["age"] = now - r["updated"]
        end = r["finished"] or now
        elapsed = max(end - r["started"], 1e-6)
        r["elapsed"] = elapsed
        done_here = r["progress"] - r.get("progress0", 0.0)  # a resumed job did not do the steps before it started
        r["rate"] = done_here / elapsed if done_here > 0 else 0.0
        total = r.get("total")
        r["eta"] = (total - r["progress"]) / r["rate"] if r["alive"] and total and r["rate"] > 0 else None
        runs.append(r)
    running = sorted((r for r in runs if r["alive"]), key=lambda r: r["started"], reverse=True)
    rest = sorted((r for r in runs if not r["alive"]), key=lambda r: r["updated"], reverse=True)[:FINISHED_KEEP]
    return running + rest
