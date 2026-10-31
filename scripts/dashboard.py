"""Live screen for everything that is running: pixi run dashboard   ->  http://127.0.0.1:8765

Shows every job that reports through robotics.status.StatusReporter (collection, DAgger, training, evaluation): progress,
ETA, key numbers, curves and the latest camera frames from the simulator, next to GPU / CPU / memory gauges.
Standard library only; the page polls /api every 1.5 s.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

from robotics.status import STATUS_DIR, list_runs

ROOT = Path(__file__).resolve().parents[1]
WATCHERS = {}  # run name -> 3D viewer process
RUNS_DIR = STATUS_DIR.parent
_hist = {"t": 0.0, "v": []}


def _thin(points, n=300):
    return points if len(points) <= n else points[:: -(-len(points) // n)] + [points[-1]]


def scan_history() -> list[dict]:
    """Every training run under runs/: curves, evaluations, saved validation samples, config. Cached for a few seconds."""
    if time.time() - _hist["t"] < 4.0:
        return _hist["v"]
    live = {r["name"]: r for r in list_runs() if r["kind"] == "train"}
    out = []
    for d in RUNS_DIR.iterdir() if RUNS_DIR.exists() else []:
        if not d.is_dir() or d.name == "status" or not ((d / "config.yaml").exists() or (d / "log.jsonl").exists()):
            continue
        try:
            cfg = yaml.safe_load((d / "config.yaml").read_text()) if (d / "config.yaml").exists() else {}
        except Exception:
            cfg = {}
        series = {"train_l1": {}, "val_l1": {}}
        last_step = 0
        if (d / "log.jsonl").exists():
            for line in (d / "log.jsonl").read_text().splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                for k in series:
                    if k in rec:
                        series[k][rec["step"]] = rec[k]  # a resumed run rewrites steps: the last value wins
                last_step = max(last_step, rec.get("step", 0))
        val = series["val_l1"]
        evals = []
        for f in sorted(d.glob("eval_*.json")):
            try:
                j = json.loads(f.read_text())
            except ValueError:
                continue
            parts = f.stem.split("_")
            evals.append({"file": f.name, "split": parts[1] if len(parts) > 1 else "?", "tag": "_".join(parts[2:]),
                          "step": j.get("step"), "episodes": j.get("episodes"), "closed_loop": j.get("closed_loop", {}),
                          "behaviour": j.get("behaviour"), "failure_breakdown": j.get("failure_breakdown"),
                          "wrong_block_rate": j.get("wrong_block_rate"), "mtime": f.stat().st_mtime})
        probe = None
        if (d / "language_probe.json").exists():
            try:
                probe = json.loads((d / "language_probe.json").read_text())
            except ValueError:
                pass
        views = []
        for f in sorted((d / "views").glob("*.jpg")) if (d / "views").exists() else []:
            m = re.match(r"(train|val)_(\d+|best)\.jpg$", f.name)
            if m:
                views.append({"file": f.name, "kind": m.group(1), "step": None if m.group(2) == "best" else int(m.group(2))})
        videos = None
        if (d / "videos" / "index.json").exists():
            try:
                videos = json.loads((d / "videos" / "index.json").read_text())
                videos["mtime"] = (d / "videos" / "index.json").stat().st_mtime
            except ValueError:
                pass
        ds = cfg.get("dataset", "")
        ds = ", ".join(ds) if isinstance(ds, list) else str(ds)
        lv = live.get(d.name)
        out.append({"name": d.name, "robot": "UR5e" if "ur5e" in ds else "SO-101", "text": bool(cfg.get("model", {}).get("text", True)),
                    "dataset": ds, "steps_cfg": cfg.get("steps"), "last_step": last_step,
                    "best_val": min(val.values()) if val else None, "running": bool(lv and lv["alive"]),
                    "mtime": max([(d / n).stat().st_mtime for n in ("config.yaml", "log.jsonl") if (d / n).exists()] or [0]),
                    "series": {k: _thin(sorted([[x, y] for x, y in v.items()])) for k, v in series.items()},
                    "evals": evals, "probe": probe, "views": views, "videos": videos, "config": cfg, "has_checkpoint": (d / "best.pt").exists()})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    _hist.update(t=time.time(), v=out)
    return out

PAGE_FILE = Path(__file__).with_name("dashboard.html")  # re-read on every request, so edits show up on reload


_gpu = {"t": 0.0, "v": None}


def gpu_stats():
    if time.time() - _gpu["t"] < 2.0:
        return _gpu["v"]
    _gpu["t"] = time.time()
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout.strip().splitlines()[0]
        name, util, used, total, temp = [x.strip() for x in out.split(",")]
        _gpu["v"] = (name, float(util), float(used), float(total), float(temp))
    except Exception:
        _gpu["v"] = None
    return _gpu["v"]


def system_stats():
    out = []
    g = gpu_stats()
    if g:
        out += [{"name": "GPU", "pct": g[1], "text": f"{g[1]:.0f}% · {g[4]:.0f}°C"},
                {"name": "GPU memory", "pct": 100 * g[2] / g[3], "text": f"{g[2] / 1024:.1f} / {g[3] / 1024:.1f} GB"}]
    ncpu = os.cpu_count() or 1
    load = os.getloadavg()[0]
    out.append({"name": "CPU load", "pct": 100 * load / ncpu, "text": f"{load:.1f} / {ncpu}"})
    try:
        mem = {k: int(v.split()[0]) for k, v in (line.split(":") for line in open("/proc/meminfo"))}
        used = mem["MemTotal"] - mem["MemAvailable"]
        out.append({"name": "Memory", "pct": 100 * used / mem["MemTotal"], "text": f"{used / 1048576:.1f} / {mem['MemTotal'] / 1048576:.1f} GB"})
    except Exception:
        pass
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the terminal quiet
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, f: Path, ctype: str):
        """Serve a file with HTTP Range support (browsers need it to loop and seek in videos)."""
        size = f.stat().st_size
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            try:
                start = int(a) if a else max(0, size - int(b))
                end = min(int(b), size - 1) if (a and b) else size - 1
                status = 206
            except ValueError:
                start, end, status = 0, size - 1, 200
        length = end - start + 1
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", "public, max-age=30")
            self.end_headers()
            with open(f, "rb") as fh:
                fh.seek(start)
                self.wfile.write(fh.read(length))
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser closed the connection (looping videos do this all the time)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/watch":
            return self._send(b"not found", "text/plain", 404)
        if self.client_address[0] not in ("127.0.0.1", "::1"):  # opening a window on this laptop is only for the laptop itself
            return self._send(json.dumps({"message": "3D windows can only be opened from the laptop itself"}).encode(), "application/json", 403)
        name = os.path.basename(urllib.parse.parse_qs(parsed.query).get("name", [""])[0])
        proc = WATCHERS.get(name)
        if proc is not None and proc.poll() is None:
            msg = "3D window is already open"
        elif not (STATUS_DIR / f"{name}.state.npz").exists():
            msg = "this run publishes no 3D state"
        else:
            WATCHERS[name] = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "watch.py"), "--run", name], cwd=ROOT,
                                              env={**os.environ, "MUJOCO_GL": "glfw"})
            msg = "3D window opened on your display"
        self._send(json.dumps({"message": msg}).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(PAGE_FILE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/history":
            self._send(json.dumps(scan_history()).encode(), "application/json")
        elif path.startswith("/loop/"):
            f = STATUS_DIR / os.path.basename(path[6:])
            if f.suffix == ".mp4" and f.is_file():
                self._send_file(f, "video/mp4")
            else:
                self._send(b"", "video/mp4", 404)
        elif path.startswith("/video/"):
            parts = path.split("/")
            f = RUNS_DIR / os.path.basename(parts[2]) / "videos" / os.path.basename(parts[3]) if len(parts) == 4 else None
            if f is not None and f.suffix == ".mp4" and f.is_file():
                self._send_file(f, "video/mp4")
            else:
                self._send(b"", "video/mp4", 404)
        elif path.startswith("/runimg/"):
            parts = path.split("/")
            f = RUNS_DIR / os.path.basename(parts[2]) / "views" / os.path.basename(parts[3]) if len(parts) == 4 else None
            if f is not None and f.suffix == ".jpg" and f.is_file():
                self._send(f.read_bytes(), "image/jpeg")
            else:
                self._send(b"", "image/jpeg", 404)
        elif path == "/api":
            self._send(json.dumps({"now": time.time(), "system": system_stats(), "runs": list_runs()}).encode(), "application/json")
        elif path.startswith("/img/"):
            name = os.path.basename(path[5:])
            f = STATUS_DIR / name
            if f.suffix == ".jpg" and f.is_file():
                self._send(f.read_bytes(), "image/jpeg")
            else:
                self._send(b"", "image/jpeg", 404)
        else:
            self._send(b"not found", "text/plain", 404)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open it from another device on your network")
    p.add_argument("--open", action="store_true", help="open the page in the default browser")
    a = p.parse_args()
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}"
    print(f"dashboard on {url}   (watching {STATUS_DIR}; Ctrl+C to stop)", flush=True)
    if a.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
