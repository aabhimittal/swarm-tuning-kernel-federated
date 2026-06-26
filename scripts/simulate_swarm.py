"""Local end-to-end proof of the swarm — no HF/Kaggle accounts needed.

Spins up the FastAPI parameter server + N worker processes on this machine, then
watches the global model's version and loss advance over real HTTP. This is the
CPU proof that the "gradient accumulation over HTTP" interconnect actually trains.

    python scripts/simulate_swarm.py --workers 4 --steps 60 --mode sync
    python scripts/simulate_swarm.py --workers 4 --steps 80 --mode async
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional

import httpx

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def wait_for_health(url: str, timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            r = httpx.get(url + "/health", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(0.3)
    raise RuntimeError(f"server did not become healthy at {url}: {last_err}")


def start_server(port: int, mode: str, world_size: int, log_path: str) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(
        {
            "SWARM_AGG_MODE": mode,
            "SWARM_WORLD_SIZE": str(world_size),
            "SWARM_PORT": str(port),
            # In async mode, workers race: tolerate staleness up to the swarm size so
            # concurrent pushes land instead of being rejected. Sync stays strict.
            "SWARM_STALENESS_TOLERANCE": str(world_size if mode == "async" else 0),
            "SWARM_CHECKPOINT": "",          # ephemeral for the demo
            "PYTHONPATH": REPO_ROOT + os.pathsep + env.get("PYTHONPATH", ""),
            "PYTHONUNBUFFERED": "1",
        }
    )
    log = open(log_path, "w")
    return subprocess.Popen(
        [sys.executable, "-m", "server.app"],
        cwd=REPO_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def start_workers(
    server_url: str, n: int, steps: int, batch_size: int, log_dir: str
) -> List[subprocess.Popen]:
    procs = []
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    for i in range(n):
        log = open(os.path.join(log_dir, f"worker_{i}.log"), "w")
        procs.append(
            subprocess.Popen(
                [
                    sys.executable, "-m", "worker.run_worker",
                    "--server", server_url,
                    "--worker-id", f"w{i}",
                    "--shard", str(i),
                    "--num-shards", str(n),
                    "--steps", str(steps),
                    "--batch-size", str(batch_size),
                    "--seed", str(1000 + i),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )
    return procs


def main() -> int:
    p = argparse.ArgumentParser(description="Local multi-worker swarm simulation.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=60, help="Iterations per worker.")
    p.add_argument("--mode", choices=["sync", "async"], default="sync")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    server_url = f"http://localhost:{args.port}"
    os.makedirs("/tmp/swarm_logs", exist_ok=True)
    server_log = "/tmp/swarm_logs/server.log"

    print(f"== Swarm simulation: mode={args.mode} workers={args.workers} steps={args.steps} ==")
    server = start_server(args.port, args.mode, args.workers, server_log)
    try:
        wait_for_health(server_url)
        cfg = httpx.get(server_url + "/config").json()
        print(f"server up | model={cfg['model']} | mode={cfg['mode']}")

        workers = start_workers(server_url, args.workers, args.steps, args.batch_size, "/tmp/swarm_logs")

        history = []  # (version, last_loss)
        first_loss = None
        while any(w.poll() is None for w in workers):
            try:
                st = httpx.get(server_url + "/status", timeout=3.0).json()
                loss = st.get("last_loss")
                if loss is not None and loss == loss:  # not NaN
                    if first_loss is None:
                        first_loss = loss
                    if not history or history[-1][0] != st["version"]:
                        history.append((st["version"], loss))
                        print(f"  v={st['version']:>4}  step={st['step']:>4}  loss={loss:.4f}")
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)

        for w in workers:
            w.wait()

        final = httpx.get(server_url + "/status").json()
        final_loss = final.get("last_loss")
        print("\n== Result ==")
        print(f"global version : {final['version']}")
        print(f"optimizer steps: {final['step']}")
        print(f"first loss     : {first_loss}")
        print(f"final loss     : {final_loss}")

        ok = (
            first_loss is not None
            and final_loss is not None
            and final_loss < first_loss
            and final["version"] > 0
        )
        print("RESULT:", "PASS ✅ loss decreased over HTTP" if ok else "FAIL ❌")
        return 0 if ok else 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        print(f"(server log: {server_log})")


if __name__ == "__main__":
    raise SystemExit(main())
