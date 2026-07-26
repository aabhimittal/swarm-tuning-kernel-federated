"""Local end-to-end proof of the swarm — no HF/Kaggle accounts needed.

Spins up the FastAPI parameter server + N worker processes on this machine, then
watches the global model's version and loss advance over real HTTP. This is the
CPU proof that the "gradient accumulation over HTTP" interconnect actually trains.

    python scripts/simulate_swarm.py --workers 4 --steps 60 --mode sync
    python scripts/simulate_swarm.py --workers 4 --steps 80 --mode async

    # 50x less bandwidth via Top-K sparsification with error feedback
    python scripts/simulate_swarm.py --workers 4 --steps 60 --compression topk

    # 1/5th the round trips via local steps (FedAvg-style)
    python scripts/simulate_swarm.py --workers 4 --steps 20 --local-steps 5

    # Red-team: one worker submits reversed gradients. `mean` follows it and
    # stops learning; the coordinate-wise rules out-vote it.
    python scripts/simulate_swarm.py --workers 4 --byzantine 1 --rule mean
    python scripts/simulate_swarm.py --workers 4 --byzantine 1 --rule median
"""

from __future__ import annotations

import argparse
import math
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


def start_server(
    port: int,
    mode: str,
    world_size: int,
    log_path: str,
    rule: str = "mean",
    grad_clip: float = 1.0,
) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(
        {
            "SWARM_AGG_MODE": mode,
            "SWARM_AGG_RULE": rule,
            "SWARM_GRAD_CLIP": str(grad_clip),
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
    server_url: str,
    n: int,
    steps: int,
    batch_size: int,
    log_dir: str,
    compression: str = "none",
    local_steps: int = 1,
    byzantine: int = 0,
) -> List[subprocess.Popen]:
    procs = []
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    for i in range(n):
        # The last `byzantine` workers are the attackers.
        is_bad = i >= n - byzantine
        log = open(os.path.join(log_dir, f"worker_{i}.log"), "w")
        cmd = [
            sys.executable, "-m", "worker.run_worker",
            "--server", server_url,
            "--worker-id", ("evil" if is_bad else "w") + str(i),
            "--shard", str(i),
            "--num-shards", str(n),
            "--steps", str(steps),
            "--batch-size", str(batch_size),
            "--seed", str(1000 + i),
            "--compression", compression,
            "--local-steps", str(local_steps),
        ]
        if is_bad:
            # Scale the reversed gradient so the attackers outweigh the honest
            # majority in a plain mean: b*scale > (n-b). Anything less and even
            # `mean` shrugs it off, which would make the demo prove nothing.
            scale = 2.0 * (n - byzantine) / byzantine
            cmd += ["--byzantine", str(scale)]
        procs.append(
            subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        )
    return procs


def main() -> int:
    p = argparse.ArgumentParser(description="Local multi-worker swarm simulation.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=60, help="Iterations per worker.")
    p.add_argument("--mode", choices=["sync", "async"], default="sync")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--rule", choices=["mean", "median", "trimmed_mean", "krum"], default="mean"
    )
    p.add_argument("--compression", choices=["none", "topk", "topk-int8"], default="none")
    p.add_argument("--local-steps", type=int, default=1)
    p.add_argument(
        "--byzantine",
        type=int,
        default=0,
        help="How many of the workers upload poisoned gradients (red-team test).",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Server-side gradient-norm clip; <=0 disables. Clipping is itself a "
        "partial defence (it bounds a poisoned update's magnitude), so disable it "
        "to see plain `mean` actually break.",
    )
    args = p.parse_args()
    if args.byzantine >= args.workers:
        p.error("--byzantine must be fewer than --workers")

    server_url = f"http://localhost:{args.port}"
    os.makedirs("/tmp/swarm_logs", exist_ok=True)
    server_log = "/tmp/swarm_logs/server.log"

    print(
        f"== Swarm simulation: mode={args.mode} rule={args.rule} workers={args.workers} "
        f"steps={args.steps} compression={args.compression} local_steps={args.local_steps}"
        + (f" byzantine={args.byzantine}" if args.byzantine else "")
        + " =="
    )
    server = start_server(
        args.port, args.mode, args.workers, server_log, rule=args.rule,
        grad_clip=args.grad_clip,
    )
    try:
        wait_for_health(server_url)
        cfg = httpx.get(server_url + "/config").json()
        print(f"server up | model={cfg['model']} | mode={cfg['mode']} | rule={cfg['rule']}")

        workers = start_workers(
            server_url,
            args.workers,
            args.steps,
            args.batch_size,
            "/tmp/swarm_logs",
            compression=args.compression,
            local_steps=args.local_steps,
            byzantine=args.byzantine,
        )

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
        summary = httpx.get(server_url + "/workers").json().get("summary", {})

        print("\n== Result ==")
        print(f"global version : {final['version']}")
        print(f"optimizer steps: {final['step']}")
        print(f"first loss     : {first_loss}")
        print(f"final loss     : {final_loss}")
        if args.compression != "none":
            print(
                f"bandwidth      : {summary.get('wire_bytes', 0):,} B sent vs "
                f"{summary.get('dense_bytes', 0):,} B dense "
                f"({summary.get('compression_ratio', 1)}x smaller)"
            )

        if args.byzantine:
            # The attackers submit reversed gradients, so the question is whether
            # the model still *learns*, not whether the weights exploded. (An
            # inflated-magnitude attack is a non-event here: clipping rescales it
            # and AdamW normalises by the second moment, so the step stays ~lr.)
            norm = final.get("weight_norm")
            print(f"weight norm    : {norm}")
            survived = (
                final.get("healthy")
                and first_loss is not None
                and final_loss is not None
                and final_loss < first_loss
            )
            print(
                "RESULT:",
                f"SURVIVED ✅ {args.rule} kept learning through "
                f"{args.byzantine} poisoned worker(s)"
                if survived
                else f"POISONED ❌ {args.rule} stopped learning under "
                f"{args.byzantine} poisoned worker(s)",
            )
            return 0 if survived else 1

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
