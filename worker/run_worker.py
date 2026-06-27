"""CLI entrypoint for a swarm worker.

Examples
--------
    # against a local server
    python -m worker.run_worker --server http://localhost:7860 --worker-id w0 --steps 50

    # against a live Hugging Face Space, as one shard of a 4-way split.
    # NOTE: use --token=VALUE (the '=' form) or the SWARM_TOKEN env var — a bare
    # "--token VALUE" breaks if the token starts with '-' (argparse reads it as a flag).
    export SWARM_TOKEN=...   # or pass --token="$SWARM_TOKEN"
    python -m worker.run_worker \
        --server https://USER-swarm-server.hf.space \
        --worker-id kaggle-1 --shard 0 --num-shards 4 --steps 200
"""

from __future__ import annotations

import argparse
import os

from worker.client import SwarmClient


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a federated swarm-tuning worker.")
    p.add_argument(
        "--server",
        default=os.environ.get("SWARM_SERVER", "http://localhost:7860"),
        help="Parameter server base URL (HF Space URL in production).",
    )
    p.add_argument("--worker-id", default=os.environ.get("SWARM_WORKER_ID", "worker"))
    p.add_argument("--steps", type=int, default=100, help="Number of pull/train/push iterations.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--shard", type=int, default=0, help="This worker's data shard index.")
    p.add_argument("--num-shards", type=int, default=1, help="Total number of data shards.")
    p.add_argument(
        "--token",
        default=os.environ.get("SWARM_TOKEN", ""),
        help="Bearer token for the Space. Prefer the SWARM_TOKEN env var, or the "
        "--token=VALUE form — a token starting with '-' breaks bare '--token VALUE'.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    client = SwarmClient(
        server_url=args.server,
        worker_id=args.worker_id,
        batch_size=args.batch_size,
        shard=args.shard,
        num_shards=args.num_shards,
        token=args.token,
        seed=args.seed,
    )
    print(
        f"[{args.worker_id}] connected to {args.server} | "
        f"model params={client.model.num_params():,} | "
        f"vocab={client.model_cfg.vocab_size} | shard {args.shard}/{args.num_shards}"
    )
    try:
        client.run(steps=args.steps, verbose=not args.quiet)
    except KeyboardInterrupt:
        print(f"[{args.worker_id}] interrupted")
    finally:
        client.close()


if __name__ == "__main__":
    main()
