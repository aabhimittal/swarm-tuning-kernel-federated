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

from swarm import compression as C
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
    p.add_argument(
        "--compression",
        choices=list(C.MODES),
        default=os.environ.get("SWARM_COMPRESSION", C.NONE),
        help="Top-K sparsify gradients before upload (with error feedback). "
        "'topk-int8' additionally quantizes the surviving values.",
    )
    p.add_argument(
        "--compress-ratio",
        type=float,
        default=0.01,
        help="Fraction of gradient entries to transmit under Top-K (default 1%%).",
    )
    p.add_argument(
        "--local-steps",
        type=int,
        default=1,
        help="Local optimizer steps per round. >1 uploads the accumulated parameter "
        "delta (FedAvg-style), cutting round trips by this factor.",
    )
    p.add_argument("--local-lr", type=float, default=1e-3, help="LR for local steps.")
    p.add_argument(
        "--byzantine",
        type=float,
        default=0.0,
        metavar="MAGNITUDE",
        help="Red-team mode: upload garbage of this magnitude instead of a real "
        "gradient, to verify your own swarm's robust aggregation holds. 0 = off.",
    )
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
        compression=args.compression,
        compress_ratio=args.compress_ratio,
        local_steps=args.local_steps,
        local_lr=args.local_lr,
        byzantine=args.byzantine,
    )
    print(
        f"[{args.worker_id}] connected to {args.server} | "
        f"model params={client.model.num_params():,} | "
        f"vocab={client.model_cfg.vocab_size} | shard {args.shard}/{args.num_shards} | "
        f"compression={args.compression} | local_steps={args.local_steps}"
    )
    try:
        client.run(steps=args.steps, verbose=not args.quiet)
    except KeyboardInterrupt:
        print(f"[{args.worker_id}] interrupted")
    finally:
        client.close()


if __name__ == "__main__":
    main()
