# 🐝 Federated "Swarm" Tuning — Gradient Accumulation over HTTP

Train a model without a single big GPU by splitting the work across a **swarm** of
free, ephemeral instances. A **Hugging Face Space** acts as the *Parameter Server*
holding the global weights; **Kaggle notebooks** (or friends' machines, or local
processes) act as *Workers* that pull the weights, train on a micro-batch, and
push back **only the gradients**.

The server averages the gradients and takes one optimizer step. The global batch
is the *sum* of every worker's micro-batch — mathematically the same as one large
batch on one big GPU. We've replaced the NVLink hardware interconnect with a
**software interconnect over HTTP**. That's the kernel innovation.

```
        ┌──────────────────────── Hugging Face Space ────────────────────────┐
        │  Parameter Server (FastAPI)                                         │
        │  • holds global TinyGPT weights + AdamW                             │
        │  • GET /weights   POST /gradients   GET /status                     │
        │  • averages K gradients  ->  one optimizer step  ->  version += 1   │
        └─────▲───────────────▲───────────────▲───────────────▲──────────────┘
              │ weights/grads  │               │               │   (HTTP)
        ┌─────┴─────┐   ┌──────┴────┐   ┌──────┴────┐   ┌──────┴────┐
        │ Worker 0  │   │ Worker 1  │   │ Worker 2  │   │ Worker 3  │   ...
        │ (Kaggle)  │   │ (Kaggle)  │   │ (friend)  │   │ (local)   │
        │ pull→grad │   │ pull→grad │   │ pull→grad │   │ pull→grad │
        └───────────┘   └───────────┘   └───────────┘   └───────────┘
```

## Why this lets you train "bigger than one free GPU"

A worker only ever holds **one** model replica and **one** micro-batch — no
optimizer state, no second copy, no full-dataset shard. Peak worker memory is
tiny and constant. Scale comes from *adding workers*, not from a bigger GPU. The
server keeps the authoritative weights and the optimizer; workers are stateless
gradient producers.

## Repository layout

| Path | What it is |
|------|------------|
| `swarm/` | shared core imported by **both** sides: `model.py` (TinyGPT), `data.py`, `protocol.py` (safetensors wire format), `compression.py` (Top-K + error feedback), `config.py` |
| `server/` | parameter server: `app.py` (FastAPI), `aggregator.py` (sync/async), `robust.py` (Byzantine-robust rules), `telemetry.py`, `dashboard.py`, `state.py` |
| `worker/` | `client.py` (`SwarmClient`) + `run_worker.py` (CLI) |
| `scripts/simulate_swarm.py` | local end-to-end proof: 1 server + N workers on this box |
| `deploy/` | `Dockerfile`, HF Space card, `deploy_hf_space.py` (one-command deploy) |
| `notebooks/kaggle_worker.ipynb` | ready-to-run Kaggle worker |
| `tests/` | protocol, compression, robustness, aggregator, and full-stack API tests |

## The three bottlenecks of an HTTP interconnect

Replacing NVLink with HTTP creates three problems a GPU cluster never has. Each
gets a dedicated piece of the kernel.

### 1. Bandwidth — `swarm/compression.py`

NVLink moves ~600 GB/s; a free Kaggle box moves maybe 10 MB/s. Uploading a dense
gradient every step means the worker is ~97% idle waiting on the network.

* **Top-K sparsification** sends only the largest-magnitude `--compress-ratio`
  fraction of entries (default 1%) as index/value pairs.
* **Error feedback** keeps everything that was *not* sent in a local residual and
  folds it into the next step, so a small persistent gradient accumulates until
  it is big enough to transmit. This is what makes aggressive sparsification
  safe; the delivered mass satisfies `sum(sent) == n·grad − residual` exactly.
* **int8 quantization** (`--compression topk-int8`) shrinks the surviving values
  another ~4x with a per-tensor scale.

```bash
python scripts/simulate_swarm.py --workers 4 --steps 60 --compression topk
```

### 2. Latency — local steps (FedAvg / FedOpt)

A round trip to a Space costs far more than a forward/backward pass. `--local-steps N`
runs N local SGD steps and uploads the accumulated **parameter delta** as a
pseudo-gradient, which the server feeds to its own AdamW. That is FedAvg with a
server optimizer, and it cuts round trips by N×.

```bash
python scripts/simulate_swarm.py --workers 4 --steps 20 --local-steps 5
```

### 3. Trust — `server/robust.py`

The parameter server is a public URL, so anyone with the token can push. Plain
averaging has a **breakdown point of zero**: one worker sending a single huge
tensor moves the global mean arbitrarily far and destroys the model.

| `SWARM_AGG_RULE` | Behaviour |
|---|---|
| `mean` | standard accumulation — fastest, zero robustness |
| `median` | coordinate-wise median; tolerates <50% attackers |
| `trimmed_mean` | drop the extremes per coordinate, average the rest |
| `krum` | keep only the most consensual gradient, discard the rest |

`trimmed_mean` and `krum` tolerate `int(world_size * SWARM_TRIM_RATIO)` attackers;
`median` tolerates up to half the swarm. Size `SWARM_TRIM_RATIO` to the fraction
of contributors you are willing to distrust.

**Magnitude attacks are a non-event here — direction attacks are the threat.**
Measured on this stack: a worker uploading a 1e6-magnitude gradient fails to move
the model at all, even with `--rule mean` *and* `--grad-clip 0`. Two mechanisms
independently absorb it — `SWARM_GRAD_CLIP` rescales the aggregated update, and
AdamW divides by the running second moment, so the step stays bounded by ~`lr`
however large the upload.

Neither mechanism looks at *direction*. A minority submitting correctly-scaled
but **reversed** gradients steers the global model while never tripping a norm
bound, and Adam happily follows the poisoned direction at full step size. That is
what the coordinate-wise rules are for, and it is what the red-team mode
simulates:

```bash
python scripts/simulate_swarm.py --workers 4 --byzantine 1 --rule mean    # POISONED
python scripts/simulate_swarm.py --workers 4 --byzantine 1 --rule median  # SURVIVED
```

`--byzantine N` turns the last N workers into attackers that negate their real
gradient (and report their honest loss, so they look legitimate in telemetry).

## Live dashboard

The Space serves a self-contained dashboard at `/` — global loss curve, the
contributor leaderboard, bandwidth saved by compression, model health, and a box
to sample text from the current global model. No CDN, no build step.

Extra endpoints behind it: `/workers` (per-worker ledger), `/history` (loss
series), `/generate?prompt=…&tokens=…` (sample the global model). `/status` also
reports `weight_norm` and `healthy`, which is the honest liveness signal — a
poisoned update explodes the norm long before the reported loss reflects it.

## Quickstart (local, CPU)

```bash
pip install -r requirements.txt

# Prove the whole interconnect end-to-end: server + 4 workers, asserts loss drops.
python scripts/simulate_swarm.py --workers 4 --steps 60 --mode sync
python scripts/simulate_swarm.py --workers 4 --steps 80 --mode async
```

Or run the pieces by hand:

```bash
# Terminal 1 — parameter server
SWARM_AGG_MODE=sync SWARM_WORLD_SIZE=2 uvicorn server.app:app --port 7860

# Terminals 2 & 3 — two workers, each its own data shard
python -m worker.run_worker --server http://localhost:7860 --worker-id w0 --shard 0 --num-shards 2 --steps 100
python -m worker.run_worker --server http://localhost:7860 --worker-id w1 --shard 1 --num-shards 2 --steps 100

# Watch the global model advance
curl -s http://localhost:7860/status
```

## Deploy to Hugging Face + Kaggle

```bash
# 1) Deploy the parameter server to a Space (needs an HF *write* token)
export HF_TOKEN=hf_xxx
python deploy/deploy_hf_space.py --repo-id YOUR_USERNAME/swarm-server

# 2) Open notebooks/kaggle_worker.ipynb on Kaggle, set SERVER_URL to the Space,
#    and run it. Launch several copies (yours + friends') to grow the swarm.
```

Set a `SWARM_TOKEN` secret on the Space to require a bearer token on `/gradients`
so random internet traffic can't poison your global model. Gradients travel as
**safetensors** (no pickle), so uploads can never execute code.

## Aggregation modes

* **`sync`** (default) — buffer `SWARM_WORLD_SIZE` gradient sets at the current
  version, average element-wise, take one optimizer step, bump the version. Exact
  gradient accumulation across the swarm.
* **`async`** — apply each gradient as it arrives, scaled by staleness
  `1 / (1 + age)`. Higher throughput, noisier, no barrier.

Configured via env vars (see the table in `deploy/README_SPACE.md`).

## Tests

```bash
pip install pytest
pytest -q
```

## How it works (the kernel)

1. **Worker** `GET /weights` → loads the global `state_dict` (safetensors).
2. **Worker** runs one forward/backward on a micro-batch → `{param_name: grad}`.
3. **Worker** `POST /gradients` with the model version it trained against.
4. **Server** rejects stale versions (configurable tolerance), then either buffers
   (sync) or applies immediately (async). When a step fires, it loads the averaged
   grads onto the params, clips, `optimizer.step()`, and increments the version.

All model mutation on the server happens under a single lock, so concurrent
uploads from the whole swarm are serialized and can never corrupt the weights.
