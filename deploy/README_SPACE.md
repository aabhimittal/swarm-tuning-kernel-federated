---
title: Swarm Parameter Server
emoji: 🐝
colorFrom: yellow
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🐝 Swarm Parameter Server

The **Parameter Server** for [federated "swarm" tuning](https://github.com/aabhimittal/swarm-tuning-kernel-federated).
It holds the global TinyGPT weights and averages gradients pushed over HTTP by a
swarm of workers (Kaggle notebooks, friends' machines, anything that can `POST`).

Open the Space to see a **live dashboard**: the global loss curve, the swarm
contributor leaderboard, bandwidth saved by gradient compression, and a box to
sample text from the model as it trains.

## Endpoints

| Method | Path          | Purpose                                          |
|--------|---------------|--------------------------------------------------|
| GET    | `/`           | live dashboard                                   |
| GET    | `/health`     | liveness probe                                   |
| GET    | `/config`     | model + swarm config (workers self-verify)       |
| GET    | `/weights`    | download global weights (safetensors)            |
| POST   | `/gradients`  | upload gradients (optionally Top-K compressed)   |
| GET    | `/status`     | version, step, pending, mode, rule, weight norm  |
| GET    | `/workers`    | per-worker contribution ledger + bandwidth stats |
| GET    | `/history`    | rolling (version, loss) series                   |
| GET    | `/generate`   | sample text from the current global model        |
| GET    | `/checkpoint` | download current weights as a file               |

## Configuration (Space → Settings → Variables and secrets)

| Variable                  | Default | Meaning                                       |
|---------------------------|---------|-----------------------------------------------|
| `SWARM_AGG_MODE`          | `sync`  | `sync` (average K then step) or `async`       |
| `SWARM_AGG_RULE`          | `mean`  | `mean`, `median`, `trimmed_mean`, `krum`      |
| `SWARM_TRIM_RATIO`        | `0.25`  | fraction trimmed per end / assumed attackers  |
| `SWARM_WORLD_SIZE`        | `4`     | gradients buffered per step in sync mode      |
| `SWARM_LR`                | `3e-4`  | AdamW learning rate                           |
| `SWARM_STALENESS_TOLERANCE` | `0`   | how many versions behind a grad may be        |
| `SWARM_TOKEN`             | _empty_ | if set, `/gradients` requires a bearer token  |

**Running an open swarm?** Set `SWARM_AGG_RULE` to `trimmed_mean` or `median`.
`SWARM_GRAD_CLIP` already bounds how *large* any single update can be, but it
cannot tell a helpful direction from a hostile one — a minority pushing modest,
correctly-scaled but reversed gradients will still steer the global model.
Coordinate-wise rules decide each entry by majority instead.

Set `SWARM_TOKEN` as a **secret** to stop random internet traffic from poisoning
your global model. Gradients are transported as **safetensors**, so uploads can
never execute code.
