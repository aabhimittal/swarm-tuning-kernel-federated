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

## Endpoints

| Method | Path          | Purpose                                        |
|--------|---------------|------------------------------------------------|
| GET    | `/health`     | liveness probe                                 |
| GET    | `/config`     | model + swarm config (workers self-verify)     |
| GET    | `/weights`    | download global weights (safetensors)          |
| POST   | `/gradients`  | upload gradients (safetensors) → aggregated    |
| GET    | `/status`     | version, step, pending, mode, last loss        |
| GET    | `/checkpoint` | download current weights as a file             |

## Configuration (Space → Settings → Variables and secrets)

| Variable                  | Default | Meaning                                       |
|---------------------------|---------|-----------------------------------------------|
| `SWARM_AGG_MODE`          | `sync`  | `sync` (average K then step) or `async`       |
| `SWARM_WORLD_SIZE`        | `4`     | gradients buffered per step in sync mode      |
| `SWARM_LR`                | `3e-4`  | AdamW learning rate                           |
| `SWARM_STALENESS_TOLERANCE` | `0`   | how many versions behind a grad may be        |
| `SWARM_TOKEN`             | _empty_ | if set, `/gradients` requires a bearer token  |

Set `SWARM_TOKEN` as a **secret** to stop random internet traffic from poisoning
your global model. Gradients are transported as **safetensors**, so uploads can
never execute code.
