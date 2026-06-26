"""FastAPI parameter server — the swarm's "Parameter Server" on a Hugging Face Space.

Endpoints
---------
GET  /health      liveness probe
GET  /config      model + swarm config (so a worker can self-verify it matches)
GET  /weights     current global weights (safetensors) + X-Model-Version header
POST /gradients   upload gradients (safetensors) -> aggregated -> maybe a step
GET  /status      JSON snapshot: version, step, pending, mode, last_loss
GET  /checkpoint  download current weights as a safetensors file

Run locally:   uvicorn server.app:app --host 0.0.0.0 --port 7860
On HF Spaces:  the Dockerfile runs exactly that command on port 7860.
"""

from __future__ import annotations

import io
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse

from server.aggregator import (
    GradientAggregator,
    InvalidGradientError,
    StaleGradientError,
)
from server.state import GlobalState
from swarm import protocol as P
from swarm.config import ModelConfig, SwarmConfig


def create_app() -> FastAPI:
    swarm_cfg = SwarmConfig().validate()
    state = GlobalState.bootstrap(ModelConfig(), swarm_cfg)
    aggregator = GradientAggregator(state)

    app = FastAPI(title="Swarm Parameter Server", version="0.1.0")
    app.state.global_state = state
    app.state.aggregator = aggregator
    app.state.swarm_cfg = swarm_cfg

    def _check_auth(authorization: Optional[str]) -> None:
        token = swarm_cfg.token
        if not token:
            return
        if authorization != P.bearer(token):
            raise HTTPException(status_code=401, detail="invalid or missing token")

    @app.get(P.EP_HEALTH)
    def health():
        return {"status": "ok"}

    @app.get(P.EP_CONFIG)
    def config():
        return {
            "model": state.model_cfg.to_dict(),
            "mode": swarm_cfg.mode,
            "world_size": swarm_cfg.world_size,
            "auth_required": bool(swarm_cfg.token),
        }

    @app.get(P.EP_STATUS)
    def status():
        return aggregator.status()

    @app.get(P.EP_WEIGHTS)
    async def weights():
        data, version = await run_in_threadpool(_serialize_weights, state)
        return Response(
            content=data,
            media_type=P.CONTENT_TYPE,
            headers={P.H_MODEL_VERSION: str(version)},
        )

    @app.get(P.EP_CHECKPOINT)
    async def checkpoint():
        data, version = await run_in_threadpool(_serialize_weights, state)
        return StreamingResponse(
            io.BytesIO(data),
            media_type=P.CONTENT_TYPE,
            headers={
                P.H_MODEL_VERSION: str(version),
                "Content-Disposition": f'attachment; filename="global_v{version}.safetensors"',
            },
        )

    @app.post(P.EP_GRADIENTS)
    async def gradients(
        request: Request,
        x_model_version: int = Header(...),
        x_loss: Optional[float] = Header(default=None),
        x_worker_id: Optional[str] = Header(default=None),
        authorization: Optional[str] = Header(default=None),
    ):
        _check_auth(authorization)
        body = await request.body()
        try:
            result = await run_in_threadpool(
                _process_gradients, aggregator, body, x_model_version, x_loss
            )
        except StaleGradientError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except InvalidGradientError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return JSONResponse(
            {
                "accepted": result.accepted,
                "applied": result.applied,
                "version": result.version,
                "step": result.step,
                "pending": result.pending,
                "last_loss": result.last_loss,
                "worker_id": x_worker_id,
            }
        )

    return app


def _serialize_weights(state: GlobalState):
    # Read version and weights together; a concurrent step may bump version between
    # the two, which is harmless — the worker reports whichever version it received.
    version = state.version
    data = P.serialize_tensors(state.weights_state_dict())
    return data, version


def _process_gradients(
    aggregator: GradientAggregator, body: bytes, version: int, loss: Optional[float]
):
    grads = P.deserialize_tensors(body)
    return aggregator.submit(grads, worker_version=version, loss=loss)


app = create_app()


def main() -> None:
    """Console-script entrypoint: ``swarm-server``."""
    import os

    import uvicorn

    uvicorn.run(
        "server.app:app",
        host=os.environ.get("SWARM_HOST", "0.0.0.0"),
        port=int(os.environ.get("SWARM_PORT", os.environ.get("PORT", 7860))),
        log_level=os.environ.get("SWARM_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
