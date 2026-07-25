"""FastAPI parameter server — the swarm's "Parameter Server" on a Hugging Face Space.

Endpoints
---------
GET  /            live dashboard (loss curve, contributors, sampling)
GET  /health      liveness probe
GET  /config      model + swarm config (so a worker can self-verify it matches)
GET  /weights     current global weights (safetensors) + X-Model-Version
POST /gradients   upload gradients (safetensors, optionally Top-K compressed)
GET  /status      JSON snapshot: version, step, pending, mode, rule, last_loss
GET  /workers     per-worker contribution ledger + bandwidth summary
GET  /history     rolling (version, loss) history for the chart
GET  /generate    sample text from the current global model
GET  /checkpoint  download current weights as a safetensors file

Run locally:   uvicorn server.app:app --host 0.0.0.0 --port 7860
On HF Spaces:  the Dockerfile runs exactly that command on port 7860.
"""

from __future__ import annotations

import io
import math
from typing import Optional

import torch
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from server.aggregator import (
    GradientAggregator,
    InvalidGradientError,
    StaleGradientError,
)
from server.dashboard import DASHBOARD_HTML
from server.state import GlobalState
from server.telemetry import SwarmTelemetry
from swarm import compression as C
from swarm import protocol as P
from swarm.config import ModelConfig, SwarmConfig

# Reject absurd uploads before deserializing anything.
MAX_UPLOAD_BYTES = 256 * 1024 * 1024


def create_app() -> FastAPI:
    swarm_cfg = SwarmConfig().validate()
    state = GlobalState.bootstrap(ModelConfig(), swarm_cfg)
    aggregator = GradientAggregator(state)
    telemetry = SwarmTelemetry()

    app = FastAPI(title="Swarm Parameter Server", version="0.2.0")
    app.state.global_state = state
    app.state.aggregator = aggregator
    app.state.telemetry = telemetry
    app.state.swarm_cfg = swarm_cfg

    def _check_auth(authorization: Optional[str]) -> None:
        token = swarm_cfg.token
        if not token:
            return
        if authorization != P.bearer(token):
            raise HTTPException(status_code=401, detail="invalid or missing token")

    @app.get(P.EP_DASHBOARD, response_class=HTMLResponse, include_in_schema=False)
    def dashboard():
        return HTMLResponse(DASHBOARD_HTML)

    @app.get(P.EP_HEALTH)
    def health():
        return {"status": "ok"}

    @app.get(P.EP_CONFIG)
    def config():
        return {
            "model": state.model_cfg.to_dict(),
            "mode": swarm_cfg.mode,
            "rule": swarm_cfg.rule,
            "world_size": swarm_cfg.world_size,
            "auth_required": bool(swarm_cfg.token),
            "compression_supported": list(C.MODES),
        }

    @app.get(P.EP_STATUS)
    def status():
        return aggregator.status()

    @app.get(P.EP_WORKERS)
    def workers():
        return {"workers": telemetry.workers(), "summary": telemetry.summary()}

    @app.get(P.EP_HISTORY)
    def history():
        return {"history": telemetry.history()}

    @app.get(P.EP_GENERATE)
    async def generate(
        prompt: str = Query(default="", max_length=512),
        tokens: int = Query(default=200, ge=1, le=1000),
        temperature: float = Query(default=0.8, gt=0.0, le=5.0),
    ):
        if state.tokenizer is None:
            raise HTTPException(status_code=503, detail="tokenizer unavailable")
        text = await run_in_threadpool(_sample, state, prompt, tokens, temperature)
        return {"text": text, "version": state.version, "prompt": prompt}

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
        x_compression: str = Header(default=C.NONE),
        x_local_steps: int = Header(default=1),
        authorization: Optional[str] = Header(default=None),
    ):
        _check_auth(authorization)
        body = await request.body()
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="gradient payload too large")

        scheme = (x_compression or C.NONE).lower()
        if scheme not in C.MODES:
            raise HTTPException(status_code=400, detail=f"unknown compression {scheme!r}")

        # Clients control this header; NaN/inf must never enter server state.
        if x_loss is not None and not math.isfinite(x_loss):
            x_loss = None

        # A compressed upload is only a fraction of a dense one; record both so
        # the dashboard can show what the interconnect actually saved.
        wire_bytes = len(body)
        dense_bytes = state.dense_gradient_bytes

        try:
            result = await run_in_threadpool(
                _process_gradients, aggregator, state, body, x_model_version, x_loss, scheme
            )
        except StaleGradientError as e:
            telemetry.record_push(
                x_worker_id, outcome="stale", wire_bytes=wire_bytes, dense_bytes=dense_bytes
            )
            raise HTTPException(status_code=409, detail=str(e))
        except (InvalidGradientError, KeyError, ValueError) as e:
            telemetry.record_push(
                x_worker_id, outcome="rejected", wire_bytes=wire_bytes, dense_bytes=dense_bytes
            )
            raise HTTPException(status_code=400, detail=str(e))

        telemetry.record_push(
            x_worker_id,
            outcome="accepted",
            loss=x_loss,
            wire_bytes=wire_bytes,
            dense_bytes=dense_bytes,
            applied=result.applied,
            local_steps=max(int(x_local_steps), 1),
        )
        if result.applied:
            telemetry.record_step(result.version, result.last_loss)

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
    aggregator: GradientAggregator,
    state: GlobalState,
    body: bytes,
    version: int,
    loss: Optional[float],
    scheme: str,
):
    payload = P.deserialize_tensors(body)
    if scheme != C.NONE:
        payload = C.decompress(payload, state.param_shapes, mode=scheme)
    return aggregator.submit(payload, worker_version=version, loss=loss)


def _sample(state: GlobalState, prompt: str, tokens: int, temperature: float) -> str:
    tok = state.tokenizer
    model = state.model
    ids = tok.encode(prompt) if prompt else []
    if not ids:
        # Newline is a safe, always-present seed for a char-level model.
        ids = tok.encode("\n") or [0]
    idx = torch.tensor([ids], dtype=torch.long)
    was_training = model.training
    model.eval()
    try:
        out = model.generate(idx, max_new_tokens=tokens, temperature=temperature)
    finally:
        if was_training:
            model.train()
    return tok.decode(out[0].tolist())


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
