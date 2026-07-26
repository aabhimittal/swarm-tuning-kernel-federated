"""End-to-end tests through the real ASGI app (no network, but the full stack)."""

import os

import pytest
import torch
from fastapi.testclient import TestClient

from swarm import compression as C
from swarm import protocol as P


@pytest.fixture
def client(monkeypatch):
    # Small, fast model; one gradient per step so tests don't need a quorum.
    monkeypatch.setenv("SWARM_N_LAYER", "1")
    monkeypatch.setenv("SWARM_N_EMBD", "32")
    monkeypatch.setenv("SWARM_N_HEAD", "2")
    monkeypatch.setenv("SWARM_BLOCK_SIZE", "16")
    monkeypatch.setenv("SWARM_WORLD_SIZE", "1")
    monkeypatch.setenv("SWARM_CHECKPOINT", "")
    monkeypatch.setenv("SWARM_AGG_MODE", "sync")
    monkeypatch.setenv("SWARM_AGG_RULE", "mean")
    monkeypatch.setenv("SWARM_TOKEN", "")
    import importlib

    import server.app as app_module

    importlib.reload(app_module)
    with TestClient(app_module.create_app()) as c:
        yield c


def _shapes_from(client):
    """Parameter shapes, derived exactly the way a real worker derives them:
    rebuild the model from /config. (state_dict is not usable here — it also
    carries buffers and the tied wte/lm_head alias.)"""
    from swarm.config import ModelConfig
    from swarm.model import build_model

    cfg = ModelConfig.from_dict(client.get(P.EP_CONFIG).json()["model"])
    model = build_model(cfg)
    return {n: p.shape for n, p in model.named_parameters()}


def test_dashboard_and_core_endpoints(client):
    r = client.get("/")
    assert r.status_code == 200 and "Swarm Parameter Server" in r.text
    assert client.get(P.EP_HEALTH).json()["status"] == "ok"

    cfg = client.get(P.EP_CONFIG).json()
    assert cfg["rule"] == "mean"
    assert set(cfg["compression_supported"]) == set(C.MODES)

    st = client.get(P.EP_STATUS).json()
    assert st["version"] == 0 and st["rule"] == "mean"

    assert client.get(P.EP_WORKERS).json()["workers"] == []
    assert client.get(P.EP_HISTORY).json()["history"] == []


def test_generate_returns_text_from_the_global_model(client):
    r = client.get(P.EP_GENERATE, params={"tokens": 32, "prompt": "First"})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["text"], str) and len(body["text"]) > 0
    assert body["version"] == 0


def test_uncompressed_push_advances_the_model(client):
    shapes = _shapes_from(client)
    grads = {k: torch.full(s, 0.01) for k, s in shapes.items()}
    r = client.post(
        P.EP_GRADIENTS,
        content=P.serialize_tensors(grads),
        headers={
            P.H_MODEL_VERSION: "0",
            P.H_LOSS: "3.5",
            P.H_WORKER_ID: "w0",
            P.H_COMPRESSION: C.NONE,
        },
    )
    assert r.status_code == 200
    assert r.json()["applied"] is True
    assert client.get(P.EP_STATUS).json()["version"] == 1


def test_compressed_push_is_accepted_and_recorded(client):
    shapes = _shapes_from(client)
    grads = {k: torch.randn(s) for k, s in shapes.items()}
    wire = C.compress(grads, mode=C.TOPK, ratio=0.05)

    r = client.post(
        P.EP_GRADIENTS,
        content=P.serialize_tensors(wire),
        headers={
            P.H_MODEL_VERSION: "0",
            P.H_LOSS: "3.2",
            P.H_WORKER_ID: "sparse-worker",
            P.H_COMPRESSION: C.TOPK,
        },
    )
    assert r.status_code == 200 and r.json()["applied"] is True

    payload = client.get(P.EP_WORKERS).json()
    row = next(w for w in payload["workers"] if w["worker_id"] == "sparse-worker")
    assert row["accepted"] == 1
    # The whole point: fewer bytes on the wire than a dense upload.
    assert row["wire_bytes"] < row["dense_bytes"]
    assert payload["summary"]["compression_ratio"] > 1.0

    hist = client.get(P.EP_HISTORY).json()["history"]
    assert hist and hist[-1]["loss"] == pytest.approx(3.2)


def test_int8_compressed_push_is_accepted(client):
    shapes = _shapes_from(client)
    grads = {k: torch.randn(s) for k, s in shapes.items()}
    wire = C.compress(grads, mode=C.TOPK_INT8, ratio=0.05)
    r = client.post(
        P.EP_GRADIENTS,
        content=P.serialize_tensors(wire),
        headers={
            P.H_MODEL_VERSION: "0",
            P.H_LOSS: "3.1",
            P.H_COMPRESSION: C.TOPK_INT8,
        },
    )
    assert r.status_code == 200 and r.json()["applied"] is True


def test_unknown_compression_and_bad_payload_are_rejected(client):
    r = client.post(
        P.EP_GRADIENTS,
        content=b"x",
        headers={P.H_MODEL_VERSION: "0", P.H_COMPRESSION: "brotli-9000"},
    )
    assert r.status_code == 400

    # Well-formed safetensors, wrong keys for the model.
    bad = P.serialize_tensors({"not_a_param": torch.zeros(3)})
    r = client.post(
        P.EP_GRADIENTS,
        content=bad,
        headers={P.H_MODEL_VERSION: "0", P.H_COMPRESSION: C.NONE},
    )
    assert r.status_code == 400
    assert client.get(P.EP_WORKERS).json()["summary"]["known_workers"] >= 1


def test_stale_push_is_rejected_and_counted(client):
    shapes = _shapes_from(client)
    grads = {k: torch.zeros(s) for k, s in shapes.items()}
    hdr = {P.H_MODEL_VERSION: "0", P.H_LOSS: "3.0", P.H_WORKER_ID: "laggard"}
    assert client.post(P.EP_GRADIENTS, content=P.serialize_tensors(grads), headers=hdr).status_code == 200
    # Version is now 1; re-submitting against 0 is stale.
    r = client.post(P.EP_GRADIENTS, content=P.serialize_tensors(grads), headers=hdr)
    assert r.status_code == 409
    row = next(w for w in client.get(P.EP_WORKERS).json()["workers"] if w["worker_id"] == "laggard")
    assert row["stale"] == 1
