import pytest
import torch

from server.aggregator import (
    GradientAggregator,
    InvalidGradientError,
    StaleGradientError,
)
from tests.conftest import const_grads, make_state, zero_grads


def test_average_is_elementwise_mean():
    from server import robust

    state = make_state()
    a = const_grads(state, 2.0)
    b = const_grads(state, 4.0)
    avg = robust.aggregate([a, b], rule="mean")
    for k in a:
        assert torch.allclose(avg[k], torch.full_like(avg[k], 3.0))


def test_sync_buffers_until_world_size_then_steps():
    state = make_state(mode="sync", world_size=3)
    agg = GradientAggregator(state)

    r1 = agg.submit(zero_grads(state), worker_version=0, loss=1.0)
    assert (r1.applied, r1.pending, r1.version) == (False, 1, 0)

    r2 = agg.submit(zero_grads(state), worker_version=0, loss=1.0)
    assert (r2.applied, r2.pending, r2.version) == (False, 2, 0)

    r3 = agg.submit(zero_grads(state), worker_version=0, loss=2.0)
    assert r3.applied is True
    assert r3.version == 1
    assert r3.step == 1
    assert r3.pending == 0
    # last_loss is the mean of the buffered losses (1,1,2) = 1.333...
    assert state.last_loss == pytest.approx((1.0 + 1.0 + 2.0) / 3.0)


def test_async_applies_every_gradient_immediately():
    state = make_state(mode="async", tolerance=10)
    agg = GradientAggregator(state)

    r1 = agg.submit(zero_grads(state), worker_version=0, loss=1.0)
    assert r1.applied is True and r1.version == 1
    # Now at version 1; a worker still at version 0 has age=1, within tolerance.
    r2 = agg.submit(zero_grads(state), worker_version=0, loss=1.0)
    assert r2.applied is True and r2.version == 2


def test_stale_gradient_rejected_with_strict_tolerance():
    state = make_state(mode="async", tolerance=0)
    agg = GradientAggregator(state)
    agg.submit(zero_grads(state), worker_version=0, loss=1.0)  # -> version 1
    with pytest.raises(StaleGradientError):
        agg.submit(zero_grads(state), worker_version=0, loss=1.0)  # age 1 > tol 0


def test_invalid_keys_rejected():
    state = make_state()
    agg = GradientAggregator(state)
    with pytest.raises(InvalidGradientError):
        agg.submit({"not_a_param": torch.zeros(2)}, worker_version=0, loss=1.0)


def test_actual_optimizer_step_reduces_loss_on_a_fixed_batch():
    """End-to-end-ish: feeding real gradients should lower the loss on that batch."""
    from server.state import GlobalState
    from swarm.config import SwarmConfig
    from swarm.data import build_dataset
    from tests.conftest import tiny_model_cfg

    ds = build_dataset(block_size=8)
    cfg = tiny_model_cfg(vocab_size=ds.vocab_size)
    swarm_cfg = SwarmConfig(
        mode="sync", world_size=1, checkpoint_path="", checkpoint_every=0
    ).validate()
    state = GlobalState(cfg, swarm_cfg)
    agg = GradientAggregator(state)
    gen = torch.Generator().manual_seed(0)
    x, y = ds.get_batch(8, generator=gen)

    def loss_now():
        state.model.eval()
        with torch.no_grad():
            _, l = state.model(x, y)
        return float(l)

    before = loss_now()
    for _ in range(15):
        state.model.train()
        state.model.zero_grad(set_to_none=True)
        _, l = state.model(x, y)
        l.backward()
        grads = {n: p.grad.detach().clone() for n, p in state.model.named_parameters()}
        agg.submit(grads, worker_version=state.version, loss=l.item())
    after = loss_now()
    assert after < before
