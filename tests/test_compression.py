import torch

from swarm import compression as C


def _grads(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(4, 25, generator=g),
        "b": torch.randn(200, generator=g),
    }


def _shapes(d):
    return {k: v.shape for k, v in d.items()}


def test_topk_transmits_only_the_requested_fraction():
    grads = _grads()
    wire = C.compress(grads, mode=C.TOPK, ratio=0.1)
    for name, g in grads.items():
        k = C.topk_count(g.numel(), 0.1)
        assert wire[name + C.IDX].numel() == k
        assert wire[name + C.VAL].numel() == k


def test_topk_keeps_the_largest_magnitude_entries():
    grads = {"a": torch.tensor([0.1, -5.0, 0.2, 4.0])}
    dense = C.decompress(C.compress(grads, ratio=0.5), _shapes(grads))
    # The two largest-|v| entries survive exactly; the rest are zeroed.
    assert torch.allclose(dense["a"], torch.tensor([0.0, -5.0, 0.0, 4.0]))


def test_roundtrip_is_lossless_at_full_ratio():
    grads = _grads()
    dense = C.decompress(C.compress(grads, ratio=1.0), _shapes(grads))
    for k in grads:
        assert torch.allclose(dense[k], grads[k], atol=1e-6)


def test_int8_quantization_is_close_and_smaller():
    grads = _grads()
    plain = C.compress(grads, mode=C.TOPK, ratio=0.5)
    quant = C.compress(grads, mode=C.TOPK_INT8, ratio=0.5)
    assert C.payload_bytes(quant) < C.payload_bytes(plain)

    dense_q = C.decompress(quant, _shapes(grads), mode=C.TOPK_INT8)
    dense_p = C.decompress(plain, _shapes(grads), mode=C.TOPK)
    for k in grads:
        peak = dense_p[k].abs().max().item()
        # int8 with a per-tensor scale: error is bounded by half a quantization step.
        assert (dense_q[k] - dense_p[k]).abs().max().item() <= peak / 127.0


def test_compression_actually_shrinks_the_payload():
    grads = _grads()
    wire = C.compress(grads, ratio=0.01)
    d, w, ratio = C.compression_report(grads, wire)
    assert w < d and ratio > 1.0


def test_error_feedback_conserves_gradient_mass_exactly():
    """Dropped mass must reappear later, not vanish — this is what makes Top-K work.

    The exact invariant: with ``g_t = grad + r_{t-1}`` and ``r_t = g_t - sent_t``,
    the telescoping sum gives ``sum(sent) == n*grad - r_n``. Nothing is ever lost,
    only deferred into the residual.
    """
    torch.manual_seed(0)
    n = 120
    grad = {"a": torch.randn(500)}
    residual = {}
    shapes = _shapes(grad)

    accumulated = torch.zeros(500)
    for _ in range(n):
        wire = C.compress(grad, mode=C.TOPK, ratio=0.05, residual=residual)
        accumulated += C.decompress(wire, shapes)["a"]

    delivered_plus_pending = accumulated + residual["a"]
    assert torch.allclose(delivered_plus_pending, grad["a"] * n, atol=1e-3)
    # Coordinates keep getting their turn: at 5% per round nearly all 500 have
    # been delivered, versus the fixed 25 that plain Top-K would ever send.
    delivered = (accumulated.abs() > 0).sum().item()
    assert delivered > 0.9 * 500


def test_error_feedback_error_shrinks_as_rounds_accumulate():
    """Relative error decays like 1/n — the residual is bounded, the target grows."""
    torch.manual_seed(0)
    grad = {"a": torch.randn(400)}
    shapes = _shapes(grad)

    def rel_err_after(rounds):
        residual, acc = {}, torch.zeros(400)
        for _ in range(rounds):
            wire = C.compress(grad, mode=C.TOPK, ratio=0.05, residual=residual)
            acc += C.decompress(wire, shapes)["a"]
        target = grad["a"] * rounds
        return ((acc - target).norm() / target.norm()).item()

    assert rel_err_after(200) < rel_err_after(25) < 1.0


def test_without_error_feedback_most_coordinates_never_arrive():
    """Contrast case: plain Top-K keeps re-sending the same entries."""
    torch.manual_seed(0)
    grad = {"a": torch.randn(500)}
    shapes = _shapes(grad)
    acc = torch.zeros(500)
    for _ in range(120):
        acc += C.decompress(C.compress(grad, mode=C.TOPK, ratio=0.05), shapes)["a"]
    delivered = (acc.abs() > 0).sum().item()
    assert delivered <= C.topk_count(500, 0.05)  # stuck on the same 25 entries


def test_error_feedback_residual_holds_the_undelivered_remainder():
    grad = {"a": torch.tensor([3.0, 1.0, 2.0, 0.5])}
    residual = {}
    C.compress(grad, mode=C.TOPK, ratio=0.25, residual=residual)
    # Only the largest (3.0) was sent; the rest is carried forward.
    assert torch.allclose(residual["a"], torch.tensor([0.0, 1.0, 2.0, 0.5]))


def test_none_mode_is_passthrough():
    grads = _grads()
    assert C.compress(grads, mode=C.NONE) == grads
