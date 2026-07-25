import pytest
import torch

from server import robust


def honest(value, n=1):
    """n honest workers all reporting roughly the same small gradient."""
    return [{"w": torch.full((8,), value)} for _ in range(n)]


def poisoned(value=1e6):
    return {"w": torch.full((8,), value)}


def test_mean_is_destroyed_by_a_single_poisoned_worker():
    """Motivates the other rules: plain averaging has zero breakdown point."""
    buf = honest(1.0, 4) + [poisoned()]
    out = robust.aggregate(buf, rule="mean")
    assert out["w"].abs().max() > 1000  # one attacker dominated the update


@pytest.mark.parametrize("rule", ["median", "trimmed_mean"])
def test_robust_rules_stop_sign_flip_attacks_that_clipping_cannot(rule):
    """The attack gradient-norm clipping does *not* stop.

    Clipping bounds an update's magnitude, so a huge poisoned tensor gets scaled
    back down. It does nothing about *direction*: a minority submitting modest,
    correctly-scaled but reversed gradients still drags the mean toward zero (or
    past it). Coordinate-wise rules reject them on the majority vote instead.

    Note the bound: trimmed_mean only removes ``int(n * trim_ratio)`` values per
    end, so it tolerates that many attackers and no more. Size trim_ratio to the
    fraction of the swarm you are willing to distrust.
    """
    honest_g = 1.0
    # 6 honest + 2 attackers. trim_ratio 0.25 over 8 workers trims 2 per end,
    # which is exactly the number of attackers — the rule's tolerance bound.
    buf = honest(honest_g, 6) + [{"w": torch.full((8,), -5.0)} for _ in range(2)]

    biased = robust.aggregate(buf, rule="mean")["w"]
    defended = robust.aggregate(buf, rule=rule, trim_ratio=0.25)["w"]

    # The mean is dragged below zero — the update now points the wrong way.
    assert biased.mean().item() < 0
    # The robust rule keeps the honest consensus and its sign.
    assert defended.mean().item() > 0
    assert torch.allclose(defended, torch.full((8,), honest_g), atol=1e-6)


@pytest.mark.parametrize("rule", ["median", "trimmed_mean", "krum"])
def test_robust_rules_survive_a_poisoned_worker(rule):
    buf = honest(1.0, 6) + [poisoned()]
    out = robust.aggregate(buf, rule=rule, trim_ratio=0.25)
    # The attacker is ignored: result stays near the honest consensus.
    assert torch.allclose(out["w"], torch.ones(8), atol=1e-3)


@pytest.mark.parametrize("rule", ["median", "trimmed_mean", "krum"])
def test_robust_rules_survive_multiple_colluding_attackers(rule):
    buf = honest(2.0, 7) + [poisoned(1e6), poisoned(-1e6)]
    out = robust.aggregate(buf, rule=rule, trim_ratio=0.25)
    assert torch.allclose(out["w"], torch.full((8,), 2.0), atol=1e-2)


def test_median_matches_torch_on_clean_input():
    buf = [{"w": torch.tensor([1.0, 9.0])}, {"w": torch.tensor([3.0, 1.0])},
           {"w": torch.tensor([2.0, 5.0])}]
    out = robust.aggregate(buf, rule="median")
    assert torch.allclose(out["w"], torch.tensor([2.0, 5.0]))


def test_trimmed_mean_drops_both_extremes():
    vals = [1.0, 2.0, 3.0, 4.0, 100.0]
    buf = [{"w": torch.tensor([v])} for v in vals]
    # trim=1 per end -> mean of [2,3,4]
    out = robust.aggregate(buf, rule="trimmed_mean", trim_ratio=0.2)
    assert torch.allclose(out["w"], torch.tensor([3.0]))


def test_trimmed_mean_falls_back_to_mean_when_swarm_too_small():
    buf = honest(1.0, 2)
    out = robust.aggregate(buf, rule="trimmed_mean", trim_ratio=0.25)
    assert torch.allclose(out["w"], torch.ones(8))


def test_krum_returns_one_of_the_submitted_gradients():
    buf = honest(1.0, 5) + [poisoned()]
    out = robust.aggregate(buf, rule="krum", trim_ratio=0.25)
    assert any(torch.equal(out["w"], g["w"]) for g in buf)


def test_single_gradient_passes_through_for_every_rule():
    for rule in ("mean", "median", "trimmed_mean", "krum"):
        out = robust.aggregate(honest(1.5, 1), rule=rule)
        assert torch.allclose(out["w"], torch.full((8,), 1.5))


def test_unknown_rule_and_empty_buffer_raise():
    with pytest.raises(ValueError):
        robust.aggregate(honest(1.0, 2), rule="nope")
    with pytest.raises(ValueError):
        robust.aggregate([], rule="mean")
