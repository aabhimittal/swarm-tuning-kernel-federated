import torch

from swarm import protocol as P


def test_serialize_roundtrip_preserves_values_and_dtypes():
    tensors = {
        "a": torch.randn(3, 4),
        "b": torch.arange(10, dtype=torch.long),
        "c": torch.ones(1),
    }
    blob = P.serialize_tensors(tensors)
    assert isinstance(blob, (bytes, bytearray))
    out = P.deserialize_tensors(blob)

    assert set(out.keys()) == set(tensors.keys())
    for k in tensors:
        assert out[k].dtype == tensors[k].dtype
        assert torch.equal(out[k], tensors[k])


def test_serialize_handles_noncontiguous_tensor():
    # A transposed view is non-contiguous; protocol must make it contiguous.
    t = torch.randn(4, 5).t()
    out = P.deserialize_tensors(P.serialize_tensors({"x": t}))
    assert torch.equal(out["x"], t.contiguous())


def test_bearer_format():
    assert P.bearer("abc") == "Bearer abc"
