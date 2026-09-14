"""Contract tests for the packed-token dataset."""

import json

import numpy as np
import pytest

# Guarded so the suite still collects on a machine without torch: the
# Phase 3 data tests are pure Python and should stay runnable there.
torch = pytest.importorskip("torch")

from src.train.dataset import PackedTokenDataset, iter_eval_batches  # noqa: E402

CTX = 16
N = 5000


@pytest.fixture
def packed(tmp_path):
    """A synthetic .bin whose token at index i is (i % 8000), plus a sidecar."""
    ids = np.arange(N, dtype=np.uint16) % 8000
    p = tmp_path / "test_train.bin"
    ids.tofile(p)
    p.with_suffix(".json").write_text(json.dumps({
        "n_tokens": N, "n_tokens_no_eos": N - 10, "n_bytes_utf8": N * 9,
        "vocab_size": 8000, "eos_id": 3}))
    return p


def test_targets_are_inputs_shifted_by_one(packed):
    """y[:, t] must equal x[:, t+1] -- this is the whole training objective."""
    ds = PackedTokenDataset(packed, CTX, seed=0)
    x, y = ds.get_batch(4)
    assert x.shape == y.shape == (4, CTX)
    assert x.dtype == torch.long
    torch.testing.assert_close(x[:, 1:], y[:, :-1])


def test_windows_match_the_underlying_stream(packed):
    """Each row must be a contiguous slice of the file, not a shuffled gather."""
    ds = PackedTokenDataset(packed, CTX, seed=0)
    x, _ = ds.get_batch(8)
    for row in x:
        start = int(row[0])
        expected = torch.tensor([(start + i) % 8000 for i in range(CTX)])
        torch.testing.assert_close(row, expected)


def test_same_seed_reproduces_the_same_batches(packed):
    """Determinism is what makes a resumed run replay the same sample stream."""
    a = PackedTokenDataset(packed, CTX, seed=123).get_batch(4)
    b = PackedTokenDataset(packed, CTX, seed=123).get_batch(4)
    torch.testing.assert_close(a[0], b[0])
    c = PackedTokenDataset(packed, CTX, seed=456).get_batch(4)
    assert not torch.equal(a[0], c[0])


def test_generator_state_round_trips(packed):
    """Saving and restoring generator state continues the same sequence."""
    ds = PackedTokenDataset(packed, CTX, seed=7)
    ds.get_batch(4)
    state = ds.state_dict()
    expected = ds.get_batch(4)[0]

    ds2 = PackedTokenDataset(packed, CTX, seed=7)
    ds2.get_batch(4)
    ds2.load_state_dict(state)
    torch.testing.assert_close(ds2.get_batch(4)[0], expected)


def test_stale_sidecar_is_rejected(packed):
    """A sidecar that disagrees with the .bin must fail loudly, not silently."""
    packed.with_suffix(".json").write_text(json.dumps({"n_tokens": 999}))
    with pytest.raises(ValueError, match="stale"):
        PackedTokenDataset(packed, CTX, seed=0)


def test_missing_bin_is_reported_clearly(tmp_path):
    """A wrong path should say so rather than fail deep inside numpy."""
    with pytest.raises(FileNotFoundError):
        PackedTokenDataset(tmp_path / "nope.bin", CTX)


def test_eval_windows_are_deterministic_and_non_overlapping(packed):
    """Validation must cover the split once, identically on every call."""
    a = [x for x, _ in iter_eval_batches(packed, CTX, 4)]
    b = [x for x, _ in iter_eval_batches(packed, CTX, 4)]
    assert len(a) == len(b)
    for u, v in zip(a, b):
        torch.testing.assert_close(u, v)

    starts = [int(r[0]) for batch in a for r in batch]
    assert starts == sorted(starts)
    assert all(t - s == CTX for s, t in zip(starts, starts[1:])), "windows overlap"


def test_eval_respects_the_token_cap(packed):
    """max_tokens bounds the periodic in-training validation pass."""
    n = sum(x.numel() for x, _ in iter_eval_batches(packed, CTX, 4, max_tokens=800))
    assert 0 < n <= 800 + CTX * 4
