"""Memory-mapped access to the packed ``uint16`` token streams.

``pack_tokens.py`` wrote each split as a flat array of token ids with one
``<eos>`` after every document. Because that file is a contiguous array of
fixed-width integers, it can be memory-mapped: the OS pages in only the windows
actually touched, so a 1.06 GiB corpus costs almost no RAM and there is no
per-step tokenisation, JSON parsing, or Drive round-trip in the training loop.

Two sampling modes, for two different jobs:

  * :class:`PackedTokenDataset` -- random windows, for training. Sampling
    positions at random rather than walking the file in order decorrelates the
    examples inside a batch (consecutive windows would otherwise come from the
    same document) and makes resume trivial: restoring the generator state is
    enough, with no cursor to track.

  * :func:`iter_eval_batches` -- deterministic, non-overlapping windows covering
    the split exactly once. Validation loss has to be comparable across steps
    and across runs, so it must not depend on a random draw.

Documents are packed back-to-back, so a window can straddle an ``<eos>`` and
contain the tail of one document plus the head of the next. That is intended:
the model sees ``<eos>`` in context and learns that documents end, which is what
makes generation terminate rather than run on forever.
"""

import json
from pathlib import Path

import numpy as np
import torch


class PackedTokenDataset:
    """Random-window sampler over a packed ``.bin`` token stream.

    Args:
        bin_path: Path to ``{lang}_{split}.bin``.
        ctx_len: Sequence length; each example is ``ctx_len + 1`` tokens, split
            into inputs and targets shifted by one position.
        seed: Seed for the sampling generator.
        device: Device to place batches on.
    """

    def __init__(self, bin_path, ctx_len: int, seed: int = 42, device: str = "cpu"):
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"packed tokens not found: {self.bin_path}")

        # mode="r" keeps it read-only and shareable across worker processes.
        self.data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        self.ctx_len = ctx_len
        self.device = device
        self.generator = torch.Generator().manual_seed(seed)

        if len(self.data) <= ctx_len + 1:
            raise ValueError(
                f"{self.bin_path.name} holds {len(self.data)} tokens, too few "
                f"for ctx_len {ctx_len}")

        self.stats = self._load_sidecar()

    def _load_sidecar(self) -> dict:
        """Load the packing statistics written next to the ``.bin``."""
        p = self.bin_path.with_suffix(".json")
        if not p.exists():
            return {}
        stats = json.loads(p.read_text(encoding="utf-8"))
        if stats.get("n_tokens") != len(self.data):
            raise ValueError(
                f"{self.bin_path.name} has {len(self.data):,} tokens but its "
                f"sidecar claims {stats.get('n_tokens'):,} -- the pair is stale")
        return stats

    def __len__(self) -> int:
        """Number of tokens in the split (not the number of batches)."""
        return len(self.data)

    def get_batch(self, batch_size: int):
        """Draw a random batch of (inputs, targets).

        Args:
            batch_size: Number of sequences.

        Returns:
            (x, y), each of shape (batch_size, ctx_len), dtype long. ``y`` is
            ``x`` shifted one position left, so predicting ``y[:, t]`` from
            ``x[:, :t+1]`` is exactly next-token prediction.
        """
        hi = len(self.data) - self.ctx_len - 1
        ix = torch.randint(hi, (batch_size,), generator=self.generator)

        # .astype(np.int64) forces a copy out of the memmap; nn.Embedding needs
        # int64 and uint16 is not a dtype torch accepts directly.
        x = torch.stack([
            torch.from_numpy(self.data[i:i + self.ctx_len].astype(np.int64))
            for i in ix.tolist()])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1:i + 1 + self.ctx_len].astype(np.int64))
            for i in ix.tolist()])

        if self.device.startswith("cuda"):
            # pin + non_blocking overlaps the host-to-device copy with compute.
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def state_dict(self) -> dict:
        """Generator state, so a resumed run continues the same sample stream."""
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state: dict):
        """Restore a generator state produced by :meth:`state_dict`."""
        if state and "generator_state" in state:
            self.generator.set_state(state["generator_state"])


def iter_eval_batches(bin_path, ctx_len: int, batch_size: int,
                      max_tokens: int = None, device: str = "cpu"):
    """Yield deterministic, non-overlapping (x, y) batches over a split.

    Walks the token stream in strides of ``ctx_len``, so every token is
    predicted exactly once and no context is double-counted. This is what makes
    the reported validation loss and perplexity reproducible.

    Args:
        bin_path: Path to a packed ``.bin``.
        ctx_len: Sequence length.
        batch_size: Sequences per batch.
        max_tokens: Stop after roughly this many tokens. Used for the periodic
            in-training validation pass; pass None for the full split at the end.
        device: Device to place batches on.

    Yields:
        (x, y) tensors of shape (B, ctx_len), where the final batch may be
        smaller than ``batch_size``.
    """
    data = np.memmap(Path(bin_path), dtype=np.uint16, mode="r")
    limit = len(data) - 1
    if max_tokens is not None:
        limit = min(limit, max_tokens)

    starts = range(0, limit - ctx_len, ctx_len)
    batch = []
    for s in starts:
        batch.append(s)
        if len(batch) == batch_size:
            yield _stack(data, batch, ctx_len, device)
            batch = []
    if batch:
        yield _stack(data, batch, ctx_len, device)


def _stack(data, starts, ctx_len: int, device: str):
    """Build one (x, y) pair from a list of window start offsets."""
    x = torch.stack([torch.from_numpy(data[s:s + ctx_len].astype(np.int64))
                     for s in starts])
    y = torch.stack([torch.from_numpy(data[s + 1:s + 1 + ctx_len].astype(np.int64))
                     for s in starts])
    return x.to(device), y.to(device)
