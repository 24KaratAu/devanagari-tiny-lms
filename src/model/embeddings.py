"""Input representation: token embeddings + learned absolute positions.

Self-attention is permutation invariant — it computes a weighted sum over
positions with no notion of order — so position has to be injected explicitly.
We use *learned absolute* positional embeddings: a second lookup table indexed
by position, added to the token embedding.

Consequence: the position table has a fixed number of rows (``ctx_len``), so the
model cannot process a sequence longer than ``ctx_len``. Sinusoidal or rotary
schemes would extrapolate; learned absolute deliberately does not. It is chosen
here because it is the simplest thing that works at a fixed 512-token context,
and because deleting one ``nn.Embedding`` gives a clean bonus ablation.
"""

import torch
import torch.nn as nn


class TokenPositionEmbedding(nn.Module):
    """Map token ids to vectors and add positional information.

    Args:
        vocab_size: Number of tokens in the (per-language) vocabulary.
        d_model: Model dimension.
        ctx_len: Maximum sequence length; number of rows in the position table.
        dropout: Embedding dropout probability.
        learned_pos: If False, positional embeddings are omitted entirely
            (the no-positional-embedding ablation).
    """

    def __init__(self, vocab_size: int, d_model: int, ctx_len: int,
                 dropout: float = 0.1, learned_pos: bool = True):
        super().__init__()
        self.ctx_len = ctx_len
        self.learned_pos = learned_pos
        self.token_emb = nn.Embedding(vocab_size, d_model)     # (V, d)
        self.pos_emb = nn.Embedding(ctx_len, d_model) if learned_pos else None
        self.drop = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Embed a batch of token id sequences.

        Args:
            idx: Token ids, shape (B, T), dtype long, values in [0, vocab_size).

        Returns:
            Embeddings of shape (B, T, d_model).
        """
        B, T = idx.shape
        if T > self.ctx_len:
            raise ValueError(
                f"sequence length {T} exceeds ctx_len {self.ctx_len}; learned "
                f"absolute positions have no row for position {T - 1}")

        x = self.token_emb(idx)                                # (B, T) -> (B, T, d)
        if self.pos_emb is not None:
            # (T,) -> (T, d), broadcast over the batch to (B, T, d).
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)
        return self.drop(x)
