"""Multi-head causal self-attention, implemented from first principles.

No ``nn.Transformer*`` and no pre-built attention: only ``nn.Linear``,
``nn.Dropout``, and explicit tensor algebra, as the project brief requires.

Shape walk-through for one forward pass (B batch, T time, d model dim,
h heads, d_k = d / h):

    x                  (B, T, d)
    -> qkv proj        (B, T, 3d)          one fused nn.Linear
    -> split           3 x (B, T, d)
    -> view            (B, T, h, d_k)      split the channel axis into heads
    -> transpose       (B, h, T, d_k)      heads become a batch dimension
    q @ k^T            (B, h, T, T)        every query scored against every key
    / sqrt(d_k)        (B, h, T, T)        scaling (see below)
    + causal mask      (B, h, T, T)        0 on allowed, -inf on future
    softmax            (B, h, T, T)        rows sum to 1
    @ v                (B, h, T, d_k)      weighted sum of values
    -> transpose       (B, T, h, d_k)
    -> reshape         (B, T, d)           concatenate the heads
    -> out proj        (B, T, d)           W_O mixes information across heads

Why divide by sqrt(d_k): q and k entries are roughly zero-mean unit-variance, so
their dot product over d_k dimensions has variance ~d_k and grows with head
size. Feeding those large logits to softmax saturates it — the output approaches
a one-hot argmax and its gradient vanishes. Dividing by sqrt(d_k) rescales the
logits back to unit variance, keeping softmax in a responsive regime.

Why the mask is additive and applied *before* softmax: setting future logits to
-inf makes their exponentials exactly zero, so they contribute no probability
mass and receive no gradient. Zeroing probabilities *after* softmax would leave
rows that no longer sum to 1 and would still leak future information into the
normaliser.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCausalSelfAttention(nn.Module):
    """Causal (masked) multi-head self-attention.

    Args:
        d_model: Model dimension; must be divisible by n_heads.
        n_heads: Number of attention heads.
        ctx_len: Maximum sequence length, used to size the cached causal mask.
        dropout: Probability for both attention-weight and output dropout.
    """

    def __init__(self, d_model: int, n_heads: int, ctx_len: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        # One fused projection for Q, K, V is mathematically identical to three
        # separate nn.Linear(d, d) layers but is a single matmul.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)             # W_O

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Lower-triangular boolean mask, (1, 1, ctx_len, ctx_len). Registered as
        # a non-persistent buffer: it moves with .to(device) but is not a
        # parameter and is not written into checkpoints.
        causal = torch.tril(torch.ones(ctx_len, ctx_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal.view(1, 1, ctx_len, ctx_len),
                             persistent=False)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        """Apply causal self-attention.

        Args:
            x: Input of shape (B, T, d_model).
            need_weights: If True, also return the post-softmax attention
                probabilities. Required by the Phase 2 attention analysis
                (heatmaps, entropy, mean attention distance).

        Returns:
            (out, attn) where out is (B, T, d_model) and attn is
            (B, n_heads, T, T) when need_weights is True, else None.
        """
        B, T, C = x.shape

        # (B, T, 3d) -> 3 x (B, T, d) -> (B, h, T, d_k)
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # (B, h, T, d_k) @ (B, h, d_k, T) -> (B, h, T, T)
        att = (q @ k.transpose(-2, -1)) * self.scale

        # Additive causal mask: position t may attend only to positions <= t.
        # masked_fill with -inf is the additive mask in its most direct form.
        att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))

        att = F.softmax(att, dim=-1)
        weights = att if need_weights else None
        att = self.attn_dropout(att)

        # (B, h, T, T) @ (B, h, T, d_k) -> (B, h, T, d_k)
        out = att @ v

        # Concatenate heads: (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, d).
        # .contiguous() is required because transpose leaves a non-contiguous
        # view that .view() cannot reinterpret.
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out, weights
