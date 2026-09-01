"""Pre-norm Transformer decoder block: attention sublayer + feed-forward sublayer.

Pre-norm vs post-norm. Post-norm (the original 2017 formulation) computes
``LayerNorm(x + Sublayer(x))``, which places a normalisation on the residual
path itself: the identity shortcut is rescaled at every layer, so gradients are
repeatedly transformed on the way back and deep stacks need careful LR warmup to
avoid diverging. Pre-norm computes ``x + Sublayer(LayerNorm(x))``, leaving the
residual path a clean additive identity from input to output. Gradients reach
early layers undistorted, which is why pre-norm is the standard choice for deep
stacks and why it is used here.

What breaks without each piece:
  * no residual connections -> gradients must pass through every sublayer
    multiplicatively; an 8-layer stack trains far worse or not at all.
  * no LayerNorm -> activation scale drifts layer to layer and training
    destabilises, especially under fp16.
  * no FFN -> attention only ever forms weighted averages of existing value
    vectors; the FFN is the per-position non-linear transform that lets the
    model compute new features rather than just mix old ones.
"""

import torch
import torch.nn as nn

from src.model.attention import MultiHeadCausalSelfAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Linear -> Dropout.

    Applied independently and identically at every position, which is what
    "position-wise" means: (B, T, d) -> (B, T, d_ff) -> (B, T, d).

    Args:
        d_model: Model dimension.
        d_ff: Inner (expanded) dimension; larger than d_model.
        dropout: Dropout applied to the sublayer output.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (B, T, d_model). Returns: (B, T, d_model)."""
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """One pre-norm decoder block.

        x = x + Attn(LN1(x))
        x = x + FFN(LN2(x))

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        ctx_len: Maximum sequence length (sizes the causal mask).
        dropout: Dropout for attention weights and both sublayer outputs.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, ctx_len: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(d_model, n_heads, ctx_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        """Args:
            x: (B, T, d_model).
            need_weights: propagate the attention-probability hook.

        Returns:
            (x, attn) with x of shape (B, T, d_model) and attn (B, h, T, T) or None.
        """
        attn_out, weights = self.attn(self.ln1(x), need_weights=need_weights)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, weights
