"""Verify empirically that the model cannot see the future.

Required by the project brief (§2.1): "verify empirically that the model cannot
see the future -- for example, show that changing token t+1 does not change the
logits at position t."

This is the test that catches a genuinely silent bug. A model with a broken
causal mask trains normally and reports an excellent validation loss, because it
is partly copying the answer from the future rather than predicting it. Nothing
downstream would reveal it.
"""

import torch

from src.model.transformer_lm import TransformerLM, TransformerLMConfig

CFG = TransformerLMConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                d_ff=128, ctx_len=32, dropout=0.0)


def _model():
    """A small deterministic model in eval mode (dropout off)."""
    torch.manual_seed(0)
    return TransformerLM(CFG).eval()


def test_future_token_does_not_change_past_logits():
    """Changing the token at t+1 must leave logits at every position <= t intact."""
    model = _model()
    torch.manual_seed(1)
    x = torch.randint(0, CFG.vocab_size, (2, CFG.ctx_len))

    with torch.no_grad():
        base, _, _ = model(x)

    t = CFG.ctx_len // 2
    x2 = x.clone()
    # Perturb position t+1 to a definitely different token.
    x2[:, t + 1] = (x[:, t + 1] + 1) % CFG.vocab_size
    assert not torch.equal(x, x2), "perturbation was a no-op"

    with torch.no_grad():
        pert, _, _ = model(x2)

    # Positions 0..t must be bit-for-bit unaffected.
    torch.testing.assert_close(base[:, :t + 1, :], pert[:, :t + 1, :],
                               rtol=0, atol=0)
    # Sanity: position t+1 itself MUST change, otherwise the test proves nothing
    # (a model ignoring its input would pass the assertion above trivially).
    assert not torch.allclose(base[:, t + 1, :], pert[:, t + 1, :]), \
        "logits at t+1 did not react to its own input token"


def test_every_position_is_independent_of_all_later_tokens():
    """Stronger form: perturb each position in turn and check the whole prefix."""
    model = _model()
    torch.manual_seed(2)
    x = torch.randint(0, CFG.vocab_size, (1, CFG.ctx_len))
    with torch.no_grad():
        base, _, _ = model(x)

    for t in range(CFG.ctx_len - 1):
        x2 = x.clone()
        x2[:, t + 1] = (x[:, t + 1] + 7) % CFG.vocab_size
        with torch.no_grad():
            pert, _, _ = model(x2)
        torch.testing.assert_close(base[:, :t + 1, :], pert[:, :t + 1, :],
                                   rtol=0, atol=0,
                                   msg=f"position {t} leaked information from {t + 1}")


def test_attention_weights_are_strictly_lower_triangular():
    """No probability mass may sit above the diagonal, in any head or layer."""
    model = _model()
    torch.manual_seed(3)
    x = torch.randint(0, CFG.vocab_size, (2, CFG.ctx_len))
    with torch.no_grad():
        _, _, attns = model(x, need_weights=True)

    assert len(attns) == CFG.n_layers
    upper = torch.triu(torch.ones(CFG.ctx_len, CFG.ctx_len, dtype=torch.bool),
                       diagonal=1)
    for layer, a in enumerate(attns):
        assert a.shape == (2, CFG.n_heads, CFG.ctx_len, CFG.ctx_len), a.shape
        assert a[..., upper].abs().max().item() == 0.0, \
            f"layer {layer} attends to future positions"
        # Each row is a probability distribution over the allowed prefix.
        torch.testing.assert_close(a.sum(-1), torch.ones_like(a.sum(-1)),
                                   rtol=1e-5, atol=1e-5)
