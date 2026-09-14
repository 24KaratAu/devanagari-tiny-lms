"""Parameter count and tensor-shape contract for Model H / Model L."""

import pytest

# Guarded so the suite still collects on a machine without torch: the
# Phase 3 data tests are pure Python and should stay runnable there.
torch = pytest.importorskip("torch")

from src.model.transformer_lm import (TransformerLM,  # noqa: E402
                                        TransformerLMConfig)

EXPECTED_PARAMS = 25_379_840


def test_parameter_count_matches_target():
    """The shipped config must produce exactly 25,379,840 trainable parameters."""
    cfg = TransformerLMConfig.from_yaml("hindi/configs/model.yaml")
    model = TransformerLM(cfg)
    assert cfg.expected_params() == EXPECTED_PARAMS
    assert model.num_params() == EXPECTED_PARAMS, model.param_breakdown()


def test_hindi_and_nepali_are_identical_in_shape():
    """Both Phase 1 vocabularies are 8,000, so H and L have equal capacity."""
    h = TransformerLMConfig.from_yaml("hindi/configs/model.yaml")
    l = TransformerLMConfig.from_yaml("nepali/configs/model.yaml")
    assert h.expected_params() == l.expected_params() == EXPECTED_PARAMS


def test_weight_tying_shares_storage():
    """The LM head and the token embedding must be one tensor, not a copy."""
    model = TransformerLM(TransformerLMConfig())
    assert model.lm_head.weight.data_ptr() == model.embed.token_emb.weight.data_ptr()


def test_forward_shapes_and_loss():
    """logits (B,T,V), scalar loss, attentions (B,h,T,T) per layer."""
    cfg = TransformerLMConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                    d_ff=128, ctx_len=32, dropout=0.0)
    model = TransformerLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (3, cfg.ctx_len))
    y = torch.randint(0, cfg.vocab_size, (3, cfg.ctx_len))

    logits, loss, attns = model(x, targets=y, need_weights=True)
    assert logits.shape == (3, cfg.ctx_len, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)
    # An untrained model should sit near ln(vocab_size).
    assert abs(loss.item() - torch.log(torch.tensor(float(cfg.vocab_size)))) < 1.0
    assert len(attns) == cfg.n_layers
    assert attns[0].shape == (3, cfg.n_heads, cfg.ctx_len, cfg.ctx_len)


def test_sequence_longer_than_ctx_len_is_rejected():
    """Learned absolute positions have no row beyond ctx_len; fail loudly."""
    cfg = TransformerLMConfig(vocab_size=256, d_model=64, n_layers=1, n_heads=4,
                    d_ff=128, ctx_len=16, dropout=0.0)
    model = TransformerLM(cfg).eval()
    try:
        model(torch.randint(0, 256, (1, 17)))
    except ValueError as e:
        assert "ctx_len" in str(e)
    else:
        raise AssertionError("expected ValueError for T > ctx_len")


def test_ablation_config_drops_position_table():
    """pos_encoding: none removes exactly ctx_len x d_model parameters."""
    base = TransformerLMConfig()
    abl = TransformerLMConfig(pos_encoding="none")
    assert base.expected_params() - abl.expected_params() == 512 * 512
    assert TransformerLM(abl).embed.pos_emb is None
