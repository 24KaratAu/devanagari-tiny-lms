"""Prove that an interrupted run resumes exactly where it left off.

The brief: "A run that cannot resume after interruption will lose marks."

A test that merely loads a file and checks the weights match is too weak -- it
passes even when the optimizer moments, LR schedule, or RNG stream are lost, and
each of those silently changes the trajectory after a restart. So the real
assertion here is end-to-end: training N more steps after a resume must produce
*identical* losses to training those same N steps without interruption.
"""

import copy

import pytest

# Guarded so the suite still collects on a machine without torch: the
# Phase 3 data tests are pure Python and should stay runnable there.
torch = pytest.importorskip("torch")

from src.model.transformer_lm import TransformerLM, TransformerLMConfig
from src.train.checkpoint import (BEST, LATEST, find_resume_point,
                                  load_checkpoint, prune_milestones,
                                  save_checkpoint)

CFG = TransformerLMConfig(vocab_size=128, d_model=32, n_layers=2, n_heads=4,
                d_ff=64, ctx_len=16, dropout=0.1)
RUN_CONFIG = {"run_name": "test", "architecture": {"d_model": 32}}


def _build(seed=0):
    """A small model with AdamW and a cosine schedule, deterministically seeded."""
    torch.manual_seed(seed)
    model = TransformerLM(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    return model, opt, sched


def _batches(n, seed=99):
    """Fixed sequence of (x, y) batches so runs are comparable."""
    g = torch.Generator().manual_seed(seed)
    return [(torch.randint(0, CFG.vocab_size, (2, CFG.ctx_len), generator=g),
             torch.randint(0, CFG.vocab_size, (2, CFG.ctx_len), generator=g))
            for _ in range(n)]


def _step(model, opt, sched, batch):
    """One optimisation step; returns the loss value."""
    x, y = batch
    _, loss, _ = model(x, targets=y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    return loss.item()


def test_resume_reproduces_uninterrupted_training(tmp_path):
    """Losses after a save/load cycle must match an uninterrupted run exactly."""
    batches = _batches(10)

    # --- Run A: 5 steps, checkpoint, then 5 more in the same process.
    model, opt, sched = _build()
    for b in batches[:5]:
        _step(model, opt, sched, b)
    torch.manual_seed(1234)  # a dropout stream the checkpoint must capture
    ckpt = save_checkpoint(tmp_path / LATEST, model, opt, sched, None,
                           step=5, best_val_loss=1.23, config=RUN_CONFIG,
                           tokens_seen=5 * 32)
    uninterrupted = [_step(model, opt, sched, b) for b in batches[5:]]

    # --- Run B: fresh objects, load the checkpoint, run the same 5 steps.
    model2, opt2, sched2 = _build(seed=777)   # deliberately different init
    meta = load_checkpoint(ckpt, model2, opt2, sched2, None)
    assert meta["step"] == 5
    assert meta["best_val_loss"] == 1.23
    assert meta["tokens_seen"] == 5 * 32
    assert meta["config"] == RUN_CONFIG
    resumed = [_step(model2, opt2, sched2, b) for b in batches[5:]]

    assert resumed == uninterrupted, (
        f"resume diverged from uninterrupted training:\n"
        f"  uninterrupted {uninterrupted}\n  resumed       {resumed}")


def test_all_required_fields_are_present(tmp_path):
    """The five fields the brief names must be in the payload."""
    model, opt, sched = _build()
    p = save_checkpoint(tmp_path / LATEST, model, opt, sched, None,
                        step=7, best_val_loss=2.0, config=RUN_CONFIG)
    blob = torch.load(p, map_location="cpu", weights_only=False)
    for field in ("model_state_dict", "optimizer_state_dict",
                  "scheduler_state_dict", "step", "config"):
        assert field in blob and blob[field] is not None, field
    assert blob["step"] == 7
    # Guideline 6: hyperparameters written alongside the weights.
    assert (tmp_path / "config.json").exists()


def test_optimizer_moments_survive(tmp_path):
    """AdamW's exp_avg / exp_avg_sq must round-trip, not silently reset."""
    model, opt, sched = _build()
    for b in _batches(3):
        _step(model, opt, sched, b)
    before = copy.deepcopy(opt.state_dict()["state"])

    p = save_checkpoint(tmp_path / LATEST, model, opt, sched, None, 3, 9.9, RUN_CONFIG)
    model2, opt2, sched2 = _build(seed=5)
    load_checkpoint(p, model2, opt2, sched2, None)
    after = opt2.state_dict()["state"]

    assert set(before) == set(after)
    for k in before:
        for buf in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(before[k][buf], after[k][buf])


def test_find_resume_point_prefers_latest_then_falls_back(tmp_path):
    """A truncated `latest` must not strand the run; fall back to a milestone."""
    model, opt, sched = _build()
    assert find_resume_point(tmp_path) is None          # fresh start

    save_checkpoint(tmp_path / "checkpoint_step_100.pt", model, opt, sched,
                    None, 100, 1.0, RUN_CONFIG)
    save_checkpoint(tmp_path / LATEST, model, opt, sched, None, 150, 1.0, RUN_CONFIG)
    assert find_resume_point(tmp_path).name == LATEST

    # Simulate a session killed mid-write.
    (tmp_path / LATEST).write_bytes(b"\x00\x01\x02truncated")
    assert find_resume_point(tmp_path).name == "checkpoint_step_100.pt"


def test_save_is_atomic_no_tmp_left_behind(tmp_path):
    """The temporary file must be swapped away, never left next to the real one."""
    model, opt, sched = _build()
    save_checkpoint(tmp_path / LATEST, model, opt, sched, None, 1, 1.0, RUN_CONFIG)
    assert not list(tmp_path.glob("*.tmp"))


def test_prune_keeps_latest_and_best(tmp_path):
    """Milestone pruning must never delete `latest` or `best`."""
    model, opt, sched = _build()
    for s in (1000, 2000, 3000, 4000):
        save_checkpoint(tmp_path / f"checkpoint_step_{s}.pt", model, opt, sched,
                        None, s, 1.0, RUN_CONFIG)
    save_checkpoint(tmp_path / LATEST, model, opt, sched, None, 4000, 1.0, RUN_CONFIG)
    save_checkpoint(tmp_path / BEST, model, opt, sched, None, 3000, 0.5, RUN_CONFIG)

    prune_milestones(tmp_path, keep=2)
    names = {p.name for p in tmp_path.glob("*.pt")}
    assert names == {LATEST, BEST, "checkpoint_step_3000.pt",
                     "checkpoint_step_4000.pt"}, names


def test_rng_restore_survives_a_gpu_style_payload(tmp_path):
    """A checkpoint loaded with map_location="cuda" must not crash the resume.

    torch.load moves every tensor in the payload to the target device, RNG
    states included, and set_rng_state rejects anything that is not a CPU
    ByteTensor. The earlier tests all saved and loaded on CPU, so this path went
    untested and raised TypeError in a real GPU run. RNG restoration is
    best-effort: correct weights matter, an exactly replayed dropout stream does
    not.
    """
    model, opt, sched = _build()
    p = save_checkpoint(tmp_path / LATEST, model, opt, sched, None,
                        step=3, best_val_loss=1.0, config=RUN_CONFIG)

    # Mimic what map_location="cuda" does to the saved states.
    blob = torch.load(p, map_location="cpu", weights_only=False)
    blob["rng_states"]["torch"] = blob["rng_states"]["torch"].float()
    blob["rng_states"]["cuda"] = [torch.zeros(16).float()]
    torch.save(blob, p)

    model2, opt2, sched2 = _build(seed=99)
    meta = load_checkpoint(p, model2, opt2, sched2, None)   # must not raise
    assert meta["step"] == 3
    for a, b in zip(model.parameters(), model2.parameters()):
        torch.testing.assert_close(a, b)


def test_rng_restore_tolerates_missing_state(tmp_path):
    """A checkpoint with no RNG block still resumes."""
    model, opt, sched = _build()
    p = save_checkpoint(tmp_path / LATEST, model, opt, sched, None, 5, 1.0, RUN_CONFIG)
    blob = torch.load(p, map_location="cpu", weights_only=False)
    blob["rng_states"] = None
    torch.save(blob, p)

    model2, opt2, sched2 = _build(seed=11)
    assert load_checkpoint(p, model2, opt2, sched2, None)["step"] == 5
