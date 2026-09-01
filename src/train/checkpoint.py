"""Checkpoint save/resume.

The brief makes this mandatory: "Your pipeline must save and resume from
intermediate checkpoints that include model weights, optimizer state, scheduler
state, training step, and configuration. A run that cannot resume after
interruption will lose marks."

Every checkpoint therefore carries all five required fields, plus four more that
matter for a run that is actually going to be interrupted:

  * ``scaler``   - fp16 AMP GradScaler state. Its loss scale is adapted during
                   training; dropping it makes the first steps after a resume
                   numerically different from an uninterrupted run.
  * ``rng``      - Python / NumPy / Torch RNG states, so the resumed run draws
                   the same batches and dropout masks it would have drawn.
  * ``best_val_loss`` / ``tokens_seen`` - so "best so far" and the token budget
                   survive a restart rather than silently resetting.

Writes are atomic. A Colab session that dies midway through a 300 MB write to a
Drive mount would otherwise leave a truncated file exactly where the resume path
expects a valid one -- turning a recoverable interruption into a lost run. We
write to a temporary file in the same directory and ``os.replace`` it, which is
atomic on POSIX, so the destination is always either the old checkpoint or the
complete new one.

Layout on disk (one directory per language; H and L never share):

    checkpoints/
      checkpoint_latest.pt      overwritten every save_latest_every steps
      checkpoint_best.pt        overwritten when validation loss improves
      checkpoint_step_2000.pt   milestones, for the loss-curve narrative
      config.json               hyperparameters, per guideline 6
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

LATEST = "checkpoint_latest.pt"
BEST = "checkpoint_best.pt"


def _rng_state() -> dict:
    """Snapshot every RNG the training loop consumes."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict):
    """Restore RNGs saved by :func:`_rng_state`, tolerating a device change."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu")
                        else state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except (RuntimeError, ValueError):
            # Resuming on a different GPU count: model/optimizer state is still
            # valid, only the CUDA RNG stream restarts.
            pass


def save_checkpoint(path, model, optimizer, scheduler, scaler, step,
                    best_val_loss, config, tokens_seen=0, extra=None):
    """Atomically write a resume-capable checkpoint.

    Args:
        path: Destination ``.pt`` file.
        model: The :class:`~src.model.transformer_lm.TransformerLM` being trained.
        optimizer: AdamW instance (its moment buffers dominate the file size).
        scheduler: LR scheduler, or None.
        scaler: ``torch.amp.GradScaler``, or None when not using fp16.
        step: Global training step just completed.
        best_val_loss: Best validation loss seen so far.
        config: Full run configuration dict, embedded for reproducibility.
        tokens_seen: Tokens consumed so far.
        extra: Optional dict merged into the payload.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        # --- the five fields the brief requires ---
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "config": config,
        # --- needed for a bit-faithful resume ---
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "rng_states": _rng_state(),
        "best_val_loss": best_val_loss,
        "tokens_seen": tokens_seen,
    }
    if extra:
        payload.update(extra)

    # Atomic: complete the write, fsync, then swap into place.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)

    # Guideline 6: hyperparameters live next to the weights.
    cfg_path = path.parent / "config.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(config, indent=2, default=str),
                            encoding="utf-8")
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    map_location="cpu", restore_rng=True) -> dict:
    """Restore a checkpoint in place and return its bookkeeping fields.

    Args:
        path: Checkpoint file.
        model: Model to load weights into.
        optimizer / scheduler / scaler: Restored when provided. Pass None for
            evaluation-only loads.
        map_location: Device mapping; "cpu" then ``.to(device)`` is safest when
            resuming on a different GPU than the run started on.
        restore_rng: Replay the saved RNG streams.

    Returns:
        {"step", "best_val_loss", "tokens_seen", "config"}.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if restore_rng:
        _restore_rng(ckpt.get("rng_states"))

    return {
        "step": ckpt.get("step", 0),
        "best_val_loss": ckpt.get("best_val_loss", float("inf")),
        "tokens_seen": ckpt.get("tokens_seen", 0),
        "config": ckpt.get("config", {}),
    }


def find_resume_point(ckpt_dir) -> Path | None:
    """Return the checkpoint to resume from, or None for a fresh start.

    Prefers ``checkpoint_latest.pt``; falls back to the highest-numbered
    milestone if that file is missing or was corrupted by a hard kill.
    """
    d = Path(ckpt_dir)
    if not d.exists():
        return None

    latest = d / LATEST
    if latest.exists() and _is_readable(latest):
        return latest

    milestones = sorted(d.glob("checkpoint_step_*.pt"),
                        key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    for p in reversed(milestones):
        if _is_readable(p):
            return p
    return None


def _is_readable(path: Path) -> bool:
    """Cheap integrity probe: can torch parse the file's header at all?"""
    try:
        torch.load(path, map_location="meta", weights_only=False, mmap=True)
        return True
    except Exception:
        try:
            torch.load(path, map_location="cpu", weights_only=False)
            return True
        except Exception:
            return False


def prune_milestones(ckpt_dir, keep: int = 6):
    """Delete the oldest milestone checkpoints, keeping the most recent ``keep``.

    Kaggle caps notebook output at 20 GB and each checkpoint here is ~300 MB, so
    milestones are pruned rather than accumulated. ``latest`` and ``best`` are
    never touched.
    """
    ms = sorted(Path(ckpt_dir).glob("checkpoint_step_*.pt"),
                key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    for p in ms[:-keep] if keep > 0 else ms:
        p.unlink(missing_ok=True)


def checkpoint_size_mb(n_params: int, optimizer_states: int = 2) -> float:
    """Estimate checkpoint size: fp32 weights plus AdamW's moment buffers.

    For this project: 25,379,840 params x 4 bytes x (1 + 2) ~= 305 MB.
    """
    return n_params * 4 * (1 + optimizer_states) / 1e6
