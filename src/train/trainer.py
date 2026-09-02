"""Pretraining loop for Model H / Model L.

AdamW with decoupled weight decay, linear warmup into cosine decay, gradient
clipping, fp16 mixed precision, periodic validation, and checkpointing that can
resume after an interrupted session.

Run:
    python -m src.train.trainer --config hindi/configs/train.yaml

Resume is automatic and requires no flag: on startup the trainer looks for a
usable checkpoint in the configured directory and continues from it. Re-running
the same command after a Colab or Kaggle disconnect is the whole recovery
procedure.
"""

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import torch
import yaml

from src.model.transformer_lm import TransformerLM, TransformerLMConfig
from src.train.checkpoint import (BEST, LATEST, find_resume_point,
                                  load_checkpoint, prune_milestones,
                                  save_checkpoint)
from src.train.dataset import PackedTokenDataset, iter_eval_batches


def build_optimizer(model, lr, betas, eps, weight_decay):
    """AdamW with weight decay applied only to matmul weights.

    Biases and LayerNorm gain/bias are 1-D and get no decay: shrinking them
    towards zero regularises nothing and measurably hurts. This is the standard
    GPT-2 grouping.

    Returns:
        A configured ``torch.optim.AdamW``.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    print(f"  optimizer: {len(decay)} decayed tensors "
          f"({sum(p.numel() for p in decay):,} params), "
          f"{len(no_decay)} undecayed ({sum(p.numel() for p in no_decay):,})")

    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(betas), eps=eps,
                             fused=fused)


def build_scheduler(optimizer, warmup_steps, max_steps, peak_lr, min_lr):
    """Linear warmup to ``peak_lr``, then cosine decay to ``min_lr``.

    Warmup exists because Adam's second-moment estimate is unreliable in the
    first few hundred steps; taking full-size steps then can push the model into
    a bad region it never recovers from. Cosine decay anneals the learning rate
    smoothly so late training refines rather than bounces.

    Returns:
        A ``LambdaLR`` whose multiplier is relative to ``peak_lr``.
    """
    floor = min_lr / peak_lr

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(1.0, progress)
        return floor + 0.5 * (1 - floor) * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, bin_path, ctx_len, batch_size, max_tokens, device, amp_dtype):
    """Mean cross-entropy over deterministic non-overlapping windows.

    Args:
        max_tokens: Cap for the periodic in-training pass; None for the full split.

    Returns:
        (loss, perplexity). Perplexity is ``exp(loss)``, clamped for display.
    """
    model.eval()
    total, n = 0.0, 0
    for x, y in iter_eval_batches(bin_path, ctx_len, batch_size, max_tokens, device):
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            _, loss, _ = model(x, targets=y)
        total += loss.item() * x.size(0)
        n += x.size(0)
    model.train()
    loss = total / max(1, n)
    return loss, math.exp(min(loss, 20))


def mirror(path: Path, remote: str):
    """Copy a checkpoint to a remote with rclone, best-effort.

    Kaggle's ``/kaggle/working`` is discarded when an interactive session dies,
    so on that platform the only crash-safe copy is the one pushed off-box.
    Failures are logged, never raised: losing a mirror must not kill a run that
    is otherwise fine.
    """
    if not remote or not shutil.which("rclone"):
        return
    try:
        subprocess.run(["rclone", "copy", "--no-traverse", str(path), remote],
                       check=True, capture_output=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  [warn] mirror to {remote} failed: {e}")


def train(cfg: dict):
    """Run pretraining to completion, resuming if a checkpoint exists."""
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = cfg["train"].get("precision", "fp16") == "fp16" and device == "cuda"
    amp_dtype = torch.float16 if use_fp16 else None
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})"
                                 if device == "cuda" else ""))

    # --- model -------------------------------------------------------------
    model_cfg = TransformerLMConfig.from_yaml(cfg["model_config"])
    model = TransformerLM(model_cfg).to(device)
    n_params = model.num_params()
    print(f"  model: {n_params:,} trainable parameters")
    if n_params != model_cfg.expected_params():
        raise RuntimeError(f"parameter count {n_params:,} != expected "
                           f"{model_cfg.expected_params():,}")

    # --- data --------------------------------------------------------------
    ctx = model_cfg.ctx_len
    bs = cfg["train"]["batch_size"]
    accum = cfg["train"].get("grad_accum", 1)
    train_ds = PackedTokenDataset(cfg["data"]["train_bin"], ctx, seed, device)
    tok_per_step = bs * accum * ctx
    print(f"  train: {len(train_ds):,} tokens, {tok_per_step:,} tokens/step")

    # --- optimisation ------------------------------------------------------
    o = cfg["optim"]
    optimizer = build_optimizer(model, o["lr"], o["betas"], o["eps"],
                                o["weight_decay"])
    max_steps = cfg["train"]["max_steps"]
    scheduler = build_scheduler(optimizer, cfg["schedule"]["warmup_steps"],
                                max_steps, o["lr"], cfg["schedule"]["min_lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # --- resume ------------------------------------------------------------
    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step, best_val, tokens_seen = 0, float("inf"), 0

    resume_from = find_resume_point(ckpt_dir)
    if resume_from:
        # map_location="cpu" (not device): load_state_dict copies into the
        # already-on-GPU model regardless, and keeping the payload on CPU avoids
        # moving RNG states to the GPU, where torch refuses to restore them.
        meta = load_checkpoint(resume_from, model, optimizer, scheduler, scaler,
                               map_location="cpu")
        step, best_val = meta["step"], meta["best_val_loss"]
        tokens_seen = meta["tokens_seen"]
        print(f"  RESUMED from {resume_from.name} at step {step:,} "
              f"({tokens_seen:,} tokens seen, best val {best_val:.4f})")
    else:
        print("  fresh start (no checkpoint found)")

    # --- logging -----------------------------------------------------------
    log_path = Path(cfg["log"]["jsonl"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_every = cfg["log"].get("every", 20)
    log_remote = cfg["log"].get("mirror_to")

    def log(record: dict):
        """Append one JSON record; this file is the loss curve."""
        record["wall_clock"] = round(time.time() - t_start, 1)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # --- loop --------------------------------------------------------------
    ck = cfg["checkpoint"]
    ev = cfg["eval"]
    remote = ck.get("mirror_to")
    t_start = t_win = time.time()
    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"\ntraining {step:,} -> {max_steps:,}\n" + "-" * 62)

    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(accum):
            x, y = train_ds.get_batch(bs)
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                                enabled=use_fp16):
                _, loss, _ = model(x, targets=y)
                loss = loss / accum
            scaler.scale(loss).backward()
            loss_acc += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   o["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        step += 1
        tokens_seen += tok_per_step

        if step % log_every == 0:
            dt = time.time() - t_win
            tps = tok_per_step * log_every / dt
            lr_now = scheduler.get_last_lr()[0]
            peak = (torch.cuda.max_memory_allocated() / 2**30
                    if device == "cuda" else 0.0)
            total = (torch.cuda.get_device_properties(0).total_memory / 2**30
                     if device == "cuda" else 0.0)
            print(f"step {step:>6,}/{max_steps:,}  loss {loss_acc:.4f}  "
                  f"lr {lr_now:.2e}  {tps:,.0f} tok/s  grad {grad_norm:.2f}  "
                  f"mem {peak:.1f}/{total:.1f}G  "
                  f"eta {(max_steps-step)/(log_every/dt)/3600:.2f}h")
            log({"step": step, "train_loss": round(loss_acc, 5),
                 "lr": lr_now, "tokens_seen": tokens_seen,
                 "tokens_per_sec": round(tps, 1),
                 "grad_norm": round(float(grad_norm), 4),
                 "peak_mem_gib": round(peak, 2)})
            t_win = time.time()

        if step % ev["every"] == 0 or step == max_steps:
            vloss, vppl = evaluate(model, cfg["data"]["val_bin"], ctx,
                                   ev["batch_size"], ev.get("val_tokens"),
                                   device, amp_dtype)
            print(f"  [eval] step {step:,}  val_loss {vloss:.4f}  ppl {vppl:.2f}"
                  + ("  <- best" if vloss < best_val else ""))
            log({"step": step, "val_loss": round(vloss, 5),
                 "val_ppl": round(vppl, 3), "tokens_seen": tokens_seen})
            if vloss < best_val and ck.get("keep_best", True):
                best_val = vloss
                p = save_checkpoint(ckpt_dir / BEST, model, optimizer, scheduler,
                                    scaler, step, best_val, cfg, tokens_seen)
                mirror(p, remote)

        if step % ck["save_latest_every"] == 0:
            p = save_checkpoint(ckpt_dir / LATEST, model, optimizer, scheduler,
                                scaler, step, best_val, cfg, tokens_seen)
            mirror(p, remote)
            mirror(log_path, log_remote)   # the loss curve must survive too

        if step % ck["save_milestone_every"] == 0:
            p = save_checkpoint(ckpt_dir / f"checkpoint_step_{step}.pt", model,
                                optimizer, scheduler, scaler, step, best_val,
                                cfg, tokens_seen)
            mirror(p, remote)
            prune_milestones(ckpt_dir, keep=ck.get("keep_milestones", 6))

    # --- final -------------------------------------------------------------
    p = save_checkpoint(ckpt_dir / LATEST, model, optimizer, scheduler, scaler,
                        step, best_val, cfg, tokens_seen)
    mirror(p, remote)
    mirror(log_path, log_remote)
    hrs = (time.time() - t_start) / 3600
    print("-" * 62)
    print(f"done: {step:,} steps, {tokens_seen:,} tokens, {hrs:.2f} h, "
          f"best val loss {best_val:.4f} (ppl {math.exp(min(best_val, 20)):.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="training YAML")
    ap.add_argument("--max-steps", type=int, help="override, for smoke tests")
    ap.add_argument("--mirror-to", help="rclone remote for crash-safe checkpoints")
    ap.add_argument("--no-mirror", action="store_true",
                    help="disable checkpoint mirroring (use for smoke runs, so "
                         "throwaway checkpoints never reach the real remote)")
    ap.add_argument("--ckpt-dir", help="override checkpoint directory (smoke tests)")
    ap.add_argument("--log-jsonl", help="override log path (smoke tests)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    if args.mirror_to:
        cfg["checkpoint"]["mirror_to"] = args.mirror_to
    if args.no_mirror:
        cfg["checkpoint"]["mirror_to"] = ""
        cfg["log"]["mirror_to"] = ""
    if args.ckpt_dir:
        cfg["checkpoint"]["dir"] = args.ckpt_dir
    if args.log_jsonl:
        cfg["log"]["jsonl"] = args.log_jsonl
    train(cfg)


if __name__ == "__main__":
    main()
