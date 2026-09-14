"""Phase 3 reasoning finetuning: start from the pretrained checkpoint, adapt.

Structurally this is the pretraining loop with three deliberate differences,
each of which changes what is being measured:

  1. **Initialisation, not resumption.** ``init_from`` loads the Phase 2
     checkpoint's *weights only* -- no optimizer moments, no scheduler state, no
     step counter. Restoring AdamW's second moments from a run that ended at a
     3e-5 learning rate would suppress the first few hundred finetuning updates
     exactly where the model is being asked to change most. Each model starts
     from its own language's checkpoint; nothing crosses between H and L.

  2. **Loss on the answer span only.** ``ReasoningDataset`` writes
     ``IGNORE_INDEX`` over every prompt position, so gradient flows from the
     answer alone (see that module's docstring). A consequence worth stating:
     validation loss here is a mean over ~4 scored tokens per example, not over
     512, so it is *not* comparable to the pretraining loss curve and is never
     plotted against it.

  3. **Selection on exact match, not on loss.** Loss and accuracy come apart on
     this task -- a model can grow confident about the answer format while
     staying wrong about the comparison. Since exact match is what the brief
     asks to report, it is also what picks the checkpoint. Validation loss is
     still logged, and the gap between the two curves is itself reported.

Resumption is unchanged from pretraining: the finetune writes the same
checkpoint format to its own directory, so a killed Colab or Kaggle session
picks up where it stopped rather than restarting.

Usage:
    python -m src.train.finetune --config hindi/configs/finetune.yaml
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml

from src.eval.reasoning_eval import MAX_NEW, greedy_continuations, score
from src.model.transformer_lm import TransformerLM, TransformerLMConfig
from src.train.checkpoint import (find_resume_point, load_checkpoint,
                                  prune_milestones, save_checkpoint)
from src.train.reasoning_dataset import ReasoningDataset
from src.train.trainer import build_optimizer, build_scheduler, mirror

BEST = "finetune_best.pt"
LATEST = "finetune_latest.pt"


@torch.no_grad()
def validate(model, ds, batch_size: int, device: str, amp_dtype) -> float:
    """Mean cross-entropy over the validation split's answer tokens.

    Args:
        model: The model being finetuned.
        ds: A validation ``ReasoningDataset``.
        batch_size: Examples per forward pass.
        device: Device string.
        amp_dtype: Autocast dtype, or None for full precision.

    Returns:
        Token-weighted mean loss over scored positions. Batches are weighted by
        their scored-token count rather than averaged flat, because padding
        makes batches contain unequal numbers of scored tokens.
    """
    model.eval()
    total, n_tok = 0.0, 0
    from src.model.transformer_lm import IGNORE_INDEX
    for x, y in ds.iter_epoch(batch_size, shuffle=False):
        k = int((y != IGNORE_INDEX).sum())
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            _, loss, _ = model(x, targets=y)
        total += float(loss) * k
        n_tok += k
    model.train()
    return total / max(n_tok, 1)


@torch.no_grad()
def validate_em(model, ds, sp, language: str, answer_style: str, n: int,
                batch_size: int, device: str, eos_id: int, ctx_len: int) -> dict:
    """Greedy exact match on a fixed prefix of the validation split.

    A fixed prefix, not a random sample: the same examples must be scored at
    every checkpoint or the selection metric moves for reasons unrelated to the
    model.

    Args:
        model: The model being finetuned.
        ds: Validation dataset (its ``raw`` records carry gold answers).
        sp: SentencePiece processor.
        language: For answer extraction.
        answer_style: Sets the generation budget.
        n: How many validation examples to decode.
        batch_size: Prompts per forward pass.
        device: Device string.
        eos_id: End-of-sequence id.
        ctx_len: Model context length.

    Returns:
        The dict from ``reasoning_eval.score``, minus the per-example dump.
    """
    model.eval()
    recs = ds.raw[:n]
    conts = greedy_continuations(model, sp, [r["prompt"] for r in recs],
                                 MAX_NEW[answer_style], device, eos_id,
                                 batch_size, ctx_len)
    res = score(recs, conts, language)
    res.pop("_per_example", None)
    model.train()
    return res


def finetune(cfg: dict):
    """Run reasoning finetuning to completion, resuming if interrupted."""
    import sentencepiece as spm

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
    print(f"  model: {model.num_params():,} trainable parameters")

    tok_block = yaml.safe_load(open(cfg["model_config"], encoding="utf-8")) \
        .get("tokenizer", {})
    sp = spm.SentencePieceProcessor(
        model_file=cfg["data"].get("tokenizer", tok_block["model"]))
    eos_id = tok_block.get("eos_id", sp.eos_id())

    # --- data --------------------------------------------------------------
    style = cfg.get("answer_style", "direct")
    ctx = model_cfg.ctx_len
    train_ds = ReasoningDataset(cfg["data"]["train"], sp, ctx, style, seed,
                                device, tok_block.get("bos_id", 2), eos_id,
                                tok_block.get("pad_id", 0))
    val_ds = ReasoningDataset(cfg["data"]["val"], sp, ctx, style, seed, device,
                              tok_block.get("bos_id", 2), eos_id,
                              tok_block.get("pad_id", 0))
    print(f"  answer style: {style}")
    print(f"  train: {json.dumps(train_ds.token_stats())}")
    print(f"  val:   {json.dumps(val_ds.token_stats())}")

    bs = cfg["train"]["batch_size"]
    accum = cfg["train"].get("grad_accum", 1)
    spe = train_ds.steps_per_epoch(bs, accum)
    max_steps = cfg["train"].get("max_steps") or spe * cfg["train"]["epochs"]
    print(f"  {spe:,} steps/epoch x {cfg['train'].get('epochs', '?')} epochs "
          f"= {max_steps:,} steps")

    # --- optimisation ------------------------------------------------------
    o = cfg["optim"]
    optimizer = build_optimizer(model, o["lr"], o["betas"], o["eps"],
                                o["weight_decay"])
    scheduler = build_scheduler(optimizer, cfg["schedule"]["warmup_steps"],
                                max_steps, o["lr"], cfg["schedule"]["min_lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # --- initialise or resume ----------------------------------------------
    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step, best_em, best_val = 0, -1.0, float("inf")

    resume = find_resume_point(ckpt_dir) if (ckpt_dir / LATEST).exists() else None
    if resume:
        meta = load_checkpoint(resume, model, optimizer, scheduler, scaler,
                               map_location="cpu")
        step = meta["step"]
        best_em = meta["config"].get("_best_em", -1.0) if meta["config"] else -1.0
        print(f"  RESUMED finetuning from {resume.name} at step {step:,}")
    else:
        # Weights only. See the module docstring on why the optimizer state of
        # the pretraining run is deliberately discarded.
        load_checkpoint(cfg["init_from"], model, map_location="cpu",
                        restore_rng=False)
        print(f"  initialised from pretrained {cfg['init_from']}")

    # --- logging -----------------------------------------------------------
    log_path = Path(cfg["log"]["jsonl"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_every = cfg["log"].get("every", 20)
    log_remote = cfg["log"].get("mirror_to")
    t_start = t_win = time.time()

    def log(record: dict):
        """Append one JSON record; this file is the finetuning curve."""
        record["wall_clock"] = round(time.time() - t_start, 1)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    log({"event": "start", "answer_style": style, "max_steps": max_steps,
         "steps_per_epoch": spe, "init_from": str(cfg.get("init_from")),
         "train_stats": train_ds.token_stats()})

    # --- loop --------------------------------------------------------------
    ck, ev = cfg["checkpoint"], cfg["eval"]
    remote = ck.get("mirror_to")
    stream = train_ds.iter_steps(bs)
    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"\nfinetuning {step:,} -> {max_steps:,}\n" + "-" * 62)

    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(accum):
            x, y = next(stream)
            with torch.autocast(device_type=device.split(":")[0],
                                dtype=amp_dtype, enabled=use_fp16):
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

        if step % log_every == 0:
            dt = time.time() - t_win
            peak = (torch.cuda.max_memory_allocated() / 2**30
                    if device == "cuda" else 0.0)
            print(f"step {step:>6,}/{max_steps:,}  loss {loss_acc:.4f}  "
                  f"lr {scheduler.get_last_lr()[0]:.2e}  epoch {train_ds.epoch}  "
                  f"grad {grad_norm:.2f}  mem {peak:.1f}G  "
                  f"eta {(max_steps - step) / (log_every / dt) / 60:.1f}m")
            log({"step": step, "train_loss": round(loss_acc, 5),
                 "lr": scheduler.get_last_lr()[0], "epoch": train_ds.epoch,
                 "grad_norm": round(float(grad_norm), 4),
                 "peak_mem_gib": round(peak, 2)})
            t_win = time.time()

        if step % ev["every"] == 0 or step == max_steps:
            vloss = validate(model, val_ds, ev["batch_size"], device, amp_dtype)
            em = validate_em(model, val_ds, sp, cfg["language"], style,
                             ev.get("em_examples", 500), ev["batch_size"],
                             device, eos_id, ctx)
            improved = em["exact_match"] > best_em
            print(f"  [eval] step {step:,}  val_loss {vloss:.4f}  "
                  f"EM {em['exact_match']:.4f}  fmt {em['format_rate']:.3f}"
                  + ("  <- best" if improved else ""))
            log({"step": step, "val_loss": round(vloss, 5),
                 "val_em": em["exact_match"], "val_format_rate":
                 em["format_rate"], "val_em_by_hops": em["by_hops"],
                 "epoch": train_ds.epoch})
            best_val = min(best_val, vloss)
            if improved:
                best_em = em["exact_match"]
                # _best_em rides in the config blob so a resume can recover the
                # selection state, which lives outside the checkpoint schema.
                p = save_checkpoint(ckpt_dir / BEST, model, optimizer,
                                    scheduler, scaler, step, vloss,
                                    {**cfg, "_best_em": best_em},
                                    extra={"val_em": em})
                mirror(p, remote)

        if step % ck["save_latest_every"] == 0:
            p = save_checkpoint(ckpt_dir / LATEST, model, optimizer, scheduler,
                                scaler, step, best_val,
                                {**cfg, "_best_em": best_em})
            mirror(p, remote)
            mirror(log_path, log_remote)

    p = save_checkpoint(ckpt_dir / LATEST, model, optimizer, scheduler, scaler,
                        step, best_val, {**cfg, "_best_em": best_em})
    mirror(p, remote)
    mirror(log_path, log_remote)
    mins = (time.time() - t_start) / 60
    print("-" * 62)
    print(f"done: {step:,} steps, {mins:.1f} min, best val EM {best_em:.4f}")
    log({"event": "done", "steps": step, "minutes": round(mins, 2),
         "best_val_em": best_em})


def main():
    """Parse the CLI and run one finetune."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[],
                    help="dotted key=value pairs, e.g. train.epochs=1")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    for item in args.override:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
        print(f"  override {key} = {node[parts[-1]]!r}")
    finetune(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
