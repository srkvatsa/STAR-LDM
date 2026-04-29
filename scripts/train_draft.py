"""Train the speculative decoding draft model.

Trains a tiny transformer (~17M params) to mimic the full STAR-LDM pipeline's
token generation distribution using sequence-level knowledge distillation.

Supports two training modes:
    1. Cross-entropy on generated tokens (default, uses .json shards)
    2. KL divergence against saved logits (better, uses .json + .pt shards)

Usage:
    # Cross-entropy training (mini run, 2k samples)
    python -m scripts.train_draft \
        --data_dir data/draft_training \
        --output_dir checkpoints/draft \
        --epochs 30 --lr 3e-4 --val_split 0.1

    # KL distillation (if logits were saved)
    python -m scripts.train_draft \
        --data_dir data/draft_training \
        --output_dir checkpoints/draft \
        --use_kl --kl_weight 1.0 --epochs 30

    # Resume training
    python -m scripts.train_draft \
        --data_dir data/draft_training \
        --output_dir checkpoints/draft \
        --resume checkpoints/draft/latest.pt
"""

import argparse
import glob
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from star_ldm.decoding.draft_model import DraftTransformerLM, DraftModelConfig


class DraftTrainingDataset(Dataset):
    """Dataset of (prefix_ids, generated_ids) pairs from STAR-LDM.

    Each sample is a concatenation of prefix + generated tokens, trained
    with next-token prediction loss only on the generated portion.
    """

    def __init__(self, data_dir: str, max_seq_len: int = 1024, use_kl: bool = False):
        self.max_seq_len = max_seq_len
        self.use_kl = use_kl
        self.samples = []
        self.logits_cache = {}

        # Load all JSON shards
        shard_files = sorted(glob.glob(os.path.join(data_dir, "shard_*.json")))
        for shard_file in shard_files:
            with open(shard_file) as f:
                shard_data = json.load(f)
            shard_idx = shard_file.split("shard_")[-1].split(".")[0]

            # Load corresponding logits if KL mode
            logits_data = None
            if use_kl:
                logits_path = os.path.join(data_dir, f"shard_{shard_idx}_logits.pt")
                if os.path.exists(logits_path):
                    logits_data = torch.load(logits_path, map_location="cpu", weights_only=True)

            for i, sample in enumerate(shard_data):
                prefix_ids = sample["prefix_ids"]
                gen_ids = sample["generated_ids"]
                if not gen_ids:
                    continue
                self.samples.append({
                    "prefix_ids": prefix_ids,
                    "generated_ids": gen_ids,
                    "shard_idx": shard_idx,
                    "sample_idx": i,
                })
                if logits_data is not None and i < len(logits_data):
                    self.logits_cache[(shard_idx, i)] = logits_data[i]

        print(f"Loaded {len(self.samples)} samples from {len(shard_files)} shards")
        if use_kl:
            print(f"  KL logits available for {len(self.logits_cache)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prefix_ids = sample["prefix_ids"]
        gen_ids = sample["generated_ids"]

        # Concatenate prefix + generated
        all_ids = prefix_ids + gen_ids
        if len(all_ids) > self.max_seq_len:
            all_ids = all_ids[:self.max_seq_len]

        input_ids = torch.tensor(all_ids[:-1], dtype=torch.long)
        labels = torch.tensor(all_ids[1:], dtype=torch.long)

        # Loss mask: only compute loss on generated tokens (not prefix)
        loss_mask = torch.zeros_like(labels, dtype=torch.bool)
        gen_start = len(prefix_ids) - 1
        loss_mask[gen_start:] = True

        result = {"input_ids": input_ids, "labels": labels, "loss_mask": loss_mask}

        if self.use_kl:
            key = (sample["shard_idx"], sample["sample_idx"])
            if key in self.logits_cache:
                teacher_logits = self.logits_cache[key].float()
                gen_len = loss_mask.sum().item()
                if teacher_logits.shape[0] < gen_len:
                    pad = torch.zeros(gen_len - teacher_logits.shape[0], teacher_logits.shape[1])
                    teacher_logits = torch.cat([teacher_logits, pad], dim=0)
                elif teacher_logits.shape[0] > gen_len:
                    teacher_logits = teacher_logits[:gen_len]
                result["teacher_logits"] = teacher_logits

        return result


def collate_fn(batch):
    """Pad sequences to same length within batch."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    padded = {"input_ids": [], "labels": [], "loss_mask": []}
    has_teacher = "teacher_logits" in batch[0]
    if has_teacher:
        padded["teacher_logits"] = []

    for b in batch:
        pad_len = max_len - b["input_ids"].shape[0]
        padded["input_ids"].append(F.pad(b["input_ids"], (0, pad_len), value=0))
        padded["labels"].append(F.pad(b["labels"], (0, pad_len), value=-100))
        padded["loss_mask"].append(F.pad(b["loss_mask"], (0, pad_len), value=False))

        if has_teacher and "teacher_logits" in b:
            tl = b["teacher_logits"]
            gen_max = padded["loss_mask"][-1].sum().item()
            if tl.shape[0] < gen_max:
                tl = F.pad(tl, (0, 0, 0, gen_max - tl.shape[0]))
            padded["teacher_logits"].append(tl)

    result = {
        "input_ids": torch.stack(padded["input_ids"]),
        "labels": torch.stack(padded["labels"]),
        "loss_mask": torch.stack(padded["loss_mask"]),
    }
    if has_teacher and padded["teacher_logits"]:
        result["teacher_logits"] = padded["teacher_logits"]

    return result


def compute_loss(model, batch, device, use_kl=False, kl_weight=1.0, kl_temp=1.0):
    """Compute training loss: cross-entropy + optional KL divergence."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    loss_mask = batch["loss_mask"].to(device)

    output = model(input_ids=input_ids)
    logits = output.logits

    B, T, V = logits.shape
    ce_loss = F.cross_entropy(
        logits.view(-1, V), labels.view(-1),
        ignore_index=-100, reduction="none",
    ).view(B, T)

    masked_ce = (ce_loss * loss_mask.float()).sum() / loss_mask.float().sum().clamp(min=1)

    total_loss = masked_ce
    kl_loss_val = torch.tensor(0.0)

    if use_kl and "teacher_logits" in batch:
        kl_loss_val = _compute_kl_loss(logits, batch, loss_mask, device, kl_temp)
        total_loss = total_loss + kl_weight * kl_loss_val

    return total_loss, masked_ce, kl_loss_val


def _compute_kl_loss(logits, batch, loss_mask, device, temperature):
    """Forward KL: KL(teacher || student)."""
    teacher_list = batch.get("teacher_logits", [])
    if not teacher_list:
        return torch.tensor(0.0, device=device)

    kl_total = torch.tensor(0.0, device=device)
    count = 0

    for b_idx, teacher_logits in enumerate(teacher_list):
        gen_positions = loss_mask[b_idx].nonzero(as_tuple=True)[0]
        if gen_positions.numel() == 0:
            continue

        student_logits_gen = logits[b_idx, gen_positions]
        teacher_logits_gen = teacher_logits[:gen_positions.shape[0]].to(device)

        min_len = min(teacher_logits_gen.shape[0], student_logits_gen.shape[0])
        teacher_logits_gen = teacher_logits_gen[:min_len]
        student_logits_gen = student_logits_gen[:min_len]

        p_teacher = F.softmax(teacher_logits_gen / temperature, dim=-1)
        log_p_student = F.log_softmax(student_logits_gen / temperature, dim=-1)

        kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
        kl_total = kl_total + kl * (temperature ** 2)
        count += 1

    return kl_total / max(count, 1)


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp, amp_dtype, use_kl=False, kl_weight=1.0, kl_temp=1.0):
    """Run validation and return average loss + perplexity."""
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_tokens = 0
    num_batches = 0

    for batch in dataloader:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            total, ce, kl = compute_loss(model, batch, device, use_kl, kl_weight, kl_temp)

        total_loss += total.item()
        total_ce += ce.item()
        total_tokens += batch["loss_mask"].sum().item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_ce = total_ce / max(num_batches, 1)
    ppl = math.exp(min(avg_ce, 20))  # clamp to avoid overflow
    model.train()
    return avg_loss, avg_ce, ppl


def train(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    print(f"Device: {device}")

    # Mixed precision setup
    use_amp = args.amp
    if device.type == "mps":
        amp_dtype = torch.float16  # MPS supports fp16 autocast
    elif device.type == "cuda":
        amp_dtype = torch.bfloat16
    else:
        use_amp = False
        amp_dtype = torch.float32
    if use_amp:
        print(f"Mixed precision: {amp_dtype}")

    # Build model
    config = DraftModelConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    )

    # Resume or init
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        if "config" in ckpt:
            config = DraftModelConfig(**ckpt["config"])
        model = DraftTransformerLM(config)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
    else:
        model = DraftTransformerLM(config)

    model = model.to(device)
    print(f"Draft model: {model.num_parameters:,} params "
          f"({model.num_parameters_no_embedding:,} non-embedding)")

    # torch.compile on supported backends
    if args.compile and device.type in ("cuda", "mps"):
        backend = "inductor" if device.type == "cuda" else "aot_eager"
        print(f"Compiling model with backend={backend}...")
        model = torch.compile(model, backend=backend)

    # Dataset + train/val split
    full_dataset = DraftTrainingDataset(
        args.data_dir, max_seq_len=config.max_seq_len, use_kl=args.use_kl
    )

    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    # Deterministic split
    gen = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=gen).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    print(f"Train: {n_train} samples, Val: {n_val} samples")

    # Use num_workers=0 on MPS to avoid multiprocessing overhead on small datasets
    num_workers = 0 if (device.type == "mps" and n_train < 50000) else args.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type != "mps"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type != "mps"),
        drop_last=False,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )

    # Cosine annealing with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps // 5)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # GradScaler for CUDA AMP (MPS doesn't use scaler)
    use_scaler = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config_dict = {
        "vocab_size": config.vocab_size,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "ffn_dim": config.ffn_dim,
        "max_seq_len": config.max_seq_len,
        "dropout": config.dropout,
        "tie_embeddings": config.tie_embeddings,
    }

    print(f"\nTraining config:")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}, Weight decay: {args.weight_decay}")
    print(f"  Warmup: {warmup_steps} steps, Total: {total_steps} steps")
    print(f"  Steps/epoch: {len(train_loader)}")
    print()

    # Save run config for reproducibility
    run_config = {
        "model": config_dict,
        "training": {
            "data_dir": args.data_dir,
            "n_train": n_train,
            "n_val": n_val,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "grad_clip": args.grad_clip,
            "amp": use_amp,
            "amp_dtype": str(amp_dtype),
            "device": str(device),
            "use_kl": args.use_kl,
            "kl_weight": args.kl_weight,
            "kl_temp": args.kl_temp,
        },
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # Training log (saved to disk every epoch)
    log_path = os.path.join(args.output_dir, "training_log.json")
    training_log = []
    if args.resume and os.path.exists(log_path):
        with open(log_path) as f:
            training_log = json.load(f)

    # Training loop
    global_step = start_epoch * len(train_loader)
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_kl = 0.0
        num_batches = 0
        t0 = time.perf_counter()

        for batch in train_loader:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                total_loss, ce_loss, kl_loss = compute_loss(
                    model, batch, device,
                    use_kl=args.use_kl,
                    kl_weight=args.kl_weight,
                    kl_temp=args.kl_temp,
                )

            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(total_loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()
            epoch_ce += ce_loss.item()
            epoch_kl += kl_loss.item()
            num_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = epoch_loss / num_batches
                avg_ce = epoch_ce / num_batches
                lr = scheduler.get_last_lr()[0]
                msg = (f"  step {global_step:5d} | loss {avg_loss:.4f} | "
                       f"ce {avg_ce:.4f} | ppl {math.exp(min(avg_ce, 20)):.1f} | lr {lr:.2e}")
                if args.use_kl:
                    msg += f" | kl {epoch_kl / num_batches:.4f}"
                print(msg)

        epoch_time = time.perf_counter() - t0
        avg_train_loss = epoch_loss / max(num_batches, 1)
        avg_train_ce = epoch_ce / max(num_batches, 1)
        train_ppl = math.exp(min(avg_train_ce, 20))

        # Validation
        val_loss, val_ce, val_ppl = evaluate(
            model, val_loader, device, use_amp, amp_dtype,
            use_kl=args.use_kl, kl_weight=args.kl_weight, kl_temp=args.kl_temp,
        )

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train_loss {avg_train_loss:.4f} | train_ppl {train_ppl:.1f} | "
              f"val_loss {val_loss:.4f} | val_ppl {val_ppl:.1f} | "
              f"time {epoch_time:.1f}s")

        # Append to training log and save
        training_log.append({
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": avg_train_loss,
            "train_ce": avg_train_ce,
            "train_ppl": train_ppl,
            "val_loss": val_loss,
            "val_ce": val_ce,
            "val_ppl": val_ppl,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_s": epoch_time,
        })
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        # Checkpoint
        ckpt = {
            "model": model.state_dict() if not isinstance(model, torch._dynamo.eval_frame.OptimizedModule)
                     else model._orig_mod.state_dict(),
            "config": config_dict,
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "best_val_loss": min(best_val_loss, val_loss),
        }

        latest_path = os.path.join(args.output_dir, "latest.pt")
        torch.save(ckpt, latest_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(args.output_dir, "best.pt")
            torch.save(ckpt, best_path)
            print(f"  ** New best val_loss: {best_val_loss:.4f} (ppl {val_ppl:.1f})")
        else:
            patience_counter += 1

        if (epoch + 1) % args.save_every == 0:
            epoch_path = os.path.join(args.output_dir, f"epoch_{epoch+1:03d}.pt")
            torch.save(ckpt, epoch_path)

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nEarly stopping after {args.patience} epochs without improvement")
            break

    print(f"\nTraining complete.")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train draft model for speculative decoding")

    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data to use for validation (default 0.1)")

    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (0 = disabled)")

    # Hardware optimization
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Mixed precision (default: on)")
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile (aot_eager on MPS, inductor on CUDA)")

    # KL distillation
    parser.add_argument("--use_kl", action="store_true")
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--kl_temp", type=float, default=2.0)

    # Logging/saving
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
