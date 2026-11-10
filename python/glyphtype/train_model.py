"""Train a compact single-line CTC recognizer on Phase 5 manifests."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import resource

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from glyphtype.dataset import LineDataset
from glyphtype.model import LineRecognizer
from glyphtype.model_data import DEFAULT_CHARSET, MAX_WIDTH, collate_samples


def manifest_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for split in ("train", "validation"):
        digest.update((root / f"{split}.jsonl").read_bytes())
    return digest.hexdigest()


def make_loader(root: Path, split: str, batch_size: int, shuffle: bool,
                max_width: int) -> DataLoader:
    dataset = LineDataset(root, split)
    if not dataset:
        raise ValueError(f"{split} split is empty")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, collate_fn=lambda samples: collate_samples(
                          samples, DEFAULT_CHARSET, max_width))


def run_epoch(model: LineRecognizer, loader: DataLoader, loss_fn: nn.CTCLoss,
              device: torch.device, optimizer: torch.optim.Optimizer | None,
              scaler: torch.amp.GradScaler | None) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    batches = 0
    for images, widths, targets, target_lengths, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        context = torch.autocast("cuda", dtype=torch.float16) if scaler is not None else nullcontext()
        with torch.set_grad_enabled(training), context:
            logits = model(images)
            lengths = model.output_lengths(widths)
            log_probs = logits.log_softmax(-1).permute(1, 0, 2)
            loss = loss_fn(log_probs.float(), targets, lengths, target_lengths)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite CTC loss; inspect labels and image widths")
        if training:
            if scaler is None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
        total += float(loss.detach().cpu())
        batches += 1
    return total / batches


def train(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("batch size, epochs, and patience must be positive")
    if args.max_width < 64 or args.max_width % 4:
        raise ValueError("max width must be at least 64 and divisible by 4")
    if args.hidden_size < 1 or args.learning_rate <= 0:
        raise ValueError("hidden size and learning rate must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    digest = manifest_digest(args.dataset)
    train_loader = make_loader(args.dataset, "train", args.batch_size, True, args.max_width)
    val_loader = make_loader(args.dataset, "validation", args.batch_size, False, args.max_width)
    model = LineRecognizer(len(DEFAULT_CHARSET) + 1, args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=False)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    start_epoch = 1
    best_loss = float("inf")
    bad_epochs = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if (checkpoint["dataset_digest"] != digest or
                checkpoint["charset"] != DEFAULT_CHARSET or
                checkpoint["max_width"] != args.max_width or
                checkpoint["hidden_size"] != args.hidden_size):
            raise ValueError("Resume settings or dataset differ from checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_validation_loss"]
        bad_epochs = checkpoint["bad_epochs"]
    if not args.resume and (args.output / "latest.pt").exists():
        raise ValueError("Output already has a checkpoint; use --resume or a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    history = args.output / "history.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler)
        val_loss = run_epoch(model, val_loader, loss_fn, device, None, None)
        improved = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        bad_epochs = 0 if improved else bad_epochs + 1
        checkpoint = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_validation_loss": best_loss, "bad_epochs": bad_epochs,
            "charset": DEFAULT_CHARSET, "max_width": args.max_width,
            "hidden_size": args.hidden_size, "dataset_digest": digest,
        }
        temporary = args.output / "latest.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(args.output / "latest.pt")
        if improved:
            temporary = args.output / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / "best.pt")
        memory = {"process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        if device.type == "cuda":
            memory["gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch, "train_loss": train_loss,
                                     "validation_loss": val_loss, "best": improved,
                                     **memory}) + "\n")
        print(f"epoch {epoch}: train={train_loss:.4f} validation={val_loss:.4f} "
              f"best={best_loss:.4f}", flush=True)
        if bad_epochs >= args.patience:
            print(f"Early stopping after {bad_epochs} epochs without improvement")
            break


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/synthetic"))
    parser.add_argument("--output", type=Path, default=Path("models/line_ocr"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--max-width", type=int, default=MAX_WIDTH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    try:
        train(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
