"""Unified training engine with validation-only model selection."""
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from src.evaluation.visualization import plot_history
from .losses import build_loss


def _epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    y_true, y_pred, losses = [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        losses.append(loss.item())
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(1).detach().cpu().tolist())
    return {"loss": sum(losses) / max(1, len(losses)),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def fit(model, train_loader, val_loader, config, checkpoint, device):
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config.get("lr", 0.0001),
                                  weight_decay=config.get("weight_decay", 0.0001))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)
    use_amp = bool(config.get("mixed_precision", False) and str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    class_weights = None
    if config.get("class_weighting", False):
        rows = train_loader.dataset.rows
        mapping = train_loader.dataset.class_to_idx
        counts = torch.zeros(len(mapping), dtype=torch.float32)
        for row in rows:
            counts[mapping[str(row["label"])]] += 1
        class_weights = (counts.sum() / (len(counts) * counts.clamp_min(1))).to(device)
    criterion = build_loss(config, class_weights)
    best, stale, history = -1.0, 0, []
    started = time.time()
    total_epochs = config.get("epochs", 30)
    print("Training on {} for up to {} epochs".format(device, total_epochs), flush=True)
    for epoch in range(total_epochs):
        train = _epoch(model, train_loader, criterion, device, optimizer, scaler)
        val = _epoch(model, val_loader, criterion, device)
        row = {"epoch": epoch + 1, "train": train, "val": val}
        history.append(row)
        scheduler.step(val["macro_f1"])
        print("Epoch {}/{} - train loss {:.4f}, acc {:.4f} | val loss {:.4f}, acc {:.4f}, macro-F1 {:.4f}".format(
            epoch + 1, total_epochs, train["loss"], train["accuracy"],
            val["loss"], val["accuracy"], val["macro_f1"]), flush=True)
        if val["macro_f1"] > best:
            best, stale = val["macro_f1"], 0
            Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "val_macro_f1": best, "config": config}, checkpoint)
        else:
            stale += 1
        if stale >= config.get("patience", 6):
            break
    history_path = Path(checkpoint).with_suffix(".history.json")
    history_path.write_text(json.dumps(
        {"history": history, "training_seconds": time.time() - started}, indent=2), encoding="utf-8")
    plot_history(history, Path("outputs/figures") / (Path(checkpoint).stem + "_curves.png"))
    print("Training complete in {:.1f}s. Best validation macro-F1: {:.4f}. Checkpoint: {}".format(
        time.time() - started, best, checkpoint), flush=True)
    return history
