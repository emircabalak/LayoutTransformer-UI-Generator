import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP,
    SAVE_EVERY, EARLY_STOP_PATIENCE, LR_WARMUP_STEPS, LR_MIN,
    CHECKPOINT_DIR, FIGURES_DIR, PAD_TOKEN,
)

DRIVE_CKPT_DIR = "/content/drive/MyDrive/layout_transformer_v2_checkpoints"


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename):
    path = os.path.join(CHECKPOINT_DIR, filename)

    data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
    }
    torch.save(data, path)
    print(f"  [Checkpoint] Kaydedildi: {path}")

    try:
        import shutil
        os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
        drive_path = os.path.join(DRIVE_CKPT_DIR, filename)
        shutil.copy2(path, drive_path)
        print(f"  [Checkpoint] Drive'a yedeklendi: {drive_path}")
    except Exception as e:
        print(f"  [Checkpoint] Drive yedekleme atlandi: {e}")


def load_checkpoint(model, optimizer, scheduler, filename):
    path = os.path.join(CHECKPOINT_DIR, filename)

    if not os.path.exists(path):
        drive_path = os.path.join(DRIVE_CKPT_DIR, filename)
        if os.path.exists(drive_path):
            import shutil
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            shutil.copy2(drive_path, path)
            print(f"  [Checkpoint] Drive'dan geri yuklendi: {drive_path}")

    if os.path.exists(path):
        try:
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None and checkpoint.get("scheduler_state_dict"):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"  [Checkpoint] Yuklendi: {path} (epoch {checkpoint['epoch']})")
            return checkpoint["epoch"]
        except (RuntimeError, KeyError) as e:
            print(f"  [UYARI] Checkpoint uyumsuz (model mimarisi degismis olabilir): {e}")
            print(f"  [UYARI] Sifirdan egitim baslatiliyor...")
            return 0

    return 0


def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, resume=True):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    total_steps = num_epochs * len(train_loader)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=LR_MIN
    )

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(model, optimizer, lr_scheduler, "layout_transformer_latest.pt")

    if start_epoch >= num_epochs:
        print(f"Egitim zaten tamamlanmis (epoch {start_epoch}/{num_epochs}).")
        return model, [], []

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []
    start_time = time.time()

    global_step = start_epoch * len(train_loader)

    print(f"\n{'='*60}")
    print(f"  LayoutTransformer Egitimi")
    print(f"  Epoch: {start_epoch + 1} -> {num_epochs}")
    print(f"  Batch boyutu: {train_loader.batch_size}")
    print(f"  Egitim ornekleri: {len(train_loader.dataset)}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Egitim]",
                     leave=False)

        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            target_ids = batch["target_ids"].to(DEVICE)
            padding_mask = batch["padding_mask"].to(DEVICE)

            global_step += 1
            if global_step <= LR_WARMUP_STEPS:
                warmup_lr = LEARNING_RATE * global_step / LR_WARMUP_STEPS
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr

            logits = model(input_ids, padding_mask=padding_mask)

            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = target_ids.reshape(-1)
            loss = criterion(flat_logits, flat_targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            if global_step > LR_WARMUP_STEPS:
                lr_scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                target_ids = batch["target_ids"].to(DEVICE)
                padding_mask = batch["padding_mask"].to(DEVICE)

                logits = model(input_ids, padding_mask=padding_mask)
                flat_logits = logits.reshape(-1, logits.size(-1))
                flat_targets = target_ids.reshape(-1)
                loss = criterion(flat_logits, flat_targets)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:>3}/{num_epochs} | "
              f"Egitim: {avg_train_loss:.4f} | "
              f"Dogrulama: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Sure: {int(elapsed)}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, lr_scheduler, epoch + 1,
                            avg_val_loss, "layout_transformer_best.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % SAVE_EVERY == 0 or epoch == num_epochs - 1:
            save_checkpoint(model, optimizer, lr_scheduler, epoch + 1,
                            avg_val_loss, "layout_transformer_latest.pt")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping! {EARLY_STOP_PATIENCE} epoch boyunca iyilesme yok.")
            save_checkpoint(model, optimizer, lr_scheduler, epoch + 1,
                            avg_val_loss, "layout_transformer_latest.pt")
            break

    total_time = time.time() - start_time
    print(f"\n  Egitim tamamlandi! Sure: {int(total_time)}s")
    print(f"  En iyi dogrulama: {best_val_loss:.4f}")

    _save_loss_plot(train_losses, val_losses, start_epoch)

    return model, train_losses, val_losses


def _save_loss_plot(train_losses, val_losses, start_epoch=0):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = range(start_epoch + 1, start_epoch + len(train_losses) + 1)
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0d1117')
        ax.set_facecolor('#1a1a2e')
        ax.plot(epochs, train_losses, 'o-', color='#ff6b6b', label='Egitim', markersize=2)
        ax.plot(epochs, val_losses, 'o-', color='#4ecdc4', label='Dogrulama', markersize=2)
        ax.set_xlabel('Epoch', color='white')
        ax.set_ylabel('Loss (Cross-Entropy)', color='white')
        ax.set_title('LayoutTransformer Egitim Sureci', color='white', fontweight='bold')
        ax.legend(facecolor='#2a2a4a', edgecolor='gray', labelcolor='white')
        ax.tick_params(colors='gray')
        ax.grid(alpha=0.2)
        plt.tight_layout()

        path = os.path.join(FIGURES_DIR, "training_loss.png")
        plt.savefig(path, dpi=150, facecolor='#0d1117')
        plt.close()
        print(f"  [Grafik] Kaydedildi: {path}")
    except Exception as e:
        print(f"  [Grafik] Kaydedilemedi: {e}")
