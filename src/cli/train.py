import argparse
import os
from pathlib import Path
from typing import Tuple
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")

from ..data.dataset import create_dataloaders
from ..data.kannada_mnist_csv import create_csv_dataloaders
from ..models.cnn import ImprovedKannadaCNN, KannadaCNN
from ..utils.transforms import build_transforms


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, device, scaler, loss_fn, scheduler=None):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = len(loader)
    
    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="train", leave=False)):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            _, logits = model(imgs)
            loss = loss_fn(logits, labels)
            
        scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Step scheduler if it's OneCycleLR
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        total_loss += loss.item() * imgs.size(0)
        total_acc += accuracy_from_logits(logits, labels) * imgs.size(0)
        
        # Log to wandb if available
        if WANDB_AVAILABLE and wandb.run is not None and batch_idx % 50 == 0:
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/batch_acc": accuracy_from_logits(logits, labels),
                "train/lr": optimizer.param_groups[0]['lr']
            })
    
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    all_preds = []
    all_labels = []
    
    for imgs, labels in tqdm(loader, desc="val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        _, logits = model(imgs)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        total_acc += accuracy_from_logits(logits, labels) * imgs.size(0)
        
        # Store predictions for detailed analysis
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset), all_preds, all_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--kaggle_csv", action="store_true", help="Use Kaggle Kannada-MNIST CSV files")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--use_improved_model", action="store_true", help="Use ImprovedKannadaCNN instead of KannadaCNN")
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "onecycle"])
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    args = p.parse_args()

    # Initialize wandb if requested
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="kannada-handwriting-recognition",
            config=vars(args),
            name=f"improved_model_{int(time.time())}"
        )
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Continuing without logging.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enhanced transforms
    train_tfms, val_tfms = build_transforms(args.image_size, grayscale=args.grayscale or True)
    
    # Create data loaders
    if args.kaggle_csv:
        bundle = create_csv_dataloaders(args.data_dir, train_tfms, val_tfms, batch_size=args.batch_size)
    else:
        bundle = create_dataloaders(args.data_dir, train_tfms, val_tfms, batch_size=args.batch_size)

    print(f"Dataset loaded: {bundle.num_classes} classes")
    print(f"Train samples: {len(bundle.train.dataset)}")
    print(f"Val samples: {len(bundle.val.dataset)}")

    # Create model
    if args.use_improved_model:
        model = ImprovedKannadaCNN(
            in_channels=1 if (args.grayscale or True) else 3,
            embedding_dim=args.embedding_dim,
            num_classes=bundle.num_classes
        ).to(device)
        print("Using ImprovedKannadaCNN")
    else:
        model = KannadaCNN(
            in_channels=1 if (args.grayscale or True) else 3,
            embedding_dim=args.embedding_dim,
            num_classes=bundle.num_classes
        ).to(device)
        print("Using KannadaCNN")

    # Enhanced optimizer with better parameters
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate scheduler
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    elif args.scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=args.lr * 10, 
            epochs=args.epochs, 
            steps_per_epoch=len(bundle.train),
            pct_start=0.1
        )

    # Enhanced loss function with label smoothing
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    best_acc = 0.0
    best_path = Path(args.out_dir) / "best_improved.pt"
    start_time = time.time()

    print(f"Starting training for {args.epochs} epochs...")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Training
        tr_loss, tr_acc = train_one_epoch(model, bundle.train, optimizer, device, scaler, loss_fn, scheduler)
        
        # Validation
        va_loss, va_acc, val_preds, val_labels = evaluate(model, bundle.val, device, loss_fn)
        
        # Step scheduler (except for OneCycleLR which steps during training)
        if args.scheduler == "cosine":
            scheduler.step()
        elif args.scheduler == "plateau":
            scheduler.step(va_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Log to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": tr_loss,
                "train/acc": tr_acc,
                "val/loss": va_loss,
                "val/acc": va_acc,
                "lr": optimizer.param_groups[0]['lr'],
                "epoch_time": epoch_time
            })
        
        # Save best model
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "model": model.state_dict(),
                "num_classes": bundle.num_classes,
                "embedding_dim": args.embedding_dim,
                "grayscale": True,
                "architecture": "ImprovedKannadaCNN" if args.use_improved_model else "KannadaCNN",
                "val_acc": va_acc,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
            }, best_path)
            print(f"✅ New best model saved! (acc={best_acc:.4f})")

    total_time = time.time() - start_time
    print(f"\n🎉 Training completed!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🏆 Best validation accuracy: {best_acc:.4f}")
    print(f"💾 Best model saved to: {best_path}")
    
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()


