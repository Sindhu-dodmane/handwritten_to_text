#!/usr/bin/env python3
"""
Simple Kannada Handwriting Recognition Training Script
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torchvision.transforms as T

# Add src to path
sys.path.append('src')

from data.dataset import create_dataloaders
from models.cnn import ImprovedKannadaCNN, KannadaCNN


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, device, scaler, loss_fn):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    
    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="train", leave=False)):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            _, logits = model(imgs)
            loss = loss_fn(logits, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * imgs.size(0)
        total_acc += accuracy_from_logits(logits, labels) * imgs.size(0)
    
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    
    for imgs, labels in tqdm(loader, desc="val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        _, logits = model(imgs)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        total_acc += accuracy_from_logits(logits, labels) * imgs.size(0)
    
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--use_improved_model", action="store_true", help="Use ImprovedKannadaCNN instead of KannadaCNN")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Simple transforms without lambda functions
    train_tfms = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((args.image_size, args.image_size)),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.485], std=[0.229])
    ])
    
    val_tfms = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Create data loaders with num_workers=0
    bundle = create_dataloaders(args.data_dir, train_tfms, val_tfms, batch_size=args.batch_size, num_workers=0)

    print(f"Dataset loaded: {bundle.num_classes} classes")
    print(f"Train samples: {len(bundle.train.dataset)}")
    print(f"Val samples: {len(bundle.val.dataset)}")

    # Create model
    if args.use_improved_model:
        model = ImprovedKannadaCNN(
            in_channels=1,
            embedding_dim=args.embedding_dim,
            num_classes=bundle.num_classes
        ).to(device)
        print("Using ImprovedKannadaCNN")
    else:
        model = KannadaCNN(
            in_channels=1,
            embedding_dim=args.embedding_dim,
            num_classes=bundle.num_classes
        ).to(device)
        print("Using KannadaCNN")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Loss function
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
        tr_loss, tr_acc = train_one_epoch(model, bundle.train, optimizer, device, scaler, loss_fn)
        
        # Validation
        va_loss, va_acc = evaluate(model, bundle.val, device, loss_fn)
        
        # Step scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
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
            }, best_path)
            print(f"✅ New best model saved! (acc={best_acc:.4f})")

    total_time = time.time() - start_time
    print(f"\n🎉 Training completed!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🏆 Best validation accuracy: {best_acc:.4f}")
    print(f"💾 Best model saved to: {best_path}")


if __name__ == "__main__":
    main()

