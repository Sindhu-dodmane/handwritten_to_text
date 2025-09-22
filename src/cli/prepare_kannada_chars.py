import argparse
import os
from pathlib import Path
from typing import List, Tuple
import random
import shutil


def list_images(directory: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files: List[Path] = []
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def stratified_split(filepaths: List[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    files = list(filepaths)
    rng.shuffle(files)
    n_val = int(len(files) * val_ratio)
    val_files = files[:n_val]
    train_files = files[n_val:]
    return train_files, val_files


def copy_files(files: List[Path], dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = dst_dir / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def prepare(src_dir: Path, out_dir: Path, val_ratio: float, seed: int):
    classes = sorted([d.name for d in src_dir.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found in {src_dir}")

    print(f"Found {len(classes)} classes:")
    for i, cls in enumerate(classes):
        print(f"  {i+1:2d}. {cls}")
    
    total_imgs = 0
    for cls in classes:
        cls_dir = src_dir / cls
        imgs = list_images(cls_dir)
        if len(imgs) == 0:
            print(f"warning: no images in {cls_dir}")
            continue
        train_files, val_files = stratified_split(imgs, val_ratio, seed)
        copy_files(train_files, out_dir / "train" / cls)
        copy_files(val_files, out_dir / "val" / cls)
        total_imgs += len(imgs)
        print(f"  {cls}: {len(train_files)} train, {len(val_files)} val")

    print(f"\nPrepared ImageFolder dataset at: {out_dir}")
    print(f"Total images processed: {total_imgs}")
    print(f"Classes: {len(classes)}")


def main():
    p = argparse.ArgumentParser(description="Prepare ImageFolder from Kaggle Kannada Characters dataset")
    p.add_argument("--src", type=str, required=True, help="Path to extracted Kaggle dataset root (folders per class)")
    p.add_argument("--out", type=str, default="data", help="Output root directory for ImageFolder structure")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    prepare(Path(args.src), Path(args.out), args.val_ratio, args.seed)


if __name__ == "__main__":
    main()


