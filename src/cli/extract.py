import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.data.kannada_mnist_csv import create_csv_dataloaders
from src.models.cnn import KannadaCNN
from src.utils.transforms import build_transforms


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    feats, labels = [], []
    for imgs, ys in tqdm(loader, desc="extract"):
        imgs = imgs.to(device)
        z, _ = model(imgs)
        feats.append(z.cpu().numpy())
        labels.append(ys.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--kaggle_csv", action="store_true")
    p.add_argument("--out", type=str, default="features")
    p.add_argument("--image_size", type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tfms, val_tfms = build_transforms(args.image_size, grayscale=True)
    if args.kaggle_csv:
        bundle = create_csv_dataloaders(args.data_dir, train_tfms, val_tfms, batch_size=512, num_workers=2)
    else:
        bundle = create_dataloaders(args.data_dir, train_tfms, val_tfms, batch_size=512, num_workers=2)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = KannadaCNN(in_channels=1, embedding_dim=ckpt["embedding_dim"], num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model"])

    os.makedirs(args.out, exist_ok=True)
    tr_x, tr_y = extract_embeddings(model, bundle.train, device)
    va_x, va_y = extract_embeddings(model, bundle.val, device)

    np.savez_compressed(Path(args.out) / "train.npz", x=tr_x, y=tr_y)
    np.savez_compressed(Path(args.out) / "val.npz", x=va_x, y=va_y)
    print("saved features to", args.out)


if __name__ == "__main__":
    main()


