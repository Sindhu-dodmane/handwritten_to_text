from dataclasses import dataclass
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    num_classes: int
    class_to_idx: dict


def create_dataloaders(
    data_dir: str,
    train_tfms,
    val_tfms,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataBundle:
    train_ds = ImageFolder(root=f"{data_dir}/train", transform=train_tfms)
    val_ds = ImageFolder(root=f"{data_dir}/val", transform=val_tfms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return DataBundle(
        train=train_loader,
        val=val_loader,
        num_classes=len(train_ds.classes),
        class_to_idx=train_ds.class_to_idx,
    )


