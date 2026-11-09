"""Dataset pour le projet de classification de cellules sanguines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

if TYPE_CHECKING:
    import pandas as pd

    from src.tasks import TaskSpec

AUTOTUNE_WORKERS = 4  # a ajuster selon le nombre de coeurs disponibles (os.cpu_count())


class BloodCellDataset(Dataset):
    """Renvoie ``(image, label)`` ; type et forme du label dictés par ``task``."""

    def __init__(self, data: pd.DataFrame, task: TaskSpec, transform=None):
        self.data = data
        self.task = task
        self.transforms = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        with Image.open(row.path) as img:
            image = v2.functional.pil_to_tensor(img.convert("RGB"))
        if self.transforms:
            image = self.transforms(image)
        label = torch.tensor(row[self.task.label_column], dtype=self.task.label_dtype)
        if self.task.target_shape == "vector1":
            label = label.unsqueeze(0)
        return image, label

    def get_details(self, idx: int):
        """Détails de l'image à l'index ``idx``."""
        return self.data.iloc[idx]

    def get_num_classes(self) -> int:
        """Nombre de classes distinctes présentes dans le DataFrame."""
        return self.data[self.task.label_column].nunique()


class DatasetTransform(Dataset):
    """Applique un transform à la volée sur un ``Subset`` / ``Dataset``."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
