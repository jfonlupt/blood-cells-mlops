"""Modèle baseline (CNN maison) pour comparer le préprocessing."""

from __future__ import annotations

from torch import nn


class BaselineClassifier(nn.Module):
    """Petit CNN de référence."""

    def __init__(self, num_classes: int):
        """Construit le CNN baseline.

        Args:
            num_classes: Nombre de classes en sortie.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),  # 224x224x16
            nn.MaxPool2d(kernel_size=2),  # 112x112x16
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),  # 112x112x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 56x56x32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),  # 54x54x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 27x27x64
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 27 * 27, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """Applique le réseau à un batch d'images."""
        return self.classifier(self.features(x))
