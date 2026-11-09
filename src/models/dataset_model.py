"""Modèle de classification d'images pour le projet de classification de cellules sanguines (Barcelone + Munich)."""

from __future__ import annotations

import numpy as np
from torch import nn


class DatasetClassifier(nn.Module):
    """CNN maison pour la tâche « dataset » (Barcelone vs Munich)."""

    def __init__(self, img_size=(224, 224), channels=3, nb_conv=2):
        """Construit le CNN à partir de la taille d'image et du nombre de couches conv."""
        super().__init__()
        in_channels = channels
        out_channels = 16
        feature_map_size = np.array(img_size)
        features = np.prod(feature_map_size) * channels

        self.features = nn.Sequential()
        for i in range(nb_conv):
            self.features.add_module(
                f"conv_{i + 1}",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3),
                    padding=1,
                ),
            )
            self.features.add_module(f"maxpool_{i + 1}", nn.MaxPool2d(kernel_size=2))
            self.features.add_module(f"relu_{i + 1}", nn.ReLU())

            feature_map_size = feature_map_size // 2
            features = out_channels * np.prod(feature_map_size)
            in_channels = out_channels
            out_channels *= 2

        self.features.add_module("flatten", nn.Flatten())

        self.classifier = nn.Sequential(
            nn.Linear(features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """Applique le réseau à un batch d'images."""
        return self.classifier(self.features(x))
