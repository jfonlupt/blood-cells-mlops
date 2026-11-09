"""Modèle CNN multi-blocs configurable (profondeur/canaux/noyau par bloc)."""

from __future__ import annotations

from torch import nn


class MultiArchiModel(nn.Module):

    def __init__(
        self,
        num_classes: int,
        blocks: list,
        classifier: list,
        channels: int = 3,
        use_batchnorm=True,
        use_gap=True,
        img_size=(224, 224),
    ):
        super().__init__()

        self.blocks = blocks
        self.classifier_sizes = classifier
        self.use_batchnorm = use_batchnorm
        self.use_gap = use_gap

        in_channels = channels
        height, width = img_size
        self.features = nn.Sequential()
        # =====================
        # CNN
        # =====================
        for i, block in enumerate(self.blocks):

            depth = block.depth
            out_channels = block.channels
            kernel = block.kernel

            for j in range(depth):
                self.features.add_module(
                    f"block{i+1}_conv{j+1}",
                    nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2),
                )
                if use_batchnorm:
                    self.features.add_module(f"block{i+1}_bn{j+1}", nn.BatchNorm2d(out_channels))
                self.features.add_module(f"block{i+1}_relu{j+1}", nn.ReLU())
                in_channels = out_channels

            self.features.add_module(f"block{i+1}_pool", nn.MaxPool2d(2))
            height //= 2
            width //= 2

        # =====================
        # CLASSIFIER
        # =====================

        if use_gap:
            self.features.add_module("global_average_pooling", nn.AdaptiveAvgPool2d(1))
            in_features = in_channels
        else:
            in_features = in_channels * height * width

        self.features.add_module("flatten", nn.Flatten())

        self.classifier = nn.Sequential()
        for i, neurons in enumerate(classifier):
            self.classifier.add_module(f"classifier_linear{i+1}", nn.Linear(in_features, neurons))
            self.classifier.add_module(f"classifier_relu{i+1}", nn.ReLU())
            self.classifier.add_module(
                f"classifier_dropout{i+1}", nn.Dropout(0.5 if i == len(classifier) - 1 else 0.2)
            )
            in_features = neurons

        self.classifier.add_module("output", nn.Linear(in_features, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))
