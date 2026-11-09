# Monitoring de data drift (Evidently + MLflow) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Détecter le drift entre les images d'entraînement et de nouvelles images, sans jamais relire les images brutes d'entraînement — une référence compacte (descripteurs visuels + embeddings du modèle) est stockée comme artefact MLflow du run champion, puis comparée via un rapport Evidently.

**Architecture:** Chaque modèle gagne une méthode `.features(x)` (features avant la tête de classification). À l'entraînement, `runner.py` échantillonne 200 images/classe de `train_df`, calcule descripteurs + embeddings, et les écrit dans `run_dir/drift_reference/` (repris automatiquement par le `mlflow.log_artifacts` existant). Un nouveau script CLI (`drift_cli.py`) résout le modèle champion, télécharge cette référence, recalcule les mêmes features sur de nouvelles images, et produit un rapport Evidently local (HTML + JSON).

**Tech Stack:** PyTorch (`nn.Module`), pandas (+ parquet via `pyarrow`, déjà présent), MLflow (registre de modèles + artefacts de run), Evidently (nouvelle dépendance), pytest + `uv run pytest -q`.

**Spec:** `docs/superpowers/specs/2026-10-22-data-drift-monitoring-design.md`

## Global Constraints

- Python `>=3.14`, gestion des dépendances via `uv` (`uv add <pkg>`, jamais d'édition manuelle de `pyproject.toml` pour une nouvelle dépendance).
- Config typée en pydantic v2 (`src/config.py`) — aucune modification de schéma de config n'est nécessaire pour cette feature.
- Style du projet : docstrings/commentaires en français, une ligne sauf si une nuance non évidente le justifie ; `from __future__ import annotations` en tête de chaque module modifié qui ne l'a pas déjà.
- Exceptions larges intentionnelles annotées `# noqa: BLE001` (convention déjà utilisée dans `tracking.py`).
- Tests : `uv run pytest -q` (voir mémoire `python-env` du projet). Un fichier de test miroir par module modifié/créé, dans le même sous-dossier que l'existant (`tests/models/`, `tests/features/`, `tests/experiments/`).
- Le refactor `self.features`/`self.classifier` (Tâches 2-4) change les clés de `state_dict` : les champions déjà enregistrés dans MLflow pour `multi_archi`/`baseline`/`dataset` devront être ré-entraînés/re-loggués après ce changement — limitation actée dans la spec, à ne pas tenter de corriger.

## Review Focus

- **Colonne `path` traitée comme une feature par Evidently** : `descriptors.parquet`/`embeddings.parquet` portent une colonne `path` utilisée pour aligner les deux DataFrames par fusion — si elle entre dans le `Report` Evidently comme colonne numérique/catégorielle ordinaire, elle pollue (voire fait planter) `DataDriftPreset` (texte à cardinalité quasi unique). Fixé en Tâche 13 : `path` sert à la fusion puis est explicitement retirée (`.drop(columns=["path"])`) avant `report.run()`, avec un `ColumnMapping(numerical_features=...)` qui ne la liste jamais non plus.
- **Image illisible parmi les « nouvelles données »** : une image corrompue ou dans un format inattendu ne doit pas produire une trace PIL opaque au milieu d'une boucle — l'erreur doit nommer le fichier en cause. Fixé en Tâche 6 (`compute_descriptors`).
- **Classe sous-représentée dans `train_df`** (ex. une classe critique avec moins de 200 images) : `sample_stratified` ne doit jamais lever, juste prendre tout ce qui est disponible. Fixé et testé en Tâche 7.
- **Référence de drift absente d'un run antérieur à cette fonctionnalité** : le script de consommation doit échouer avec un message explicite plutôt que produire un rapport vide ou une exception MLflow opaque. Fixé en Tâche 13 (`_load_required_reference`).
- **Dossier d'images vide passé au CLI** (`--images` sur un dossier sans fichiers) : doit échouer tôt avec un message clair plutôt que de laisser Evidently échouer loin de la cause. Fixé en Tâche 13 (garde dans `run_drift_report`).

---

## Task 1: `TransferModel.features(x)`

**Files:**
- Modify: `src/models/transfer.py` (après `forward`, ligne 117-119)
- Test: `tests/models/test_transfer.py`

**Interfaces:**
- Produces: `TransferModel.features(x: torch.Tensor) -> torch.Tensor`, forme `(B, backbone.num_features)`.

- [ ] **Step 1: Write the failing test**

Ajouter à `tests/models/test_transfer.py` :

```python
def test_features_returns_backbone_output_before_head():
    model = TransferModel(_cfg(), num_classes=10)
    out = model.features(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, model.backbone.num_features)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_transfer.py::test_features_returns_backbone_output_before_head -v`
Expected: FAIL avec `AttributeError: 'TransferModel' object has no attribute 'features'`.

- [ ] **Step 3: Write minimal implementation**

Dans `src/models/transfer.py`, juste après `forward` :

```python
    def forward(self, x):
        """Features du backbone -> tête de classification."""
        return self.head(self.backbone(x))

    def features(self, x):
        """Vecteur de features du backbone, avant la tête de classification."""
        return self.backbone(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_transfer.py -v`
Expected: PASS (tous les tests du fichier, y compris les existants).

- [ ] **Step 5: Commit**

```bash
git add src/models/transfer.py tests/models/test_transfer.py
git commit -m "feat: ajoute TransferModel.features() pour la référence de drift"
```

---

## Task 2: `BaselineClassifier` — refactor `features`/`classifier` + `.features(x)`

**Files:**
- Modify: `src/models/baseline.py`
- Create: `tests/models/test_baseline.py`

**Interfaces:**
- Produces: `BaselineClassifier.features(x: torch.Tensor) -> torch.Tensor`, forme `(B, 64*27*27)`. `forward(x)` inchangé en sortie (`(B, num_classes)`).

- [ ] **Step 1: Write the failing test**

Créer `tests/models/test_baseline.py` :

```python
"""Modèle baseline : forward inchangé après le refactor features/classifier."""

import torch

from src.models.baseline import BaselineClassifier


def test_forward_output_shape_unchanged():
    model = BaselineClassifier(num_classes=10)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 10)


def test_features_returns_flattened_vector_before_classifier():
    model = BaselineClassifier(num_classes=10)
    feats = model.features(torch.randn(2, 3, 224, 224))
    assert feats.shape == (2, 64 * 27 * 27)


def test_classifier_applied_to_features_matches_forward():
    model = BaselineClassifier(num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        direct = model(x)
        via_features = model.classifier(model.features(x))
    assert torch.equal(direct, via_features)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_baseline.py -v`
Expected: FAIL (`AttributeError: 'BaselineClassifier' object has no attribute 'features'`).

- [ ] **Step 3: Write minimal implementation**

Remplacer le contenu de `src/models/baseline.py` :

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_baseline.py tests/models/test_build.py -v`
Expected: PASS (y compris `test_models_are_plain_nn_module` dans `test_build.py`, non affecté par ce refactor interne).

- [ ] **Step 5: Commit**

```bash
git add src/models/baseline.py tests/models/test_baseline.py
git commit -m "refactor: isole features/classifier dans BaselineClassifier"
```

---

## Task 3: `DatasetClassifier` — refactor `features`/`classifier` + `.features(x)`

**Files:**
- Modify: `src/models/dataset_model.py`
- Create: `tests/models/test_dataset_model.py`

**Interfaces:**
- Produces: `DatasetClassifier.features(x: torch.Tensor) -> torch.Tensor`. `forward(x)` inchangé (`(B, 1)`).

- [ ] **Step 1: Write the failing test**

Créer `tests/models/test_dataset_model.py` :

```python
"""Modèle dataset (Barcelone vs Munich) : forward inchangé après le refactor features/classifier."""

import torch

from src.models.dataset_model import DatasetClassifier


def test_forward_output_shape_unchanged():
    model = DatasetClassifier(img_size=(224, 224))
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1)


def test_features_returns_vector_before_classifier():
    model = DatasetClassifier(img_size=(224, 224))
    feats = model.features(torch.randn(2, 3, 224, 224))
    assert feats.ndim == 2
    assert feats.shape[0] == 2


def test_classifier_applied_to_features_matches_forward():
    model = DatasetClassifier(img_size=(224, 224))
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        direct = model(x)
        via_features = model.classifier(model.features(x))
    assert torch.equal(direct, via_features)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_dataset_model.py -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Write minimal implementation**

Remplacer le contenu de `src/models/dataset_model.py` :

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_dataset_model.py tests/models/test_build.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/dataset_model.py tests/models/test_dataset_model.py
git commit -m "refactor: isole features/classifier dans DatasetClassifier"
```

---

## Task 4: `MultiArchiModel` — refactor `features`/`classifier` + `.features(x)`

**Files:**
- Modify: `src/models/multi_archi_model.py`
- Create: `tests/models/test_multi_archi.py`

**Interfaces:**
- Produces: `MultiArchiModel.features(x: torch.Tensor) -> torch.Tensor`. `forward(x)` inchangé.

- [ ] **Step 1: Write the failing test**

Créer `tests/models/test_multi_archi.py` :

```python
"""Modèle multi-blocs : forward inchangé après le refactor features/classifier."""

import torch

from src.models.multi_archi_model import MultiArchiModel


def _block(depth=1, channels=8, kernel=3):
    from types import SimpleNamespace

    return SimpleNamespace(depth=depth, channels=channels, kernel=kernel)


def test_forward_output_shape_with_gap():
    model = MultiArchiModel(
        num_classes=10, blocks=[_block()], classifier=[16], img_size=(32, 32), use_gap=True
    )
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_forward_output_shape_without_gap():
    model = MultiArchiModel(
        num_classes=10, blocks=[_block()], classifier=[16], img_size=(32, 32), use_gap=False
    )
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_features_returns_vector_before_classifier():
    model = MultiArchiModel(
        num_classes=10, blocks=[_block()], classifier=[16], img_size=(32, 32), use_gap=True
    )
    feats = model.features(torch.randn(2, 3, 32, 32))
    assert feats.shape == (2, 8)  # use_gap=True -> (B, out_channels du dernier bloc)


def test_classifier_applied_to_features_matches_forward():
    model = MultiArchiModel(
        num_classes=10, blocks=[_block()], classifier=[16], img_size=(32, 32), use_gap=True
    )
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        direct = model(x)
        via_features = model.classifier(model.features(x))
    assert torch.equal(direct, via_features)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_multi_archi.py -v`
Expected: FAIL (`AttributeError: 'MultiArchiModel' object has no attribute 'features'`).

- [ ] **Step 3: Write minimal implementation**

Remplacer le contenu de `src/models/multi_archi_model.py` :

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_multi_archi.py tests/test_config.py -v`
Expected: PASS. (`tests/test_config.py` référence `MultiArchiModel` indirectement via la config — vérifier qu'aucune assertion ne dépend des anciens noms `model.block1_conv1...`.)

- [ ] **Step 5: Commit**

```bash
git add src/models/multi_archi_model.py tests/models/test_multi_archi.py
git commit -m "refactor: isole features/classifier dans MultiArchiModel"
```

---

## Task 5: Contrat `.features()` commun aux 4 modèles

**Files:**
- Modify: `tests/models/test_build.py`

**Interfaces:**
- Consumes: `build_model` (déjà existant), `.features(x)` sur chacune des 4 classes (Tâches 1-4).

- [ ] **Step 1: Write the failing test**

Ajouter à `tests/models/test_build.py` :

```python
import pytest


@pytest.mark.parametrize(
    ("name", "task_name"),
    [("baseline", "type"), ("dataset", "dataset"), ("multi_archi", "type"), ("transfer", "type")],
)
def test_every_model_exposes_features_before_head(name, task_name):
    """Filet de régression : les 4 modèles exposent .features(), testé en détail par modèle
    dans test_baseline.py/test_dataset_model.py/test_multi_archi.py/test_transfer.py
    (égalité classifier(features(x)) == forward(x)) — ici on verrouille juste le contrat
    commun sur toute la fabrique build_model.
    """
    from src.config import ArchitectureConfig, BlockConfig, TransferConfig

    kwargs = {}
    if name == "multi_archi":
        kwargs["architecture"] = ArchitectureConfig(
            blocks=[BlockConfig(depth=1, channels=8, kernel=3)], classifier=[16]
        )
    if name == "transfer":
        kwargs["transfer"] = TransferConfig(backbone="resnet18", pretrained=False)

    model = build_model(ModelConfig(name=name, **kwargs), get_task(task_name))
    assert hasattr(model, "features")
    x = torch.randn(2, 3, 224, 224)
    feats = model.features(x)
    assert feats.shape[0] == 2
    assert feats.ndim == 2
    assert feats.shape[1] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_build.py -v -k features`
Expected: PASS directement si les Tâches 1-4 sont bien faites (ce test est un filet de sécurité de régression, pas un test TDD classique "rouge d'abord" — s'il échoue, une des tâches précédentes est incomplète).

- [ ] **Step 3: (rien à implémenter — ce test valide les Tâches 1-4)**

- [ ] **Step 4: Run full model test suite**

Run: `uv run pytest tests/models/ -v`
Expected: PASS (tous les fichiers de `tests/models/`).

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_build.py
git commit -m "test: verrouille le contrat .features() commun aux 4 modèles"
```

---

## Task 6: `compute_descriptors` — descripteurs visuels bas-niveau

**Files:**
- Create: `src/features/drift_descriptors.py`
- Test: `tests/features/test_drift_descriptors.py`

**Interfaces:**
- Produces: `compute_descriptors(image_paths: Sequence[str | Path], img_size: tuple[int, int] = IMG_SIZE) -> pd.DataFrame` avec colonnes `path, width, height, aspect_ratio, brightness, sharpness, hist_r_0..N, hist_g_0..N, hist_b_0..N`.

- [ ] **Step 1: Write the failing test**

Créer `tests/features/test_drift_descriptors.py` :

```python
"""Descripteurs visuels bas-niveau pour la référence de drift."""

import numpy as np
import pytest
from PIL import Image

from src.features.drift_descriptors import compute_descriptors


def _solid_image(path, color, size=(50, 40)):
    """Image unie de couleur ``color`` (R, G, B), taille ``(w, h)``."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = color
    Image.fromarray(arr).save(path)
    return str(path)


def test_uniform_red_image_has_expected_brightness_and_dims(tmp_path):
    path = _solid_image(tmp_path / "red.png", (255, 0, 0), size=(50, 40))
    df = compute_descriptors([path])
    row = df.iloc[0]
    assert row["width"] == 50
    assert row["height"] == 40
    assert row["aspect_ratio"] == pytest.approx(50 / 40)
    assert row["brightness"] == pytest.approx(1 / 3, abs=0.02)
    assert row["sharpness"] == pytest.approx(0.0, abs=1e-4)  # image unie : pas d'arête


def test_uniform_image_histogram_concentrated_in_expected_bins(tmp_path):
    path = _solid_image(tmp_path / "red.png", (255, 0, 0), size=(20, 20))
    df = compute_descriptors([path])
    row = df.iloc[0]
    assert row["hist_r_7"] == pytest.approx(1.0, abs=1e-6)  # canal R au max -> dernier bin
    assert row["hist_g_0"] == pytest.approx(1.0, abs=1e-6)  # canal G au min -> premier bin
    assert row["hist_b_0"] == pytest.approx(1.0, abs=1e-6)


def test_unreadable_image_raises_clear_error(tmp_path):
    bad = tmp_path / "not_an_image.png"
    bad.write_text("ceci n'est pas une image")
    with pytest.raises(OSError, match=str(bad)):
        compute_descriptors([str(bad)])


def test_output_has_one_row_per_image(tmp_path):
    paths = [_solid_image(tmp_path / f"img_{i}.png", (i * 10, 0, 0)) for i in range(5)]
    df = compute_descriptors(paths)
    assert len(df) == 5
    assert list(df["path"]) == paths
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/features/test_drift_descriptors.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.features.drift_descriptors'`).

- [ ] **Step 3: Write minimal implementation**

Créer `src/features/drift_descriptors.py` :

```python
"""Descripteurs visuels bas-niveau pour la référence de drift (sans Reinhard/CLAHE/Normalize).

Calculés sur un transform minimal (resize + mise à l'échelle [0,1]) pour ne
pas masquer un drift de coloration/scanner par la normalisation colorimétrique
appliquée au reste du pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch.nn import functional as F

from src.constants import IMG_SIZE
from src.features.transform import ToUnitRange, load_image_rgb, resize_with_pad

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_N_HIST_BINS = 8
_LAPLACIAN_KERNEL = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def _sharpness(gray: torch.Tensor) -> float:
    """Variance du Laplacien (netteté) sur une image en niveaux de gris, ``(H, W)`` dans ``[0, 1]``."""
    kernel = _LAPLACIAN_KERNEL.view(1, 1, 3, 3)
    laplacian = F.conv2d(gray.unsqueeze(0).unsqueeze(0), kernel, padding=1)
    return laplacian.var().item()


def _color_histogram(image_01: torch.Tensor, n_bins: int = _N_HIST_BINS) -> dict:
    """Histogramme couleur normalisé par canal (``image_01`` : CHW float ``[0, 1]``)."""
    out = {}
    for channel, name in enumerate("rgb"):
        hist = torch.histc(image_01[channel], bins=n_bins, min=0.0, max=1.0)
        hist = hist / hist.sum().clamp(min=1.0)
        for i, value in enumerate(hist.tolist()):
            out[f"hist_{name}_{i}"] = value
    return out


def compute_descriptors(
    image_paths: Sequence[str | Path], img_size: tuple[int, int] = IMG_SIZE
) -> pd.DataFrame:
    """Descripteurs visuels bas-niveau, un par image de ``image_paths``.

    Args:
        image_paths: Chemins des images à décrire.
        img_size: Taille ``(H, W)`` à laquelle les images sont redimensionnées
            avant le calcul de la luminosité/histogramme/netteté (les
            dimensions d'origine sont, elles, gardées telles quelles dans
            ``width``/``height``).

    Returns:
        Un DataFrame avec une colonne ``path`` + les descripteurs numériques.

    Raises:
        OSError: si une image ne peut pas être lue, avec son chemin dans le message.
    """
    rows = []
    for path in image_paths:
        try:
            raw = load_image_rgb(path)  # CHW uint8
        except Exception as exc:  # noqa: BLE001 -- on reformule avec le chemin en cause
            raise OSError(f"image illisible : {path}") from exc
        _, orig_h, orig_w = raw.shape
        resized = resize_with_pad(raw, img_size, fill=255)
        image_01 = ToUnitRange()(resized)
        gray = image_01.mean(dim=0)
        row = {
            "path": str(path),
            "width": orig_w,
            "height": orig_h,
            "aspect_ratio": orig_w / orig_h,
            "brightness": image_01.mean().item(),
            "sharpness": _sharpness(gray),
        }
        row.update(_color_histogram(image_01))
        rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/features/test_drift_descriptors.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/features/drift_descriptors.py tests/features/test_drift_descriptors.py
git commit -m "feat: ajoute compute_descriptors (descripteurs visuels bas-niveau)"
```

---

## Task 7: `sample_stratified` — échantillon équilibré par classe

**Files:**
- Create: `src/experiments/drift_reference.py`
- Test: `tests/experiments/test_drift_reference.py`

**Interfaces:**
- Produces: `sample_stratified(df: pd.DataFrame, label_column: str, n_per_class: int, seed: int) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test**

Créer `tests/experiments/test_drift_reference.py` :

```python
"""Construction de la référence de drift (échantillonnage + descripteurs + embeddings)."""

import pandas as pd

from src.experiments.drift_reference import sample_stratified


def _df(counts: dict) -> pd.DataFrame:
    rows = []
    for label, n in counts.items():
        rows += [{"label": label, "path": f"/fake/{label}_{i}.png"} for i in range(n)]
    return pd.DataFrame(rows)


def test_caps_at_n_per_class():
    df = _df({"a": 300, "b": 150})
    sample = sample_stratified(df, "label", n_per_class=200, seed=0)
    counts = sample["label"].value_counts()
    assert counts["a"] == 200
    assert counts["b"] == 150


def test_class_with_fewer_rows_than_requested_keeps_all():
    df = _df({"a": 5, "b": 300})
    sample = sample_stratified(df, "label", n_per_class=200, seed=0)
    assert (sample["label"] == "a").sum() == 5
    assert (sample["label"] == "b").sum() == 200


def test_deterministic_with_fixed_seed():
    df = _df({"a": 300})
    first = sample_stratified(df, "label", n_per_class=50, seed=42)
    second = sample_stratified(df, "label", n_per_class=50, seed=42)
    assert list(first["path"]) == list(second["path"])


def test_different_seeds_can_give_different_samples():
    df = _df({"a": 300})
    first = sample_stratified(df, "label", n_per_class=50, seed=1)
    second = sample_stratified(df, "label", n_per_class=50, seed=2)
    assert list(first["path"]) != list(second["path"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments/test_drift_reference.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

Créer `src/experiments/drift_reference.py` :

```python
"""Construction de la référence de drift (échantillon + descripteurs + embeddings) d'un run."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def sample_stratified(
    df: pd.DataFrame, label_column: str, n_per_class: int, seed: int
) -> pd.DataFrame:
    """Échantillon équilibré par classe : jusqu'à ``n_per_class`` lignes par valeur de ``label_column``.

    Une classe comptant moins de ``n_per_class`` lignes est entièrement conservée
    (pas d'erreur, pas de sur-échantillonnage).

    Args:
        df: DataFrame source (typiquement ``train_df``).
        label_column: Colonne de classe (``task.label_column``).
        n_per_class: Nombre maximal de lignes tirées par classe.
        seed: Graine du tirage, pour la reproductibilité.

    Returns:
        Le sous-DataFrame concaténé, dans l'ordre des classes trié.
    """
    parts = [
        group.sample(n=min(n_per_class, len(group)), random_state=seed)
        for _, group in df.groupby(label_column, sort=True)
    ]
    return pd.concat(parts, ignore_index=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/experiments/test_drift_reference.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/experiments/drift_reference.py tests/experiments/test_drift_reference.py
git commit -m "feat: ajoute sample_stratified pour la référence de drift"
```

---

## Task 8: `build_drift_reference` — descripteurs + embeddings (skip gracieux)

**Files:**
- Modify: `src/experiments/drift_reference.py`
- Modify: `tests/experiments/test_drift_reference.py`

**Interfaces:**
- Consumes: `compute_descriptors` (Tâche 6), `sample_stratified` (Tâche 7).
- Produces: `DriftReference` (dataclass, champs `descriptors: pd.DataFrame`, `embeddings: pd.DataFrame | None`) ; `build_drift_reference(sample_df, eval_transform, model, device="cpu", batch_size=32) -> DriftReference`.

- [ ] **Step 1: Write the failing test**

Ajouter à `tests/experiments/test_drift_reference.py` :

```python
import numpy as np
import torch
from PIL import Image
from torch import nn

from src.experiments.drift_reference import DriftReference, build_drift_reference


def _image_df(tmp_path, n=6):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(np.uint8(rng.random((30, 30, 3)) * 255)).save(p)
        rows.append({"path": str(p), "label": i % 2})
    return pd.DataFrame(rows)


class _ModelWithFeatures(nn.Module):
    def __init__(self, out_dim=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = out_dim

    def features(self, x):
        pooled = self.pool(x).flatten(1)  # (B, 3)
        return pooled.repeat(1, self.out_dim // 3 + 1)[:, : self.out_dim]

    def forward(self, x):
        return self.features(x)


class _ModelWithoutFeatures(nn.Module):
    def forward(self, x):
        return x


def _identity_transform(image):
    return image.float() / 255.0


def test_build_drift_reference_includes_embeddings_when_model_supports_features(tmp_path):
    df = _image_df(tmp_path)
    ref = build_drift_reference(df, _identity_transform, _ModelWithFeatures(), device="cpu")
    assert isinstance(ref, DriftReference)
    assert len(ref.descriptors) == 6
    assert ref.embeddings is not None
    assert len(ref.embeddings) == 6
    assert "path" in ref.embeddings.columns
    emb_cols = [c for c in ref.embeddings.columns if c != "path"]
    assert len(emb_cols) == 4


def test_build_drift_reference_skips_embeddings_without_features(tmp_path):
    df = _image_df(tmp_path)
    ref = build_drift_reference(df, _identity_transform, _ModelWithoutFeatures(), device="cpu")
    assert ref.embeddings is None
    assert len(ref.descriptors) == 6


def test_build_drift_reference_batches_large_samples(tmp_path):
    df = _image_df(tmp_path, n=6)
    ref = build_drift_reference(
        df, _identity_transform, _ModelWithFeatures(), device="cpu", batch_size=2
    )
    assert len(ref.embeddings) == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments/test_drift_reference.py -v -k build_drift_reference`
Expected: FAIL (`ImportError: cannot import name 'DriftReference'`).

- [ ] **Step 3: Write minimal implementation**

Ajouter à `src/experiments/drift_reference.py` (en gardant `sample_stratified` existant) :

```python
from typing import TYPE_CHECKING

import torch

from src.features.drift_descriptors import compute_descriptors
from src.features.transform import load_image_rgb

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn


@dataclass
class DriftReference:
    """Référence de drift d'un run : descripteurs visuels + embeddings optionnels."""

    descriptors: pd.DataFrame
    embeddings: pd.DataFrame | None


def build_drift_reference(
    sample_df: pd.DataFrame,
    eval_transform: Callable,
    model: nn.Module,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
) -> DriftReference:
    """Descripteurs (toujours) + embeddings (si ``model`` expose ``.features()``).

    Args:
        sample_df: DataFrame échantillonné (colonne ``path``) — voir ``sample_stratified``.
        eval_transform: Transform d'évaluation complet (celui utilisé en inférence).
        model: Modèle dont on extrait les features ; ignoré si pas de ``.features()``.
        device: Device sur lequel exécuter ``model.features``.
        batch_size: Taille de batch pour éviter de charger tout l'échantillon en mémoire.

    Returns:
        La ``DriftReference`` ; ``embeddings`` est ``None`` si ``model`` n'a pas ``.features()``.
    """
    descriptors = compute_descriptors(sample_df["path"].tolist())
    if not hasattr(model, "features"):
        return DriftReference(descriptors=descriptors, embeddings=None)

    paths = sample_df["path"].tolist()
    was_training = model.training
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch = torch.stack(
                [eval_transform(load_image_rgb(p)) for p in batch_paths]
            ).to(device)
            chunks.append(model.features(batch).cpu())
    model.train(was_training)

    features = torch.cat(chunks, dim=0).numpy()
    embeddings = pd.DataFrame(features, columns=[f"emb_{i}" for i in range(features.shape[1])])
    embeddings.insert(0, "path", paths)
    return DriftReference(descriptors=descriptors, embeddings=embeddings)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/experiments/test_drift_reference.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/experiments/drift_reference.py tests/experiments/test_drift_reference.py
git commit -m "feat: ajoute build_drift_reference (descripteurs + embeddings, skip gracieux)"
```

---

## Task 9: Persistance dans `run_dir/drift_reference/` (`artifacts.py`)

**Files:**
- Modify: `src/experiments/artifacts.py`
- Modify: `tests/experiments/test_artifacts.py`

**Interfaces:**
- Consumes: `DriftReference` (Tâche 8, import `TYPE_CHECKING` seulement).
- Produces: `RunResult.drift_reference: DriftReference | None = None` ; `write_run_dir` écrit `run_dir/drift_reference/descriptors.parquet` (+ `embeddings.parquet` si présent).

- [ ] **Step 1: Write the failing test**

Ajouter à `tests/experiments/test_artifacts.py` :

```python
import pandas as pd

from src.experiments.drift_reference import DriftReference


def test_write_run_dir_writes_drift_reference_descriptors(tmp_path):
    result = _result(tmp_path)
    result.drift_reference = DriftReference(
        descriptors=pd.DataFrame({"path": ["a.png"], "brightness": [0.5]}), embeddings=None
    )
    run_dir = write_run_dir(result, tmp_path)
    df = pd.read_parquet(run_dir / "drift_reference" / "descriptors.parquet")
    assert df["brightness"].iloc[0] == 0.5
    assert not (run_dir / "drift_reference" / "embeddings.parquet").exists()


def test_write_run_dir_writes_drift_reference_embeddings_when_present(tmp_path):
    result = _result(tmp_path)
    result.drift_reference = DriftReference(
        descriptors=pd.DataFrame({"path": ["a.png"], "brightness": [0.5]}),
        embeddings=pd.DataFrame({"path": ["a.png"], "emb_0": [1.0]}),
    )
    run_dir = write_run_dir(result, tmp_path)
    df = pd.read_parquet(run_dir / "drift_reference" / "embeddings.parquet")
    assert df["emb_0"].iloc[0] == 1.0


def test_write_run_dir_skips_drift_reference_dir_when_absent(tmp_path):
    run_dir = write_run_dir(_result(tmp_path), tmp_path)
    assert not (run_dir / "drift_reference").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments/test_artifacts.py -v -k drift_reference`
Expected: FAIL (`AttributeError: 'RunResult' object has no attribute 'drift_reference'`).

- [ ] **Step 3: Write minimal implementation**

Dans `src/experiments/artifacts.py` :

1. Ajouter l'import de type et le champ au `RunResult` :

```python
if TYPE_CHECKING:
    from src.config import ExperimentConfig
    from src.experiments.drift_reference import DriftReference
    from src.tasks import TaskSpec
```

```python
@dataclass
class RunResult:
    ...
    n_trainable_params: int | None = None
    drift_reference: DriftReference | None = None
```

2. Dans `write_run_dir`, après le bloc `if result.preprocessing is not None:` :

```python
    if result.drift_reference is not None:
        drift_dir = run_dir / "drift_reference"
        drift_dir.mkdir(parents=True, exist_ok=True)
        result.drift_reference.descriptors.to_parquet(drift_dir / "descriptors.parquet", index=False)
        if result.drift_reference.embeddings is not None:
            result.drift_reference.embeddings.to_parquet(
                drift_dir / "embeddings.parquet", index=False
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/experiments/test_artifacts.py -v`
Expected: PASS (tous les tests du fichier).

- [ ] **Step 5: Commit**

```bash
git add src/experiments/artifacts.py tests/experiments/test_artifacts.py
git commit -m "feat: persiste la référence de drift dans run_dir/drift_reference/"
```

---

## Task 10: Hook dans `runner.py`

**Files:**
- Modify: `src/experiments/runner.py`
- Modify: `tests/experiments/test_runner_smoke.py`

**Interfaces:**
- Consumes: `sample_stratified`, `build_drift_reference` (Tâches 7-8), `build_transform` (`src/features/transform.py`, déjà existant).
- Produces: `result.drift_reference` renseigné avant `write_run_dir`.

- [ ] **Step 1: Write the failing test**

Ajouter à `tests/experiments/test_runner_smoke.py` (ce fichier est déjà gardé par le `pytestmark` skipif données) :

```python
def test_holdout_run_writes_drift_reference(tmp_path):
    cfg = ExperimentConfig.model_validate(
        {
            "task": "type",
            "seed": 0,
            "data": {"n_images": 120, "stratify": "type", "test_size": 0.25},
            "transform": {"reinhard": False, "clahe": False, "jpeg": False},
            "train": {"nb_epoch": 1, "batch_size": 16, "validation": "holdout", "val_size": 0.25},
            "output": {"root": str(tmp_path), "save_checkpoint": True},
        }
    )
    result = run_experiment(cfg, output_root=tmp_path)
    assert (result.run_dir / "drift_reference" / "descriptors.parquet").exists()
    assert (result.run_dir / "drift_reference" / "embeddings.parquet").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments/test_runner_smoke.py::test_holdout_run_writes_drift_reference -v`
Expected: FAIL (`AssertionError`, le fichier n'existe pas) — ou SKIPPED si les données ne sont pas présentes localement (comportement attendu du fichier, décrit dans son en-tête).

- [ ] **Step 3: Write minimal implementation**

Dans `src/experiments/runner.py` :

1. Ajouter les imports :

```python
from src.experiments.drift_reference import build_drift_reference, sample_stratified
from src.features.transform import build_transform
```

2. Entre le calcul de `result.preprocessing` et `write_run_dir(result, root, config.verbose)` (juste avant la ligne `write_run_dir(result, root, config.verbose)`) :

```python
    eval_transform = build_transform(
        config.transform,
        training=False,
        reinhard_normalizer=normalizer,
        dataset_mean=tb.dataset_mean,
        dataset_std=tb.dataset_std,
    )
    drift_sample = sample_stratified(train_df, task.label_column, n_per_class=200, seed=config.seed)
    result.drift_reference = build_drift_reference(
        drift_sample, eval_transform, best_model, device=device
    )

    write_run_dir(result, root, config.verbose)
```

(Supprimer l'ancien appel `write_run_dir(result, root, config.verbose)` qui suivait directement `result.preprocessing = ...` pour éviter un double appel.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/experiments/test_runner_smoke.py -v`
Expected: PASS (ou SKIPPED selon la présence des données — dans les deux cas, pas d'échec).

- [ ] **Step 5: Commit**

```bash
git add src/experiments/runner.py tests/experiments/test_runner_smoke.py
git commit -m "feat: construit et persiste la référence de drift à l'entraînement"
```

---

## Task 11: Déplacement + extension de `get_model_from_mlflow` vers `tracking.py`

**Files:**
- Modify: `src/experiments/tracking.py`
- Modify: `src/experiments/server.py`
- Modify: `tests/experiments/test_tracking.py`

**Interfaces:**
- Produces: `tracking.get_model_from_mlflow(model_name: str, client: mlflow.MlflowClient, alias: str = "champion") -> tuple[Callable, nn.Module, TaskSpec, str]` (le dernier élément est `run_id`). Lève `ValueError` si le modèle/alias n'existe pas.
- `server.get_model_from_mlflow` garde sa signature 3-tuple existante (délègue à `tracking.get_model_from_mlflow`, convertit `ValueError` en `bentoml.exceptions.InvalidArgument`).

- [ ] **Step 1: Write the failing test**

Ajouter à `tests/experiments/test_tracking.py` (imports à ajouter en tête du fichier : `import torch`, `import yaml`) :

```python
class _FakeModelVersion:
    def __init__(self, run_id="run123", source="models:/foo/1"):
        self.run_id = run_id
        self.source = source


class _FakeRegistryClient:
    def __init__(self, model_version=None, raise_exc=None):
        self._mv = model_version
        self._raise_exc = raise_exc

    def get_model_version_by_alias(self, name, alias):
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._mv


def test_get_model_from_mlflow_returns_model_task_transform_and_run_id(monkeypatch):
    import mlflow as real_mlflow

    monkeypatch.setattr(real_mlflow.pytorch, "load_model", lambda source: "FAKE_MODEL")
    monkeypatch.setattr(
        real_mlflow.artifacts,
        "load_text",
        lambda uri: yaml.safe_dump(ExperimentConfig(task="type").model_dump(mode="json")),
    )
    monkeypatch.setattr(
        real_mlflow.artifacts,
        "load_dict",
        lambda uri: {"dataset_mean": [0.5, 0.5, 0.5], "dataset_std": [0.25, 0.25, 0.25], "reinhard": None},
    )
    client = _FakeRegistryClient(model_version=_FakeModelVersion(run_id="run123"))

    transform, model, task, run_id = tracking.get_model_from_mlflow("blood_cells-type", client)

    assert run_id == "run123"
    assert model == "FAKE_MODEL"
    assert task.name == "type"
    out = transform(torch.randint(0, 255, (3, 30, 30), dtype=torch.uint8))
    assert out.shape == (3, 224, 224)


def test_get_model_from_mlflow_raises_value_error_when_alias_missing(monkeypatch):
    import mlflow as real_mlflow
    from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

    exc = real_mlflow.MlflowException("introuvable", error_code=RESOURCE_DOES_NOT_EXIST)
    client = _FakeRegistryClient(raise_exc=exc)

    with pytest.raises(ValueError, match="champion"):
        tracking.get_model_from_mlflow("blood_cells-type", client, alias="champion")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments/test_tracking.py -v -k get_model_from_mlflow`
Expected: FAIL (`AttributeError: module 'src.experiments.tracking' has no attribute 'get_model_from_mlflow'`).

- [ ] **Step 3: Write minimal implementation**

Dans `src/experiments/tracking.py` :

1. Ajouter en tête (après les imports existants) :

```python
import torch
import yaml

from src.config import ExperimentConfig
from src.experiments.inference import _reinhard_from_stats
from src.features.transform import build_transform
from src.tasks import TASKS
```

2. Ajouter la fonction (juste après `register_model`, avant `log_run`) :

```python
def get_model_from_mlflow(model_name: str, client: mlflow.MlflowClient, alias: str = "champion"):
    """Résout l'alias (ex. ``champion``) en ``(transform, model, task, run_id)``.

    Raises:
        ValueError: si le modèle ou l'alias n'existe pas dans le registre.
    """
    try:
        mlflow_model = client.get_model_version_by_alias(model_name, alias)
    except mlflow.MlflowException as e:
        if e.error_code in (
            ErrorCode.Name(RESOURCE_DOES_NOT_EXIST),
            ErrorCode.Name(INVALID_PARAMETER_VALUE),
        ):
            raise ValueError(f"modèle {model_name!r} ou alias {alias!r} introuvable") from e
        raise
    model = mlflow.pytorch.load_model(mlflow_model.source)
    run_uri = f"runs:/{mlflow_model.run_id}"
    config_data = mlflow.artifacts.load_text(f"{run_uri}/config.yaml")
    config = ExperimentConfig.model_validate(yaml.safe_load(config_data))
    preprocessing = mlflow.artifacts.load_dict(f"{run_uri}/preprocessing.json")
    normalizer = (
        _reinhard_from_stats(preprocessing["reinhard"]) if preprocessing.get("reinhard") else None
    )
    transform = build_transform(
        config.transform,
        training=False,
        reinhard_normalizer=normalizer,
        dataset_mean=torch.tensor(preprocessing["dataset_mean"]),
        dataset_std=torch.tensor(preprocessing["dataset_std"]),
    )
    return transform, model, TASKS.get(config.task), mlflow_model.run_id
```

3. Dans `src/experiments/server.py`, remplacer l'ancienne fonction (lignes 65-91) par :

```python
def get_model_from_mlflow(model_name: str, client: mlflow.MlflowClient, alias="champion"):
    try:
        transform, model, task, _run_id = tracking.get_model_from_mlflow(model_name, client, alias)
    except ValueError as exc:
        raise bentoml.exceptions.InvalidArgument(str(exc)) from exc
    return transform, model, task
```

4. Toujours dans `server.py`, ajuster les imports en tête : supprimer `from mlflow.protos.databricks_pb2 import (...)`, `from src.config import ExperimentConfig`, `from src.experiments.inference import _reinhard_from_stats`, `from src.tasks import TASKS`, `import yaml` (tous devenus inutiles ici — vérifier qu'ils ne sont pas utilisés ailleurs dans le fichier avant suppression), retirer `build_transform` de `from src.features.transform import build_transform, load_image_rgb` (garder `load_image_rgb`), et ajouter `from src.experiments import tracking`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/experiments/test_tracking.py -v`
Expected: PASS (tous les tests du fichier, y compris les existants).

Vérifier aussi que `server.py` importe toujours correctement :

Run: `uv run python -c "import src.experiments.server"`
Expected: pas d'erreur d'import (`ImportError`/`NameError` sur un import supprimé à tort).

- [ ] **Step 5: Commit**

```bash
git add src/experiments/tracking.py src/experiments/server.py tests/experiments/test_tracking.py
git commit -m "refactor: déplace get_model_from_mlflow vers tracking.py, ajoute run_id"
```

---

## Task 12: Dépendance Evidently + chargement de la référence (`drift.py`, sans Evidently)

**Files:**
- Create: `src/experiments/drift.py`
- Create: `tests/experiments/test_drift.py`

**Interfaces:**
- Produces: `_load_required_reference(run_id, filename) -> pd.DataFrame` (lève `FileNotFoundError` explicite) ; `_load_optional_reference(run_id, filename) -> pd.DataFrame | None` (retourne `None` silencieusement).

- [ ] **Step 1: Add the dependency**

Run: `uv add evidently`

Vérifier la version résolue :

Run: `uv run python -c "import evidently; print(evidently.__version__)"`

- [ ] **Step 2: Write the failing test**

Créer `tests/experiments/test_drift.py` :

```python
"""Rapport de data drift : chargement de la référence, construction du rapport."""

import pandas as pd
import pytest

from src.experiments import drift


def test_load_required_reference_reads_parquet(tmp_path, monkeypatch):
    local = tmp_path / "descriptors.parquet"
    pd.DataFrame({"path": ["a.png"], "brightness": [0.5]}).to_parquet(local)
    monkeypatch.setattr(drift.mlflow.artifacts, "download_artifacts", lambda artifact_uri: str(local))
    df = drift._load_required_reference("run123", "descriptors")
    assert df["brightness"].iloc[0] == 0.5


def test_load_required_reference_raises_clear_error_when_missing(monkeypatch):
    def _boom(artifact_uri):
        raise OSError("not found")

    monkeypatch.setattr(drift.mlflow.artifacts, "download_artifacts", _boom)
    with pytest.raises(FileNotFoundError, match="run123"):
        drift._load_required_reference("run123", "descriptors")


def test_load_optional_reference_returns_none_when_missing(monkeypatch):
    def _boom(artifact_uri):
        raise OSError("not found")

    monkeypatch.setattr(drift.mlflow.artifacts, "download_artifacts", _boom)
    assert drift._load_optional_reference("run123", "embeddings") is None


def test_load_optional_reference_reads_parquet_when_present(tmp_path, monkeypatch):
    local = tmp_path / "embeddings.parquet"
    pd.DataFrame({"path": ["a.png"], "emb_0": [1.0]}).to_parquet(local)
    monkeypatch.setattr(drift.mlflow.artifacts, "download_artifacts", lambda artifact_uri: str(local))
    df = drift._load_optional_reference("run123", "embeddings")
    assert df["emb_0"].iloc[0] == 1.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/experiments/test_drift.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.experiments.drift'`).

- [ ] **Step 4: Write minimal implementation**

Créer `src/experiments/drift.py` (début du fichier — complété en Tâche 13) :

```python
"""Rapport de data drift (Evidently) entre la référence d'un run champion et de nouvelles images.

Ne relit jamais les images brutes d'entraînement : seule la référence
compacte (``drift_reference/descriptors.parquet`` + ``embeddings.parquet``
optionnel) loggée comme artefact MLflow du run champion est téléchargée.
"""

from __future__ import annotations

import mlflow
import pandas as pd


def _load_required_reference(run_id: str, filename: str) -> pd.DataFrame:
    """Télécharge et lit ``drift_reference/<filename>.parquet`` du run ``run_id``.

    Raises:
        FileNotFoundError: si l'artefact est absent (run antérieur à cette
            fonctionnalité, ou ``output.save_checkpoint=False``) — on ne peut
            rien produire sans cette référence.
    """
    try:
        local_path = mlflow.artifacts.download_artifacts(
            f"runs:/{run_id}/drift_reference/{filename}.parquet"
        )
    except Exception as exc:  # noqa: BLE001 -- reformulé en erreur explicite ci-dessous
        raise FileNotFoundError(
            f"drift_reference/{filename}.parquet absent du run {run_id} — "
            "run antérieur à cette fonctionnalité ? Impossible de produire un rapport de drift."
        ) from exc
    return pd.read_parquet(local_path)


def _load_optional_reference(run_id: str, filename: str) -> pd.DataFrame | None:
    """Comme ``_load_required_reference``, mais renvoie ``None`` si l'artefact est absent."""
    try:
        local_path = mlflow.artifacts.download_artifacts(
            f"runs:/{run_id}/drift_reference/{filename}.parquet"
        )
    except Exception:  # noqa: BLE001 -- skip gracieux, voir spec §5
        return None
    return pd.read_parquet(local_path)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/experiments/test_drift.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/experiments/drift.py tests/experiments/test_drift.py
git commit -m "feat: ajoute evidently et le chargement de la référence de drift"
```

---

## Task 13: `run_drift_report` — construction du rapport Evidently

**Files:**
- Modify: `src/experiments/drift.py`
- Modify: `tests/experiments/test_drift.py`

**Interfaces:**
- Consumes: `_load_required_reference`/`_load_optional_reference` (Tâche 12), `build_drift_reference` (Tâche 8), `tracking.get_model_from_mlflow` (Tâche 11).
- Produces: `DriftReportResult` (dataclass : `descriptors_drifted: bool`, `embeddings_drifted: bool | None`, `report_path: Path`) ; `run_drift_report(model_name: str, image_paths: Sequence[str | Path], alias: str = "champion", output: Path = Path("drift_report")) -> DriftReportResult`.

**Écart volontaire avec la spec (§4)** : le paramètre `image_files: Sequence[bytes] | None` listé dans la spec (par symétrie avec `inference.py`) est abandonné ici — le point d'entrée retenu est un CLI/batch qui lit toujours depuis des chemins sur disque (`--images <dossier>`), jamais depuis des octets uploadés ; YAGNI tant qu'aucun appelant n'en a besoin. Si un futur endpoint bento pour ce rapport est ajouté, réintroduire le paramètre à ce moment-là.

**Note d'implémentation (vérifiée empiriquement contre `evidently==0.4.36`, la version réellement résolue par `uv add evidently` en Tâche 12)** : contrairement à la première version de ce plan, **une seule métrique suffit** : `Report(metrics=[DataDriftPreset()])`. Vérifié par un essai direct dans cet environnement :
- `DataDriftPreset()` seul produit déjà `DatasetDriftMetric` + `DataDriftTable` dans `report.as_dict()["metrics"]`, avec la clé `dataset_drift` (bool) dans le `result` de `DatasetDriftMetric` — pas besoin d'ajouter `DatasetDriftMetric()` séparément (ça produirait un doublon).
- Dès que `column_mapping.embeddings = {"model_embedding": [...]}` est renseigné, `DataDriftPreset()` ajoute **automatiquement** un `EmbeddingsDriftMetric` à la sortie, avec `method_name: "model"` **par défaut** — pas besoin de construire `EmbeddingsDriftMetric(...)` à la main (son constructeur dans cette version prend d'ailleurs `drift_method`, pas `method`, ce qui aurait planté). Son `result` contient `drift_detected` (bool).
- Sans groupe `embeddings` dans le `column_mapping`, aucun `EmbeddingsDriftMetric` n'apparaît — le skip gracieux est donc gratuit, sans branchement sur la liste de métriques.

Ne pas importer `DatasetDriftMetric`, `EmbeddingsDriftMetric`, ni `evidently.metrics.data_drift.embedding_drift_methods` : seuls `ColumnMapping`, `DataDriftPreset` et `Report` sont nécessaires.

- [ ] **Step 1: Write the failing test**

Ajouter à `tests/experiments/test_drift.py` :

```python
import numpy as np
from PIL import Image

from src.experiments.drift import DriftReportResult, run_drift_report
from src.experiments.drift_reference import DriftReference


def _write_images(tmp_path, n, color):
    paths = []
    for i in range(n):
        p = tmp_path / f"img_{i}.png"
        arr = np.zeros((30, 30, 3), dtype=np.uint8)
        arr[:, :] = color
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    return paths


def _patch_champion(monkeypatch, tmp_path, *, with_embeddings):
    import src.experiments.drift as drift_module

    def _fake_get_model(model_name, client, alias="champion"):
        return (lambda img: img.float() / 255.0), object(), None, "run123"

    monkeypatch.setattr(drift_module, "get_model_from_mlflow", _fake_get_model)

    ref_paths = _write_images(tmp_path / "ref", 20, color=(200, 200, 200))  # clair
    descriptors_ref = compute_descriptors_for_test(ref_paths)

    def _fake_required(run_id, filename):
        assert filename == "descriptors"
        return descriptors_ref

    def _fake_optional(run_id, filename):
        if not with_embeddings:
            return None
        emb = descriptors_ref[["path"]].copy()
        emb["emb_0"] = 0.1
        return emb

    monkeypatch.setattr(drift_module, "_load_required_reference", _fake_required)
    monkeypatch.setattr(drift_module, "_load_optional_reference", _fake_optional)


def compute_descriptors_for_test(paths):
    from src.features.drift_descriptors import compute_descriptors

    return compute_descriptors(paths)


def test_run_drift_report_detects_drift_on_descriptors(tmp_path, monkeypatch):
    _patch_champion(monkeypatch, tmp_path, with_embeddings=False)
    current_paths = _write_images(tmp_path / "current", 20, color=(5, 5, 5))  # sombre : drift net
    result = run_drift_report(
        "blood_cells-type", current_paths, output=tmp_path / "out"
    )
    assert isinstance(result, DriftReportResult)
    assert result.descriptors_drifted is True
    assert result.embeddings_drifted is None
    assert result.report_path.exists()


def test_run_drift_report_no_drift_when_distributions_match(tmp_path, monkeypatch):
    _patch_champion(monkeypatch, tmp_path, with_embeddings=False)
    current_paths = _write_images(tmp_path / "current", 20, color=(200, 200, 200))  # même couleur
    result = run_drift_report(
        "blood_cells-type", current_paths, output=tmp_path / "out"
    )
    assert result.descriptors_drifted is False


def test_run_drift_report_skips_embeddings_section_when_reference_absent(tmp_path, monkeypatch):
    _patch_champion(monkeypatch, tmp_path, with_embeddings=False)
    current_paths = _write_images(tmp_path / "current", 20, color=(200, 200, 200))
    result = run_drift_report("blood_cells-type", current_paths, output=tmp_path / "out")
    assert result.embeddings_drifted is None


def test_run_drift_report_raises_on_empty_image_list(tmp_path, monkeypatch):
    _patch_champion(monkeypatch, tmp_path, with_embeddings=False)
    with pytest.raises(ValueError, match="aucune image"):
        run_drift_report("blood_cells-type", [], output=tmp_path / "out")


def test_run_drift_report_propagates_missing_reference_error(tmp_path, monkeypatch):
    """La référence manquante doit remonter jusqu'à l'appelant de run_drift_report, pas être avalée."""
    import src.experiments.drift as drift_module

    def _fake_get_model(model_name, client, alias="champion"):
        return (lambda img: img.float() / 255.0), object(), None, "run123"

    monkeypatch.setattr(drift_module, "get_model_from_mlflow", _fake_get_model)

    def _boom(artifact_uri):
        raise OSError("not found")

    # _load_required_reference n'est PAS mocké ici : on vérifie la composition réelle
    # get_model_from_mlflow -> _load_required_reference -> mlflow.artifacts.download_artifacts.
    monkeypatch.setattr(drift_module.mlflow.artifacts, "download_artifacts", _boom)

    current_paths = _write_images(tmp_path / "current", 5, color=(10, 10, 10))
    with pytest.raises(FileNotFoundError, match="run123"):
        run_drift_report("blood_cells-type", current_paths, output=tmp_path / "out")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments/test_drift.py -v -k run_drift_report`
Expected: FAIL (`ImportError: cannot import name 'run_drift_report'`).

- [ ] **Step 3: Write minimal implementation**

Ajouter à `src/experiments/drift.py` :

```python
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.experiments.drift_reference import build_drift_reference
from src.experiments.tracking import get_model_from_mlflow

if TYPE_CHECKING:
    from collections.abc import Sequence

EMBEDDING_GROUP = "model_embedding"


@dataclass
class DriftReportResult:
    """Résultat d'un rapport de drift : flags de synthèse + chemin du rapport HTML."""

    descriptors_drifted: bool
    embeddings_drifted: bool | None
    report_path: Path


def run_drift_report(
    model_name: str,
    image_paths: Sequence[str | Path],
    alias: str = "champion",
    output: Path = Path("drift_report"),
) -> DriftReportResult:
    """Compare ``image_paths`` à la référence du modèle champion, sans relire le train brut.

    Raises:
        ValueError: si ``image_paths`` est vide.
        FileNotFoundError: si le run champion n'a pas de référence de drift
            (run antérieur à cette fonctionnalité).
    """
    if not image_paths:
        raise ValueError("aucune image fournie — --images doit contenir au moins un fichier")

    transform, model, _task, run_id = get_model_from_mlflow(model_name, mlflow.MlflowClient(), alias)

    descriptors_ref = _load_required_reference(run_id, "descriptors")
    embeddings_ref = _load_optional_reference(run_id, "embeddings")

    sample_df = pd.DataFrame({"path": [str(p) for p in image_paths]})
    current_ref = build_drift_reference(sample_df, transform, model, device="cpu")
    descriptors_cur = current_ref.descriptors
    embeddings_cur = current_ref.embeddings if embeddings_ref is not None else None

    numerical_features = [c for c in descriptors_ref.columns if c != "path"]
    column_mapping = ColumnMapping(numerical_features=numerical_features)
    has_embeddings = embeddings_ref is not None and embeddings_cur is not None

    if has_embeddings:
        reference_df = descriptors_ref.merge(embeddings_ref, on="path")
        current_df = descriptors_cur.merge(embeddings_cur, on="path")
        emb_cols = [c for c in embeddings_ref.columns if c != "path"]
        column_mapping.embeddings = {EMBEDDING_GROUP: emb_cols}
    else:
        reference_df = descriptors_ref
        current_df = descriptors_cur

    # Un seul DataDriftPreset() suffit : il inclut déjà DatasetDriftMetric, et
    # ajoute automatiquement un EmbeddingsDriftMetric (méthode "model" par
    # défaut) dès que column_mapping.embeddings est renseigné — rien d'autre
    # à construire à la main (vérifié contre evidently==0.4.36).
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df.drop(columns=["path"]),
        current_data=current_df.drop(columns=["path"]),
        column_mapping=column_mapping,
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "drift_report.html"
    report.save_html(str(report_path))

    summary = report.as_dict()
    (output / "drift_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    dataset_drift_result = next(
        m["result"] for m in summary["metrics"] if m["metric"] == "DatasetDriftMetric"
    )
    descriptors_drifted = bool(dataset_drift_result["dataset_drift"])

    embeddings_drifted = None
    if has_embeddings:
        embeddings_result = next(
            m["result"] for m in summary["metrics"] if m["metric"] == "EmbeddingsDriftMetric"
        )
        embeddings_drifted = bool(embeddings_result["drift_detected"])

    return DriftReportResult(
        descriptors_drifted=descriptors_drifted,
        embeddings_drifted=embeddings_drifted,
        report_path=report_path,
    )
```

Ajouter `import pandas as pd` en tête du fichier s'il n'y est pas déjà (ajouté en Tâche 12).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/experiments/test_drift.py -v`
Expected: PASS — les clés `dataset_drift`/`drift_detected` et le comportement de `DataDriftPreset()` ont été vérifiés empiriquement contre la version installée (`evidently==0.4.36`), ce n'est plus une hypothèse à confirmer.

- [ ] **Step 5: Commit**

```bash
git add src/experiments/drift.py tests/experiments/test_drift.py
git commit -m "feat: ajoute run_drift_report (Evidently, un seul rapport combiné)"
```

---

## Task 14: `drift_cli.py`

**Files:**
- Create: `src/experiments/drift_cli.py`

**Interfaces:**
- Consumes: `run_drift_report` (Tâche 13).

- [ ] **Step 1: Write the implementation**

Créer `src/experiments/drift_cli.py` (pas de test dédié — convention du projet : les wrappers `argparse` de `src/experiments/*_cli.py` n'ont pas de fichier de test propre, cf. `promote_cli.py`/`grid_cli.py`/`retrain_cli.py`) :

```python
"""CLI : rapport de data drift entre le modèle champion et de nouvelles images."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.drift import run_drift_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare de nouvelles images à la référence de drift du modèle champion."
    )
    parser.add_argument("--model", required=True, help="nom du modèle enregistré dans MLflow")
    parser.add_argument("--images", required=True, type=Path, help="dossier d'images à comparer")
    parser.add_argument(
        "--alias", default="champion", help="alias MLflow à résoudre. Défaut : champion"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("drift_report"), help="dossier de sortie du rapport"
    )
    args = parser.parse_args()

    image_paths = sorted(p for p in args.images.iterdir() if p.is_file())
    result = run_drift_report(args.model, image_paths, alias=args.alias, output=args.output)

    print(f"Rapport écrit dans {result.report_path}")
    print(f"Drift descripteurs : {result.descriptors_drifted}")
    if result.embeddings_drifted is not None:
        print(f"Drift embeddings : {result.embeddings_drifted}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Vérification manuelle (pas de test automatisé — cf. convention)**

Run: `uv run python -m src.experiments.drift_cli --help`
Expected: affiche l'aide sans erreur d'import.

- [ ] **Step 3: Commit**

```bash
git add src/experiments/drift_cli.py
git commit -m "feat: ajoute le CLI de rapport de drift (drift_cli.py)"
```

---

## Fin de plan

Une fois les 14 tâches terminées : lancer la suite complète (`uv run pytest -q`), puis utiliser `superpowers:finishing-a-development-branch` pour décider de l'intégration (merge/PR) de la branche `romain`.
