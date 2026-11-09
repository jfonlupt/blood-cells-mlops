"""Inférence et Grad-CAM sur une image, pour la page de démonstration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
import yaml

import streamlit as st
from src.config import ExperimentConfig, load_config
from src.constants import CRITICALITY, DATASET_PATHS
from src.experiments.inference import rebuild_eval_transform
from src.features.dataframe import create_dataframe
from src.features.transform import load_image_rgb, resize_with_pad
from src.models import build_model
from src.models.gradcam import GradCAM, default_gradcam_layer
from src.tasks import get_task
from src.training.components import split_off_test
from utils.data import PROJECT_ROOT

if TYPE_CHECKING:
    from typing import BinaryIO

# Modèles proposés dans la démo : libellé affiché -> dossier du run relatif à la racine du projet.
DEMO_RUNS = {
    "MultiArchi — type (F1 0,894)": "results/final_comparison/multi_archi_type/runs/f8c1f669",
    "MultiArchi — criticité (F1 0,920)": "results/final_comparison/multi_archi_criticite/runs/79bb4c83",
    "DenseNet121 — type (abed1581)": "results/runs/abed1581",
    "ViT-Base/16 - type (59390a83)": "results/runs/59390a83",
    "EfficientNet-B0 - type (7cdafc88)": "results/runs/7cdafc88",
}
REQUIRED_RUN_FILES = ("config.yaml", "meta.json", "metrics_test.json", "checkpoint.pt")

# Images connues pour illustrer un biais de fond : la prédiction est juste (voire très confiante),
# mais la carte Grad-CAM montre que le modèle s'appuie sur les hématies environnantes plutôt que
# sur la cellule elle-même — cf. diagnostic Grad-CAM du modèle hiérarchique (page Résultats) sur
# le même phénomène. Chemins relatifs à la racine du projet. Clé : chemin de run (``DEMO_RUNS``).
DOMAIN_BIAS_EXAMPLES: dict[str, list[dict[str, str]]] = {
    "results/runs/59390a83": [
        {
            "path": "data/PBC_dataset_normal_DIB/ig/MY_758099.jpg",
            "note": (
                "IG prédit avec 100 % de confiance, mais le Grad-CAM s'allume sur les hématies "
                "environnantes plutôt que sur la cellule annotée : un signe du biais de site "
                "(artefacts de compression JPEG propres à PBC) plutôt qu'un raisonnement "
                "morphologique — cf. page Résultats, onglet Modèle hiérarchique."
            ),
        },
    ],
}


def domain_bias_examples(run_id: str) -> list[dict[str, str]]:
    """Exemples curés de biais de fond pour ce run : chemin résolu + note, fichiers absents filtrés."""
    examples = []
    for example in DOMAIN_BIAS_EXAMPLES.get(run_id, []):
        full_path = PROJECT_ROOT / example["path"]
        if full_path.is_file():
            examples.append({"path": str(full_path), "note": example["note"]})
    return examples


def _run_dir(run_path: str) -> Path:
    """Résout un chemin relatif de run ou un ancien identifiant depuis la racine du projet."""
    path = Path(run_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "results":
        return PROJECT_ROOT / path
    return PROJECT_ROOT / "results" / "runs" / path


def _load_demo_config(config_path: Path) -> ExperimentConfig:
    """Charge les configurations actuelles et migre le nom de loss historique retiré."""
    raw = yaml.safe_load(config_path.read_text()) or {}
    loss_config = raw.get("loss")
    if isinstance(loss_config, dict) and loss_config.get("name") == "cross_entropy":
        loss_config.pop("name")
    return ExperimentConfig.model_validate(raw)


def available_runs() -> dict[str, str]:
    """Sous-ensemble de ``DEMO_RUNS`` dont le dossier contient tous les fichiers nécessaires."""
    return {
        label: run_path
        for label, run_path in DEMO_RUNS.items()
        if all((_run_dir(run_path) / name).is_file() for name in REQUIRED_RUN_FILES)
    }


@dataclass
class DemoModel:
    """Modèle chargé, transform d'évaluation et Grad-CAM prêts à l'emploi."""

    name: str
    backbone: str
    task: str
    test_metrics: dict
    class_names: list[str]
    model: torch.nn.Module
    transform: object
    gradcam: GradCAM


@dataclass
class Prediction:
    """Résultat d'inférence sur une image."""

    probs: dict[str, float]
    pred_class: str
    criticality: str
    cam: np.ndarray
    image: np.ndarray


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_demo_model(run_id: str) -> DemoModel:
    """Charge le run ``run_id`` une seule fois pour toute la session serveur."""
    run_dir = _run_dir(run_id)
    config = _load_demo_config(run_dir / "config.yaml")
    if config.model.transfer is not None:
        # Les poids viennent du checkpoint : inutile de retélécharger ceux du pré-entraînement.
        config.model.transfer.pretrained = False
    task = get_task(config.task)
    model = build_model(config.model, task, img_size=config.transform.img_size)
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location="cpu"))
    model.eval()

    name = json.loads((run_dir / "meta.json").read_text()).get("name", run_id)
    test_metrics = json.loads((run_dir / "metrics_test.json").read_text())
    backbone = config.model.transfer.backbone if config.model.transfer else config.model.name
    return DemoModel(
        name=name,
        backbone=backbone,
        task=config.task,
        test_metrics=test_metrics,
        class_names=list(task.class_names),
        model=model,
        transform=rebuild_eval_transform(config, run_dir),
        gradcam=GradCAM(model, default_gradcam_layer(model)),
    )


@st.cache_data(show_spinner="Reconstitution du jeu de test…")
def load_test_df(run_id: str) -> pd.DataFrame:
    """Rejoue le split de test du run (mêmes données, même graine) : ``path``, ``type``, ``criticality``."""
    run_dir = _run_dir(run_id)
    config = load_config(run_dir / "config.yaml")
    if not all(DATASET_PATHS[key].is_dir() for key in config.data.datasets):
        return pd.DataFrame(columns=["path", "type", "criticality"])
    df = create_dataframe(config.data, get_task(config.task))
    _, test_df = split_off_test(df, config.data, seed=config.seed)
    return test_df[["path", "type", "criticality"]].reset_index(drop=True)


def predict(demo: DemoModel, source: str | BinaryIO) -> Prediction:
    """Classe une image (chemin ou fichier uploadé) et calcule son Grad-CAM."""
    x = demo.transform(load_image_rgb(source)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(demo.model(x)[0], dim=0)
    index = int(probs.argmax())
    cam = demo.gradcam.compute(x, target_index=index)

    h, w = cam.shape
    image = resize_with_pad(load_image_rgb(source), (h, w), fill=255)
    pred_class = demo.class_names[index]
    return Prediction(
        probs={name: float(probs[i]) for i, name in enumerate(demo.class_names)},
        pred_class=pred_class,
        criticality=CRITICALITY.get(pred_class, pred_class),
        cam=cam,
        image=image.permute(1, 2, 0).numpy() / 255.0,
    )


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Superpose la carte Grad-CAM (colormap jet) à l'image, en float [0, 1]."""
    heatmap = mpl.colormaps["jet"](cam)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * heatmap, 0.0, 1.0)
