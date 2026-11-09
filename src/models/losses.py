"""Perte unique du pipeline de classification de cellules sanguines.

``ClinicalLoss`` combine, en un seul module, les trois leviers utilisés dans le
projet — pondération du déséquilibre, focalisation, et coût clinique asymétrique.
Pour un exemple de vraie classe ``y`` et ``p`` = softmax des logits ::

    L_i = - w_y * (1 - p_y)**gamma * log(p_y)  +  lambda_cost * sum_c M[y, c] * p_c

- ``w_c`` : nombre effectif d'échantillons (Cui et al. 2019), réglé par ``beta``
  (0 = non pondéré, 1 = inverse-fréquence), multiplié par ``extra_weight`` (boost
  clinique manuel), renormalisé à moyenne 1 ;
- ``(1 - p_y)**gamma`` : terme focal (Lin et al. 2017), ``gamma=0`` = désactivé ;
- ``sum_c M[y, c] * p_c`` : coût espéré sous la distribution prédite, différentiable,
  pondéré par ``lambda_cost`` (0 = désactivé). ``M[i, j]`` = coût de prédire ``j``
  quand la vraie classe est ``i`` (diagonale nulle).

Cas particuliers : ``gamma=0, lambda_cost=0, beta=0`` -> cross-entropy nue ;
``beta>0`` -> CE class-balanced ; ``gamma>0`` -> focal ; ``lambda_cost>0`` -> perte
sensible au coût. Un lissage de labels optionnel (``label_smoothing``) est
disponible.

``BCEWithLogitsLoss`` reste utilisé pour les tâches à sortie 1 logit (``dataset``).

La perte accepte des logits bruts et des labels entiers (indices de classe),
comme ``nn.CrossEntropyLoss``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.constants import (
    CRITICALITY,
    CRITICALITY_CLASSES,
    CRITICALITY_TO_LABEL,
    FINAL_CLASSES,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.config import LossConfig
    from src.tasks import TaskSpec

# ---------------------------------------------------------------------------
# Effectifs par classe (cf. constants.py / document de stratégie)
# ---------------------------------------------------------------------------

SAMPLES_PER_CLASS = {
    "Basophil": 1297,
    "Blast": 3294,
    "Eosinophil": 3541,
    "Erythroblast": 1629,
    "IG": 2360,
    "Lymphocyte": 5162,
    "Monocyte": 3209,
    "Neutrophil": 11922,
    "Platelet": 2348,
    "Promyelocyte": 680,
}

SAMPLES_PER_CRITICALITY = {
    tier: sum(SAMPLES_PER_CLASS[c] for c, t in CRITICALITY.items() if t == tier)
    for tier in CRITICALITY_CLASSES
}  # -> {"normal": 27479, "intermediaire": 3989, "critique": 3974}

# Matrices de coût par défaut, dans l'espace de la criticité.
# cost[vraie][prédite], diagonale nulle.
DEFAULT_COST_MATRIX_3 = torch.tensor(
    [
        # prédit : normal  intermediaire  critique
        [0.0, 0.5, 1.0],  # vrai = normal        (faux positif : coût faible)
        [4.0, 0.0, 1.0],  # vrai = intermediaire (rater = coût moyen)
        [10.0, 3.0, 0.0],  # vrai = critique     (rater totalement = coût maximal)
    ],
    dtype=torch.float32,
)
DEFAULT_COST_MATRIX_2 = torch.tensor(
    [
        [0.0, 1.0],  # vrai = normal   (faux positif : coût faible)
        [10.0, 0.0],  # vrai = abnormal (faux négatif : coût maximal)
    ],
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------
# Poids de classe "class-balanced" (effective number of samples, Cui et al. 2019)
# ---------------------------------------------------------------------------
def effective_number_weights(
    samples_per_class: list[int] | np.ndarray, beta: float = 0.999
) -> np.ndarray:
    """Poids par classe, normalisés pour que leur moyenne vaille 1.

    ``beta`` règle l'intensité de la correction :

    - ``beta = 0`` -> aucun poids (tous à 1) ;
    - ``0 < beta < 1`` -> nombre effectif ``(1-beta) / (1-beta**n)`` ;
    - ``beta = 1`` -> inverse-fréquence pure ``1/n``.
    """
    counts = np.asarray(samples_per_class, dtype=np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    if beta <= 0:
        weights = np.ones_like(counts)
    elif beta >= 1:
        weights = 1.0 / counts
    else:
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
    return weights / weights.sum() * len(weights)


# ---------------------------------------------------------------------------
# Perte unique
# ---------------------------------------------------------------------------
class ClinicalLoss(nn.Module):
    """CE pondérée class-balanced + terme focal + coût espéré (cf. module docstring).

    Paramètres
    ----------
    samples_per_class : list[int] | dict[str, int]
        Effectifs par classe, dans l'ordre des indices du modèle si liste, dans
        l'ordre de ``class_names`` si dict.
    class_names : list[str] | None
        Ordre des classes (requis si ``samples_per_class`` est un dict ou si
        ``extra_weight`` nomme des classes).
    beta : float
        Paramètre du nombre effectif d'échantillons (0 = non pondéré, 1 = inverse-fréquence).
    gamma : float
        Exposant focal (0 = désactivé).
    extra_weight : dict[str, float] | None
        Multiplicateur manuel par classe (ex. ``{"Blast": 1.5}``).
    label_smoothing : float
        Lissage de labels (0 = désactivé).
    cost_matrix : torch.Tensor | list[list[float]] | None
        Matrice ``(K, K)`` déjà à la dimension du modèle. ``None`` = pas de terme de coût.
    lambda_cost : float
        Poids du terme de coût espéré (0 = désactivé).
    """

    def __init__(
        self,
        samples_per_class: list[int] | dict[str, int],
        *,
        class_names: list[str] | None = None,
        beta: float = 0.999,
        gamma: float = 0.0,
        extra_weight: dict[str, float] | None = None,
        label_smoothing: float = 0.0,
        cost_matrix: torch.Tensor | list[list[float]] | None = None,
        lambda_cost: float = 0.0,
        reduction: str = "mean",
    ):
        """Construit la perte (paramètres décrits dans la docstring de la classe)."""
        super().__init__()

        if isinstance(samples_per_class, dict):
            if class_names is None:
                raise ValueError("class_names est requis quand samples_per_class est un dict")
            counts = [samples_per_class[c] for c in class_names]
        else:
            counts = list(samples_per_class)
            class_names = class_names or list(range(len(counts)))

        weights = effective_number_weights(counts, beta=beta)
        if extra_weight:
            multipliers = np.array([extra_weight.get(c, 1.0) for c in class_names])
            weights = weights * multipliers
            weights = weights / weights.sum() * len(weights)
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.lambda_cost = float(lambda_cost)
        self.reduction = reduction

        if cost_matrix is None:
            self.cost_matrix = None
        else:
            cm = cost_matrix if torch.is_tensor(cost_matrix) else torch.tensor(cost_matrix)
            self.register_buffer("cost_matrix", cm.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: ``(batch, K)`` non normalisés. target: ``(batch,)`` indices entiers."""
        log_probs = F.log_softmax(logits, dim=1)

        if self.label_smoothing > 0:
            num_classes = logits.size(1)
            with torch.no_grad():
                smooth_target = torch.full_like(
                    log_probs, self.label_smoothing / (num_classes - 1)
                )
                smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
            target_log_probs = (smooth_target * log_probs).sum(dim=1)
            target_probs = (smooth_target * log_probs.exp()).sum(dim=1)
        else:
            target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
            target_probs = target_log_probs.exp()

        focal_term = (1.0 - target_probs).clamp(min=0.0).pow(self.gamma)  # gamma=0 -> 1
        class_w = self.class_weights.to(logits.device)[target]
        loss = -class_w * focal_term * target_log_probs

        if self.lambda_cost > 0 and self.cost_matrix is not None:
            probs = log_probs.exp()
            cost_rows = self.cost_matrix.to(logits.device)[target]  # (batch, K)
            loss = loss + self.lambda_cost * (cost_rows * probs).sum(dim=1)

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


# ---------------------------------------------------------------------------
# Fabrique de loss : instancie la perte selon la config et la tâche
# ---------------------------------------------------------------------------
def _counts_in_task_order(train_df: pd.DataFrame, task: TaskSpec) -> list[int]:
    """Effectifs par classe, dans l'ordre de ``task.class_names`` (0 si absente)."""
    seen = train_df[task.label_column].value_counts().to_dict()
    return [int(seen.get(i, 0)) for i in range(task.num_classes)]


def _criticality_indices(task: TaskSpec, k_base: int) -> list[int]:
    """Indice de palier de criticité de chaque classe de la tâche.

    ``k_base=3`` -> normal / intermediaire / critique ; ``k_base=2`` -> normal / abnormal.
    """
    if task.name == "type":
        tiers = [CRITICALITY[c] for c in task.class_names]
    elif task.name in ("criticality", "binary_criticality"):
        tiers = list(task.class_names)  # les classes sont déjà des paliers
    else:
        raise ValueError(f"cost_matrix non supporté pour task={task.name!r}")

    if k_base == 3:
        table = CRITICALITY_TO_LABEL  # {normal:0, intermediaire:1, critique:2}
    elif k_base == 2:
        table = {"normal": 0, "intermediaire": 1, "critique": 1, "abnormal": 1}
    else:
        raise ValueError("le lift ne gère que k_base in {2, 3}")

    try:
        return [table[t] for t in tiers]
    except KeyError as exc:
        raise ValueError(
            f"loss.cost_matrix {k_base}x{k_base} incompatible avec task={task.name!r} "
            f"(classe {exc} sans palier à cette granularité)."
        ) from exc


def _resolve_cost_matrix(
    raw: list[list[float]] | None, task: TaskSpec
) -> torch.Tensor | None:
    """Résout la matrice de coût finale ``(num_classes, num_classes)``.

    - ``raw is None`` -> matrice par défaut (3x3, ou 2x2 pour une tâche binaire) ;
    - matrice déjà à la dimension de la tâche -> utilisée telle quelle ;
    - matrice 2x2 ou 3x3 pour une tâche plus large (``type``) -> *liftée* via la
      criticité de chaque classe : ``M_out[i,j] = raw[crit(c_i), crit(c_j)]``.
    """
    if task.default_loss == "bce":
        return None

    if raw is None:
        base = DEFAULT_COST_MATRIX_2 if task.num_classes == 2 else DEFAULT_COST_MATRIX_3
    else:
        base = torch.tensor(raw, dtype=torch.float32)

    k = base.shape[0]
    if k == task.num_classes:
        return base
    if k not in (2, 3):
        raise ValueError(
            f"loss.cost_matrix {k}x{k} : attendu 2x2, 3x3, ou "
            f"{task.num_classes}x{task.num_classes} pour task={task.name!r}."
        )
    idx = torch.tensor(_criticality_indices(task, k), dtype=torch.long)
    return base[idx][:, idx]


def build_loss(loss_cfg: LossConfig, task: TaskSpec, train_df: pd.DataFrame) -> nn.Module:
    """Instancie la perte correspondant à la config et à la tâche.

    Args:
        loss_cfg: Hyperparamètres de la perte.
        task: Spécification de la tâche de classification.
        train_df: DataFrame d'entraînement, pour dériver les effectifs par classe.

    Returns:
        ``nn.BCEWithLogitsLoss`` pour une tâche à sortie 1 logit, sinon ``ClinicalLoss``.
    """
    if task.default_loss == "bce":
        return nn.BCEWithLogitsLoss()

    return ClinicalLoss(
        _counts_in_task_order(train_df, task),
        class_names=list(task.class_names),
        beta=loss_cfg.beta,
        gamma=loss_cfg.gamma,
        extra_weight=loss_cfg.extra_weight or None,
        label_smoothing=loss_cfg.label_smoothing,
        cost_matrix=_resolve_cost_matrix(loss_cfg.cost_matrix, task),
        lambda_cost=loss_cfg.lambda_cost,
    )


# ---------------------------------------------------------------------------
# Exemple d'instanciation + vérification rapide
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    batch = 8

    fine = ClinicalLoss(
        SAMPLES_PER_CLASS,
        class_names=FINAL_CLASSES,
        beta=0.999,
        gamma=2.0,
        extra_weight={"Blast": 1.5, "Promyelocyte": 1.5},
    )
    print("Poids par classe (10 classes) :")
    for name, w in zip(FINAL_CLASSES, fine.class_weights.tolist()):
        print(f"  {name:15s} {w:.3f}")
    print("Loss focal (exemple) :", fine(torch.randn(batch, 10), torch.randint(0, 10, (batch,))).item())

    # Perte 10 classes shapée par le coût clinique (matrice 3x3 liftée)
    from src.tasks import get_task

    task_type = get_task("type")
    cost = ClinicalLoss(
        list(SAMPLES_PER_CLASS.values()),
        class_names=FINAL_CLASSES,
        beta=0.999,
        lambda_cost=1.0,
        cost_matrix=_resolve_cost_matrix(None, task_type),
    )
    print(
        "Loss coût-clinique 10 classes (exemple) :",
        cost(torch.randn(batch, 10), torch.randint(0, 10, (batch,))).item(),
    )
