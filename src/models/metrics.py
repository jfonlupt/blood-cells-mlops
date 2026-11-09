"""Setup des métriques d'évaluation avec torchmetrics.

Pour la classification de cellules sanguines (Barcelone + Munich), en tenant
compte :
  - du déséquilibre de classes (long-tail)
  - de la criticité clinique différenciée par classe (cf. doc de stratégie
    "strategie-reequilibrage-dataset.md" du projet)

Installation : pip install torchmetrics
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    Accuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassAveragePrecision,  # AUPRC par classe (nécessite les probas, pas juste argmax)
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassFBetaScore,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassPrecisionRecallCurve,  # courbe P-R par classe (probas requises)
    MulticlassRecall,
)
from torchmetrics.functional.classification import (
    binary_average_precision,
    binary_precision_recall_curve,
)
from torchmetrics.utilities.data import dim_zero_cat

from src.constants import (
    ABNORMAL_LABELS,
    BINARY_CRITICALITY_CLASSES,
    CRITICAL_LABELS,
    CRITICALITY_CLASSES,
    FINAL_CLASSES,
    NORMAL_LABELS,
)

if TYPE_CHECKING:
    from src.tasks import TaskSpec

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FIGURES_PATH = BASE_DIR / "reports" / "figures"

# Seuil minimum de rappel exigé sur les classes critiques pour qu'un
# checkpoint soit éligible à la sélection de modèle (cf. section "Sélection
# de modèle" de la réponse précédente).
MIN_CRITICAL_RECALL = 0.95

# Nombre de seuils échantillonnés pour la courbe P-R (état borné, formes fixes).
PR_CURVE_THRESHOLDS = 100

# Métriques top-k : ajoutées seulement si la tâche a assez de classes pour qu'un
# top-k soit informatif (top-3 sur 3 classes = trivialement parfait).
TOP_K = 3
MIN_CLASSES_FOR_TOP_K = 5

# Métriques dont ``compute()`` renvoie le triplet ``(precision, recall, thresholds)``
# plutôt qu'un scalaire ou un vecteur. ``Trainer.evaluate`` s'appuie dessus pour
# aplatir ces sorties correctement (cf. ``_flatten_metric``).
CURVE_METRICS = frozenset({"pr_curve", "critical_screening_prc"})

# Cibles de rappel pour lesquelles on reporte un point de fonctionnement du
# screening critique (seuil, précision, charge de revue) — cf.
# :func:`screening_operating_points`.
SCREENING_RECALL_TARGETS = (0.95, 0.99)


# ---------------------------------------------------------------------------
# 1 bis. Métriques "criticité" dérivées de la tâche type
# ---------------------------------------------------------------------------
# En déploiement, seule une prédiction de *type critique* (Blast, Promyelocyte)
# envoie automatiquement l'image en revue médicale. Ce qui compte cliniquement
# n'est donc pas le rappel par type mais le rappel de la *classe critique
# amalgamée* : confondre Blast et Promyelocyte est sans conséquence, prédire
# Neutrophil pour l'un des deux est une cellule critique qui échappe à la revue.


class GroupedRecall(Metric):
    """Rappel d'un groupe de classes pris en bloc.

    Parmi les images dont le vrai label est dans ``group_labels``, fraction
    prédite (argmax) dans ce même groupe — quel que soit le type exact. Sur les
    types critiques : « la cellule critique sera-t-elle envoyée en revue ? ».

    ``update`` attend les logits ``(B, num_classes)`` et les labels entiers
    ``(B,)`` — même signature que le reste de la ``MetricCollection``.
    """

    higher_is_better = True
    full_state_update = False

    def __init__(self, group_labels, **kwargs):
        """``group_labels`` : indices de classe formant le groupe (ex. critiques)."""
        super().__init__(**kwargs)
        group = torch.as_tensor(sorted({int(x) for x in group_labels}))
        self.register_buffer("group_labels", group, persistent=False)
        self.add_state("hits", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("support", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Accumule les vrais du groupe et ceux prédits (argmax) dans le groupe."""
        group = self.group_labels.to(target.device)
        pred = logits.argmax(dim=-1)
        true_in_group = torch.isin(target, group)
        pred_in_group = torch.isin(pred, group)
        self.hits += (true_in_group & pred_in_group).sum()
        self.support += true_in_group.sum()

    def compute(self) -> torch.Tensor:
        """Rappel du groupe, ``nan`` si aucune image vraie du groupe."""
        if self.support == 0:
            return torch.tensor(float("nan"))
        return self.hits.float() / self.support.float()


class _CriticalMassMetric(Metric):
    """Base des métriques de screening sur la « masse de probabilité critique ».

    Accumule le score binaire dérivé (somme des softmax sur ``group_labels``) et
    le label binaire ``est critique``. La somme des probas — et non le max — est
    la probabilité prédite que la cellule soit critique, marginalisée sur le type
    critique exact.
    """

    full_state_update = False

    def __init__(self, group_labels, **kwargs):
        """``group_labels`` : indices de classe formant le groupe critique."""
        super().__init__(**kwargs)
        group = torch.as_tensor(sorted({int(x) for x in group_labels}))
        self.register_buffer("group_labels", group, persistent=False)
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Accumule ``sum(softmax[groupe])`` et ``target in groupe`` pour ce batch."""
        group = self.group_labels.to(logits.device)
        probs = logits.softmax(dim=-1)
        self.scores.append(probs[:, group].sum(dim=-1))
        self.target.append(torch.isin(target, group.to(target.device)).long())

    def _score_target(self) -> tuple[torch.Tensor, torch.Tensor]:
        return dim_zero_cat(self.scores), dim_zero_cat(self.target)


class CriticalScreeningAveragePrecision(_CriticalMassMetric):
    """AUC précision-rappel du score « masse critique » vs le label est-critique.

    Résume, indépendamment du seuil, la capacité à trier les cellules critiques
    en abaissant la barre de décision sous l'argmax.
    """

    higher_is_better = True

    def compute(self) -> torch.Tensor:
        """Average precision binaire du score dérivé."""
        scores, target = self._score_target()
        return binary_average_precision(scores, target)


class CriticalScreeningPRCurve(_CriticalMassMetric):
    """Courbe précision-rappel du score « masse critique ».

    Renvoie le triplet ``(precision, recall, thresholds)`` (comme
    ``MulticlassPrecisionRecallCurve``), pour lire le rappel et la charge de revue
    atteignables à chaque seuil.
    """

    def __init__(self, group_labels, thresholds: int = PR_CURVE_THRESHOLDS, **kwargs):
        """``thresholds`` : nombre de seuils échantillonnés (état borné)."""
        super().__init__(group_labels, **kwargs)
        self.thresholds = thresholds

    def compute(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Triplet ``(precision, recall, thresholds)`` du score dérivé."""
        scores, target = self._score_target()
        return binary_precision_recall_curve(scores, target, thresholds=self.thresholds)


def screening_operating_points(
    precision,
    recall,
    thresholds,
    *,
    n_critical: int,
    n_total: int,
    targets: tuple[float, ...] = SCREENING_RECALL_TARGETS,
) -> dict:
    """Points de fonctionnement du screening critique pour des cibles de rappel.

    Pour chaque cible ``t``, retient le seuil le plus élevé (décision la plus
    stricte) dont le rappel reste ``>= t``, et calcule la *charge de revue* =
    fraction des images totales envoyées en revue à ce seuil :

        review_load = (rappel * n_critical) / (precision * n_total)

    Args:
        precision, recall, thresholds: sorties de
            :meth:`CriticalScreeningPRCurve.compute` (``precision``/``recall``
            peuvent avoir un élément sentinelle de plus que ``thresholds``).
        n_critical: nombre d'images vraiment critiques dans le jeu évalué.
        n_total: nombre total d'images évaluées.
        targets: cibles de rappel.

    Returns:
        ``{target: {"threshold", "precision", "recall", "review_load"} | None}``
        (``None`` si la cible de rappel est inatteignable).
    """
    precision = [float(x) for x in precision]
    recall = [float(x) for x in recall]
    thresholds = [float(x) for x in thresholds]
    k = len(thresholds)

    out: dict = {}
    for t in targets:
        best = None
        for i in range(k):
            if recall[i] >= t:
                best = i
        if best is None:
            out[t] = None
            continue
        p, r, thr = precision[best], recall[best], thresholds[best]
        review_load = (r * n_critical) / (p * n_total) if p > 0 else float("nan")
        out[t] = {
            "threshold": thr,
            "precision": p,
            "recall": r,
            "review_load": review_load,
        }
    return out


# ---------------------------------------------------------------------------
# 2. MetricCollection principale (multi-classe)
# ---------------------------------------------------------------------------
def build_metrics(
    task: TaskSpec, for_test: bool = False, device: str | torch.device = "cpu"
) -> MetricCollection:
    """Fabrique la ``MetricCollection`` adaptée à la tâche.

    Args:
        task: description de la tâche (``kind`` et ``num_classes``).
        for_test: si ``True``, renvoie la collection riche (par classe, matrice
            de confusion) ; sinon des métriques légères à logguer par epoch.
        device: device sur lequel placer les métriques.

    Returns:
        La ``MetricCollection`` correspondante :
          - tâche binaire -> ``{"acc": Accuracy(task="binary")}`` ;
          - multi-classe, ``for_test=False`` -> ``acc`` + ``f1`` (légères) ;
          - multi-classe, ``for_test=True`` -> ``f1_macro``, ``mcc``,
            ``*_per_class``, ``auprc_per_class``, ``pr_curve``, ``confusion_matrix``,
            et si ``num_classes >= 5`` : ``top3_acc``, ``top3_f1``,
            ``top3_recall_per_class``.
    """
    if task.kind == "binary":
        return MetricCollection({"acc": Accuracy(task="binary")}).to(device)

    n = task.num_classes
    if not for_test:
        return MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=n),
                "f1": MulticlassF1Score(num_classes=n),
            }
        ).to(device)

    metrics = MetricCollection(
        {
            # Vue globale — jamais l'accuracy seule (biaisée par `neutrophil`)
            "acc": MulticlassAccuracy(num_classes=n, average="macro"),
            "f1_macro": MulticlassF1Score(num_classes=n, average="macro"),
            "mcc": MulticlassMatthewsCorrCoef(num_classes=n),
            # Rappel et précision PAR CLASSE (average=None -> vecteur de taille NUM_CLASSES)
            "recall_per_class": MulticlassRecall(num_classes=n, average=None),
            "precision_per_class": MulticlassPrecision(num_classes=n, average=None),
            # F-beta (beta=2 -> pondère le rappel 2x plus que la précision),
            # calculé par classe : on ne lira que les indices critiques/modérés.
            "fbeta2_per_class": MulticlassFBetaScore(beta=2.0, num_classes=n, average=None),
            # AUPRC par classe : nécessite les probabilités softmax (pas argmax)
            "auprc_per_class": MulticlassAveragePrecision(num_classes=n, average=None),
            # Courbe précision-rappel par classe (mêmes probas), pour la figure
            "pr_curve": MulticlassPrecisionRecallCurve(
                num_classes=n, thresholds=PR_CURVE_THRESHOLDS
            ),
            # Matrice de confusion complète, pour inspecter spécifiquement
            # la direction "classe critique -> classe mature normale"
            "confusion_matrix": MulticlassConfusionMatrix(num_classes=n),
        }
    )

    if n >= MIN_CLASSES_FOR_TOP_K:
        # "la vraie classe est-elle dans les TOP_K mieux notées ?" — utile pour un
        # outil d'aide à la décision. top3_acc en micro (taux de succès global),
        # top3_f1 en macro (cohérent avec f1_macro), rappel top-3 par classe.
        metrics.add_metrics(
            {
                f"top{TOP_K}_acc": MulticlassAccuracy(num_classes=n, top_k=TOP_K, average="micro"),
                f"top{TOP_K}_f1": MulticlassF1Score(num_classes=n, top_k=TOP_K, average="macro"),
                f"top{TOP_K}_recall_per_class": MulticlassRecall(
                    num_classes=n, top_k=TOP_K, average=None
                ),
            }
        )

    if task.name == "type":
        # Rappel et screening de la classe critique amalgamée (Blast + Promyelocyte).
        # Sur la tâche `criticality` (3 classes) le rappel de « critique » existe
        # déjà nativement dans `recall_per_class`, inutile de le dériver.
        metrics.add_metrics(
            {
                "grouped_critical_recall": GroupedRecall(CRITICAL_LABELS),
                "critical_screening_ap": CriticalScreeningAveragePrecision(CRITICAL_LABELS),
                "critical_screening_prc": CriticalScreeningPRCurve(CRITICAL_LABELS),
            }
        )

    return metrics.to(device)


# ---------------------------------------------------------------------------
# 3. Métrique binaire dérivée "normal vs anormal"
# ---------------------------------------------------------------------------
def build_binary_metrics(device: str | torch.device = "cpu") -> MetricCollection:
    """Rappel/précision/F1 sur la tâche binaire dérivée.

    Objectif métier n°2 du projet (détection d'anomalie) : collapse les
    classes critiques + modérées en une seule classe "anormale".
    """
    metrics = MetricCollection(
        {
            "recall": BinaryRecall(),
            "precision": BinaryPrecision(),
            "f1": BinaryF1Score(),
        },
        prefix="abnormal_",
    )
    return metrics.to(device)


def to_binary_abnormal(labels_or_preds: torch.Tensor) -> torch.Tensor:
    """Convertit un tenseur d'indices de classe en tenseur binaire 0/1.

    Fonctionne aussi bien sur les labels vrais que sur les preds argmax
    (1 = classe anormale).
    """
    abnormal_idx_tensor = torch.tensor(ABNORMAL_LABELS, device=labels_or_preds.device)
    return torch.isin(labels_or_preds, abnormal_idx_tensor).long()


# ---------------------------------------------------------------------------
# 4. Sélection de modèle : gate sur le rappel critique, puis max F1 macro
# ---------------------------------------------------------------------------
def is_checkpoint_eligible(eval_results: dict, min_recall: float = MIN_CRITICAL_RECALL) -> bool:
    """True si toutes les classes critiques atteignent le rappel minimum exigé."""
    return all(r >= min_recall for r in eval_results["critical_recall"].values())


def select_best_checkpoint(checkpoint_eval_results: list[dict]) -> int | None:
    """Sélectionne le meilleur checkpoint parmi ceux passant le gate de rappel critique.

    checkpoint_eval_results : liste de dicts retournés par evaluate(), un par
    epoch/checkpoint. Retourne l'index du meilleur checkpoint parmi ceux qui
    passent le gate de rappel critique, ou None si aucun ne le passe (auquel
    cas il faut revoir la loss/l'entraînement, pas se rabattre silencieusement
    sur le F1 macro seul).
    """
    eligible = [
        (i, r["f1_macro"])
        for i, r in enumerate(checkpoint_eval_results)
        if is_checkpoint_eligible(r)
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# 5. Confusion matrix : direction dangereuse uniquement
# ---------------------------------------------------------------------------
def critical_to_normal_confusion(confusion_matrix: torch.Tensor) -> dict:
    """Extrait les cellules "critique/modérée prédite comme normale".

    C'est la seule direction d'erreur cliniquement dangereuse.
    """
    dangerous = {}
    for true_i in ABNORMAL_LABELS:
        for pred_j in NORMAL_LABELS:
            count = confusion_matrix[true_i, pred_j].item()
            if count > 0:
                dangerous[f"{FINAL_CLASSES[true_i]} -> {FINAL_CLASSES[pred_j]}"] = count
    return dangerous


# ---------------------------------------------------------------------------
# 6. Affichage lisible des métriques renvoyées par build_metrics
# ---------------------------------------------------------------------------
_SPLIT_PREFIXES = ("train_", "val_", "test_")

# clé "propre" -> clé dans le dict de torchmetrics
_PER_CLASS_KEYS = {
    "recall": "recall_per_class",
    "precision": "precision_per_class",
    "fbeta2": "fbeta2_per_class",
    "auprc": "auprc_per_class",
    "top3_recall": "top3_recall_per_class",
}
_SCALAR_KEYS = (
    "f1_macro",
    "mcc",
    "top3_acc",
    "top3_f1",
    "grouped_critical_recall",
    "critical_screening_ap",
)


def class_names_for(num_classes: int) -> list[str]:
    """Noms de classes à utiliser selon la granularité de la tâche.

    - 2 classes  -> tâche binaire normal / anormal  (BINARY_CRITICALITY_CLASSES)
    - 3 classes  -> criticité normal / interm. / critique  (CRITICALITY_CLASSES)
    - sinon      -> types cellulaires  (FINAL_CLASSES)
    """
    if num_classes == 2:
        return list(BINARY_CRITICALITY_CLASSES)
    if num_classes == 3:
        return list(CRITICALITY_CLASSES)
    return list(FINAL_CLASSES)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _detect_split(keys) -> str:
    for prefix in _SPLIT_PREFIXES:
        if all(k.startswith(prefix) for k in keys):
            return prefix[:-1]
    return ""


def _strip_split(key: str) -> str:
    for prefix in _SPLIT_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def display_metrics(
    results: dict | MetricCollection,
    num_classes: int,
    *,
    title: str | None = None,
    save_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Met en forme les métriques renvoyées par ``build_metrics(...).compute()``.

    Distingue trois cas selon ``num_classes`` (cf. :func:`class_names_for`) :
    binaire, criticité, ou types cellulaires.

    Sépare automatiquement :
      - les métriques scalaires (``f1_macro``, ``mcc``) -> affichées et renvoyées
        dans ``scalars`` ;
      - les métriques vectorielles (``*_per_class``) -> ``DataFrame`` indexé par
        nom de classe, avec une colonne ``support`` (nombre d'images vraies par
        classe, dérivée de la matrice de confusion) quand elle est disponible ;
      - la matrice de confusion -> ``DataFrame`` + heatmaps (comptes bruts et
        version normalisée par ligne, dont la diagonale est le rappel) ;
      - la courbe précision-rappel (``pr_curve``) -> figure ``pr_curve`` (une
        ligne par classe, AP annotée si ``auprc_per_class`` est présent) ;
      - le screening critique (``critical_screening_prc``) -> figure
        ``critical_screening`` (rappel et charge de revue vs seuil) et
        ``screening_operating_points`` (seuil / précision / charge de revue aux
        rappels cibles ``SCREENING_RECALL_TARGETS``).

    Args:
        results: dict retourné par ``.compute()`` (préfixes ``train_/val_/test_``
            tolérés) ou directement la ``MetricCollection`` (``.compute()`` est
            alors appelé).
        num_classes: nombre de classes de la tâche.
        title: titre des figures / en-tête texte ; par défaut le nom du split
            détecté depuis les préfixes de clés.
        save_dir: si fourni, y sauvegarde les CSV et les PNG. ``True`` utilise
            ``reports/figures``.
        verbose: si ``True``, imprime un résumé.

    Returns:
        dict avec ``scalars`` (dict), ``per_class`` (DataFrame),
        ``confusion_matrix`` (DataFrame | None), ``screening_operating_points``
        (dict | None) et ``figures`` (dict de ``matplotlib.figure.Figure``).
    """
    if isinstance(results, MetricCollection):
        results = results.compute()

    split = _detect_split(results)
    if title is None:
        title = split
    # Les métriques "courbe" renvoient un dict {precision, recall, thresholds} :
    # on les met de côté (pas de conversion numpy directe, formes hétérogènes).
    clean = {
        _strip_split(k): _to_numpy(v)
        for k, v in results.items()
        if _strip_split(k) not in CURVE_METRICS
    }
    pr_curve = next((v for k, v in results.items() if _strip_split(k) == "pr_curve"), None)
    screening_curve = next(
        (v for k, v in results.items() if _strip_split(k) == "critical_screening_prc"), None
    )
    names = class_names_for(num_classes)

    scalars = {k: float(clean[k]) for k in _SCALAR_KEYS if k in clean}

    per_class = pd.DataFrame(
        {col: clean[key] for col, key in _PER_CLASS_KEYS.items() if key in clean},
        index=names,
    )
    per_class.index.name = "class"

    cm = clean.get("confusion_matrix")
    cm_df = pd.DataFrame(cm, index=names, columns=names) if cm is not None else None

    if cm is not None:
        # support = nombre d'images vraies par classe = somme de ligne de la matrice
        per_class["support"] = np.asarray(cm).sum(axis=1).astype(int)

    critical_names = (
        [FINAL_CLASSES[i] for i in CRITICAL_LABELS]
        if num_classes == len(FINAL_CLASSES)
        else (["critique"] if num_classes == 3 else [])
    )

    # Points de fonctionnement du screening critique (seuil / précision / charge
    # de revue à rappel cible). Nécessite la matrice de confusion pour connaître
    # le nombre de vrais critiques et le total d'images.
    screening_points = None
    if screening_curve is not None and cm is not None and num_classes == len(FINAL_CLASSES):
        cm_arr = np.asarray(cm)
        screening_points = screening_operating_points(
            screening_curve["precision"],
            screening_curve["recall"],
            screening_curve["thresholds"],
            n_critical=int(cm_arr[CRITICAL_LABELS].sum()),
            n_total=int(cm_arr.sum()),
        )

    if verbose:
        head = f"=== Métriques [{title}] ===" if title else "=== Métriques ==="
        print(head)
        for k, v in scalars.items():
            print(f"  {k:<10s} : {v:.4f}")
        if not per_class.empty:
            print("\nPar classe :")
            print(per_class.round(4).to_string())
        if critical_names and "recall" in per_class.columns:
            flagged = per_class.loc[critical_names, "recall"]
            flagged = flagged[flagged < MIN_CRITICAL_RECALL]
            if len(flagged):
                details = ", ".join(f"{n}={r:.3f}" for n, r in flagged.items())
                print(f"\n⚠ rappel critique < {MIN_CRITICAL_RECALL:.2f} : {details}")
        if screening_points is not None:
            print("\nScreening critique (seuil sur P(Blast)+P(Promyelocyte)) :")
            for target, pt in screening_points.items():
                if pt is None:
                    print(f"  rappel {target:.2f} : inatteignable")
                else:
                    print(
                        f"  rappel {target:.2f} @ seuil {pt['threshold']:.3f} : "
                        f"précision {pt['precision']:.3f}, "
                        f"charge de revue {pt['review_load']:.1%}"
                    )
        if cm_df is not None:
            print("\nMatrice de confusion (lignes = vrai, colonnes = prédit) :")
            print(cm_df.to_string())
            if num_classes == len(FINAL_CLASSES):
                dangerous = critical_to_normal_confusion(torch.as_tensor(cm))
                if dangerous:
                    print("\nErreurs cliniquement dangereuses (anormal -> normal) :")
                    for k, v in dangerous.items():
                        print(f"  {k} : {int(v)}")

    figures: dict = {}

    if cm_df is not None:
        cm_norm = cm_df.div(cm_df.sum(axis=1).replace(0, np.nan), axis=0)
        size = max(4.0, 0.75 * num_classes + 2)
        fig_cm, axes = plt.subplots(1, 2, figsize=(2 * size, size))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
        axes[0].set_title("Comptes bruts")
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=axes[1]
        )
        axes[1].set_title("Normalisée par ligne (diagonale = rappel)")
        for ax in axes:
            ax.set_xlabel("Prédit")
            ax.set_ylabel("Vrai")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        fig_cm.suptitle(f"Matrice de confusion {title}".strip())
        fig_cm.tight_layout()
        figures["confusion_matrix"] = fig_cm

    score_cols = [
        c for c in per_class.columns if c != "support"
    ]  # support = comptes, hors échelle [0,1]
    if score_cols:
        fig_pc, ax = plt.subplots(figsize=(max(6.0, 1.1 * num_classes + 2), 4))
        per_class[score_cols].plot.bar(ax=ax)
        ax.set_ylim(0, 1)
        ax.axhline(
            MIN_CRITICAL_RECALL,
            color="red",
            ls="--",
            lw=1,
            label=f"seuil {MIN_CRITICAL_RECALL:.2f}",
        )
        ax.set_title(f"Métriques par classe {title}".strip())
        ax.set_xlabel("")
        ax.set_ylabel("score")
        ax.legend(loc="lower right", ncol=len(score_cols) + 1, fontsize="small")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        fig_pc.tight_layout()
        figures["per_class"] = fig_pc

    if pr_curve is not None:
        precision = np.asarray(pr_curve["precision"])  # (num_classes, n_thresholds + 1)
        recall = np.asarray(pr_curve["recall"])
        auprc = clean.get("auprc_per_class")
        fig_prc, ax = plt.subplots(figsize=(6, 6))
        for i, cls in enumerate(names):
            order = np.argsort(recall[i])
            label = cls if auprc is None else f"{cls} (AP={auprc[i]:.2f})"
            ax.plot(recall[i][order], precision[i][order], lw=1, label=label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Rappel")
        ax.set_ylabel("Précision")
        ax.set_title(f"Courbe précision-rappel {title}".strip())
        ax.legend(loc="lower left", fontsize="small", ncol=1 if num_classes <= 6 else 2)
        fig_prc.tight_layout()
        figures["pr_curve"] = fig_prc

    if screening_curve is not None:
        # Rappel (et charge de revue) des cellules critiques en fonction du seuil
        # imposé à P(Blast)+P(Promyelocyte) — l'image part en revue si le score
        # dépasse le seuil, quel que soit celui des deux types qui domine.
        thr = np.asarray(screening_curve["thresholds"], dtype=float)
        rec = np.asarray(screening_curve["recall"], dtype=float)[: len(thr)]
        prec = np.asarray(screening_curve["precision"], dtype=float)[: len(thr)]
        fig_scr, ax = plt.subplots(figsize=(6, 4))
        ax.plot(thr, rec, lw=1.5, label="rappel critique")
        if screening_points is not None:
            cm_arr = np.asarray(cm)
            n_crit, n_tot = int(cm_arr[CRITICAL_LABELS].sum()), int(cm_arr.sum())
            with np.errstate(divide="ignore", invalid="ignore"):
                review_load = np.where(prec > 0, rec * n_crit / (prec * n_tot), np.nan)
            ax.plot(thr, review_load, lw=1.5, ls="--", label="charge de revue")
            for target, pt in screening_points.items():
                if pt is not None:
                    ax.axvline(pt["threshold"], color="grey", lw=0.8, ls=":")
                    # Ancrée sur le bord supérieur du graphe (pas au-dessus, sous
                    # le titre) et pivotée : les seuils proches restent lisibles
                    # côte à côte plutôt que de se chevaucher horizontalement.
                    ax.annotate(
                        f"r={target:.2f}",
                        xy=(pt["threshold"], 1.0),
                        xytext=(3, -3),
                        textcoords="offset points",
                        fontsize="x-small",
                        color="dimgrey",
                        rotation=90,
                        ha="left",
                        va="top",
                    )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Seuil sur P(Blast) + P(Promyelocyte)")
        ax.set_ylabel("Part des images (rappel critique / charge de revue)")
        ax.set_title(f"Screening critique {title}".strip())
        ax.legend(loc="center left", fontsize="small")
        fig_scr.tight_layout()
        figures["critical_screening"] = fig_scr

    if save_dir is not None:
        save_dir = FIGURES_PATH if save_dir is True else Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = f"metrics_{split}" if split else "metrics"
        if not per_class.empty:
            per_class.to_csv(save_dir / f"{stem}_per_class.csv")
        if cm_df is not None:
            cm_df.to_csv(save_dir / f"{stem}_confusion_matrix.csv")
        for fig_name, fig in figures.items():
            fig.savefig(save_dir / f"{stem}_{fig_name}.png", dpi=200, bbox_inches="tight")
        for fig in figures.values():
            plt.close(fig)

    return {
        "scalars": scalars,
        "per_class": per_class,
        "confusion_matrix": cm_df,
        "screening_operating_points": screening_points,
        "figures": figures,
    }
