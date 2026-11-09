"""Découpage ponctuel ("one-time") du dataset munich en un socle train/eval fixe.

``build_munich_socle`` répartit les images du munich plat
(``DATASET_PATHS[MUNICH_KEY]``) vers deux répertoires dédiés,
``DATASET_PATHS[MUNICH_TRAIN_KEY]`` et ``DATASET_PATHS[MUNICH_EVAL_KEY]``, via
des hardlinks (pas de copie : pas de doublement d'espace disque). La portion
eval devient un socle fixe, jamais ré-entraîné, utilisé comme référence pour
détecter une régression quand de futurs modèles sont ré-entraînés sur des
données en dérive.

Ce script est destiné à être exécuté une seule fois. Il refuse de s'exécuter
si l'une des destinations existe déjà et contient des fichiers, pour ne
jamais ré-mélanger silencieusement le découpage (ce qui invaliderait toutes
les décisions de promotion ultérieures basées sur ce socle).
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from sklearn.model_selection import train_test_split

from src.constants import BASE_DIR, DATASET_PATHS, MUNICH_EVAL_KEY, MUNICH_KEY, MUNICH_TRAIN_KEY
from src.helpers.data_drift import _run_git

if TYPE_CHECKING:
    from collections.abc import Callable


def _dest_is_nonempty(dest: Path) -> bool:
    return dest.exists() and any(dest.iterdir())


def build_munich_socle(
    eval_fraction: float = 0.2,
    seed: int = 0,
    source: Path | None = None,
    train_dest: Path | None = None,
    eval_dest: Path | None = None,
    repo_root: Path = BASE_DIR,
    run: Callable[[list[str], Path], None] = _run_git,
) -> None:
    """Découpe munich en socle train/eval fixe via hardlinks.

    Une fois le hardlinking terminé, le socle est versionné via dvc/git
    (``dvc add``, ``git add``, ``git commit``, ``dvc push``, ``git push``)
    pour que la garantie "permanent, partagé, one-time" du socle (cf. design
    doc §2 "Socle fixe") tienne réellement : sans ça, le découpage n'existe
    que sur la machine qui l'a lancé.

    Args:
        eval_fraction: Fraction (par classe) réservée au socle eval.
        seed: Graine passée à ``train_test_split`` pour la reproductibilité.
        source: Répertoire source (défaut : ``DATASET_PATHS[MUNICH_KEY]``).
        train_dest: Répertoire de destination train
            (défaut : ``DATASET_PATHS[MUNICH_TRAIN_KEY]``).
        eval_dest: Répertoire de destination eval
            (défaut : ``DATASET_PATHS[MUNICH_EVAL_KEY]``).
        repo_root: Répertoire de travail pour les commandes dvc/git.
        run: Exécuteur de commande injectable, par défaut un vrai appel
            subprocess (même pattern que ``commit_and_push_batch`` dans
            ``src/helpers/data_drift.py``).

    Raises:
        FileExistsError: Si ``train_dest`` ou ``eval_dest`` existe déjà et
            contient au moins un fichier -- avant tout hardlink, pour ne
            jamais ré-exécuter silencieusement le découpage.
    """
    source = Path(source) if source is not None else DATASET_PATHS[MUNICH_KEY]
    train_dest = Path(train_dest) if train_dest is not None else DATASET_PATHS[MUNICH_TRAIN_KEY]
    eval_dest = Path(eval_dest) if eval_dest is not None else DATASET_PATHS[MUNICH_EVAL_KEY]

    if _dest_is_nonempty(train_dest):
        raise FileExistsError(f"{train_dest} existe déjà et n'est pas vide.")
    if _dest_is_nonempty(eval_dest):
        raise FileExistsError(f"{eval_dest} existe déjà et n'est pas vide.")

    # Sorted explicitly: Path.glob's order follows filesystem iteration order,
    # not just `seed` -- now that the socle is persisted/shared (dvc/git
    # below), the split must be fully deterministic given the same inputs,
    # independent of the machine/filesystem that runs this.
    image_paths = sorted(
        list(source.glob("**/*.jpg"))
        + list(source.glob("**/*.png"))
        + list(source.glob("**/*.tiff"))
    )

    by_class: dict[str, list[Path]] = defaultdict(list)
    for p in image_paths:
        by_class[p.stem.split("_")[0]].append(p)

    # Split indépendamment par classe (plutôt qu'un unique split global
    # stratifié) pour que chaque classe soit représentée dans le socle eval
    # dès qu'elle en a assez pour être scindée. Une classe trop petite pour
    # être scindée (< 2 images) part intégralement en train : on ne peut pas
    # réserver une fraction d'un singleton sans en faire disparaître l'autre
    # côté.
    train_paths: list[Path] = []
    eval_paths: list[Path] = []
    for paths in by_class.values():
        if len(paths) < 2:
            train_paths.extend(paths)
            continue
        class_train, class_eval = train_test_split(
            paths, test_size=eval_fraction, random_state=seed
        )
        train_paths.extend(class_train)
        eval_paths.extend(class_eval)

    train_dest.mkdir(parents=True, exist_ok=True)
    eval_dest.mkdir(parents=True, exist_ok=True)
    for src in train_paths:
        os.link(src, train_dest / src.name)
    for src in eval_paths:
        os.link(src, eval_dest / src.name)

    dvc_paths = [str(train_dest) + ".dvc", str(eval_dest) + ".dvc"]
    run(["dvc", "add", str(train_dest), str(eval_dest)], repo_root)
    run(["git", "add", *dvc_paths], repo_root)
    run(
        [
            "git",
            "commit",
            "-m",
            f"munich socle: eval_fraction={eval_fraction}, seed={seed}",
            "--",
            *dvc_paths,
        ],
        repo_root,
    )
    run(["dvc", "push"], repo_root)
    run(["git", "push"], repo_root)


if __name__ == "__main__":
    build_munich_socle()
