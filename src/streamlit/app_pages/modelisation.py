"""Framework d'expérimentation, fonction de perte, rééquilibrage, optimisation et métriques."""

from __future__ import annotations

import pandas as pd
from utils.data import figure

import streamlit as st

st.title("Modélisation", icon=":material/model_training:")
st.markdown(
    "Classification **multi-classe** sur les 10 types cellulaires fins, ou sur les 3 niveaux de "
    "criticité clinique qui les regroupent. Les deux tâches partagent la même architecture "
    "générale et le même pipeline d'entraînement ; elles ne diffèrent que par le nombre de classes "
    "en sortie et par le réglage de la fonction de perte."
)

tab_loss, tab_sampler, tab_optim, tab_metrics, tab_framework = st.tabs(
    [
        ":material/balance: Fonction de perte",
        ":material/shuffle: Rééchantillonnage",
        ":material/settings: Optimisation",
        ":material/rule: Métriques",
        ":material/build: Framework",
    ]
)

with tab_framework:
    st.markdown(
        "Le projet a d'abord été développé avec Keras, puis migré vers **PyTorch** : écosystème "
        "plus riche pour la classification d'images (`torchvision`, `timm`), meilleure "
        "compatibilité entre versions, meilleures performances sur les machines Mac de l'équipe."
    )
    st.markdown(
        "Un **framework d'expérimentation maison** permet de tester rapidement différents "
        "modèles — architectures maison ou *backbones* pré-entraînés — ainsi que différentes "
        "configurations de transformation, d'augmentation et d'optimisation, sans dupliquer de "
        "code d'entraînement, tout en conservant une trace systématique des configurations "
        "utilisées et des résultats obtenus (recherche par grille, journal des runs)."
    )
    st.markdown(
        "C'est cette infrastructure qui permet les comparaisons appariées multi-seed du grid "
        "search de rééquilibrage — voir l'onglet *Résultats* pour explorer le journal des runs."
    )

with tab_loss:
    st.markdown(
        "Le jeu d'entraînement fusionné est **fortement déséquilibré** (≈ 17:1 entre neutrophile "
        "et promyélocyte) et **cliniquement asymétrique** (rater un blaste est bien plus grave "
        "qu'une fausse alerte). `ClinicalLoss` (`src/models/losses.py`) combine trois leviers :"
    )

    st.markdown(
        "**1. Pondération par effectif** — nombre effectif d'échantillons, $\\beta \\in [0, 1]$"
    )
    st.latex(
        r"w'_c = \begin{cases} 1 & \beta = 0 \\[4pt] "
        r"\dfrac{1-\beta}{1-\beta^{\,n_c}} & 0 < \beta < 1 \\[8pt] "
        r"1/n_c & \beta = 1 \end{cases}"
    )
    st.caption(
        "β = 0 : pas de correction · β = 1 : inverse-fréquence exacte. Les valeurs usuelles sont supérieures à 0,99."
    )

    st.markdown("**2. Terme focal** — concentre l'apprentissage sur les exemples ambigus")
    st.latex(r"(1 - p_y)^{\gamma}, \qquad \gamma = 0 \Rightarrow \text{désactivé}")

    st.markdown("**3. Coût clinique** — pénalise les confusions selon leur gravité (matrice $M$)")
    st.latex(
        r"M_3 = \begin{pmatrix} 0 & 0{,}5 & 1 \\ 4 & 0 & 1 \\ 10 & 3 & 0 \end{pmatrix}"
        r"\qquad M_2 = \begin{pmatrix} 0 & 1 \\ 10 & 0 \end{pmatrix}"
    )
    st.caption(
        "Ordre : normal, intermédiaire, critique. Confondre critique→normal coûte 10× plus qu'une fausse alerte normal→critique."
    )

    st.markdown("**Perte complète**")
    st.latex(
        r"\mathcal{L}_i = \underbrace{-\, w_y\,(1-p_y)^{\gamma}\,\log p_y}_{\text{pondération + focale}} "
        r"+ \underbrace{\lambda \sum_{c} M_{y_i,\,c}\; p_c}_{\text{coût espéré}}"
    )

    st.warning(
        "Une matrice de coût mal calibrée peut sembler inutile : le premier essai sur "
        "`criticality` dégradait le F1 macro sans gain de rappel critique — ses coûts d'erreur "
        "étaient trop élevés pour ce déséquilibre précis. Une fois ajustée, le terme de coût "
        "devient le meilleur levier de la tâche.",
        icon=":material/lightbulb:",
    )

with tab_sampler:
    st.markdown(
        "Avec `train.sampler = weight`, le `DataLoader` tire les exemples avec remise "
        "proportionnellement à un poids de classe $\\pi_c \\propto (1/n_c)^{\\alpha}$. "
        "$\\alpha = 0$ : tirage uniforme (déséquilibre brut) · $\\alpha = 1$ : toutes les classes "
        "équiprobables en espérance."
    )
    st.image(
        figure("sampler_alpha.png"),
        caption="Effectif tiré par classe et par époque selon sampler_alpha.",
    )

    alpha_table = pd.DataFrame(
        [
            (0.00, 17.5, 0.87, "12 000 / 670"),
            (0.25, 8.8, 0.93, "9 350 / 1 070"),
            (0.50, 4.1, 0.97, "6 900 / 1 700"),
            (0.75, 2.0, 0.99, "5 030 / 2 510"),
            (1.00, 1.0, 1.00, "3 530 / 3 550"),
        ],
        columns=[
            "α",
            "Rapport max/min",
            "Entropie normalisée",
            "Neutrophiles / promyélocytes tirés",
        ],
    )
    st.dataframe(alpha_table, hide_index=True)
    st.caption(
        "Le sampler et la pondération de la perte corrigent le même déséquilibre : les cumuler à "
        "pleine intensité le corrige deux fois. Le prix d'un α élevé est une redondance croissante "
        "des classes rares (jusqu'à ×5 vues par époque pour le promyélocyte à α = 1)."
    )

    st.subheader("Grid search sur le rééquilibrage", icon=":material/grid_on:")
    st.markdown(
        "Balayage de $\\beta \\in \\{0, 0{,}99, 0{,}999, 0{,}9999, 1\\}$, $\\gamma \\in \\{0, 0{,}5, 1, 2\\}$, "
        "$\\alpha \\in \\{0, 0{,}5, 1\\}$ et de plusieurs matrices de coût clinique, répliqué sur "
        "plusieurs seeds (47 configurations × 3 seeds)."
    )
    grid = pd.DataFrame(
        [
            ("Criticité (3)", "Class-balanced + focale, sans coût", 0.919, 0.957, False),
            ("Criticité (3)", "Retenue : coût calibré seul (λ=1, β=γ=0)", 0.909, 0.964, True),
            ("Type (10)", "Class-balanced + focale + sampler, sans coût", 0.867, 0.845, False),
            (
                "Type (10)",
                "Retenue : coût calibré + sampler plein (α=1) + β=0,999",
                0.871,
                0.910,
                True,
            ),
        ],
        columns=[
            "Tâche",
            "Configuration (validation, multi-seed)",
            "F1 macro",
            "Rappel critique min.",
            "Retenue",
        ],
    )
    st.dataframe(
        grid,
        hide_index=True,
        column_config={
            "F1 macro": st.column_config.ProgressColumn(
                min_value=0.8, max_value=1.0, format="%.3f"
            ),
            "Rappel critique min.": st.column_config.ProgressColumn(
                min_value=0.8, max_value=1.0, format="%.3f"
            ),
            "Retenue": st.column_config.CheckboxColumn(),
        },
    )
    st.markdown(
        "Sur `criticality`, le coût calibré **seul** dépasse déjà *class-balanced* + focale sur le "
        "rappel critique. Sur `type`, à l'inverse, **cumuler les trois leviers aide** : le "
        "déséquilibre y est trop sévère pour qu'un rééchantillonnage partiel ou une matrice seule "
        "suffisent. La recette qui gagne sur une tâche n'est pas transposable telle quelle à l'autre."
    )

with tab_optim:
    st.markdown("**Optimiseur** — `Adam`, utilisé pour l'ensemble des expériences")
    st.caption(
        "SGD, Adam, AdamW disponibles. Sur `type`, AdamW obtient 0,911 de F1 macro test contre "
        "0,916 pour Adam : aucun effet mesurable. SGD non creusé davantage (coût de réglage "
        "supplémentaire pour un gain non démontré sur les budgets courts du grid search)."
    )

    st.markdown("**Scheduler de taux d'apprentissage** — `plateau` retenu pour les runs finaux")
    scheduler_table = pd.DataFrame(
        [
            ("none", "Taux constant", "Référence de comparaison"),
            (
                "plateau",
                "Réduit le taux si `val_loss` stagne (patience configurable)",
                "Retenu pour les runs finaux",
            ),
            (
                "cosine",
                "Décroissance lisse en demi-cosinus jusqu'à un minimum",
                "Exige de connaître le nombre d'époques à l'avance",
            ),
            ("step", "Divise le taux tous les n épochs/3", "Chutes brutales"),
        ],
        columns=["Politique", "Principe", "Remarque"],
    )
    st.dataframe(scheduler_table, hide_index=True)

    st.markdown("**Early stopping** — arrête l'entraînement dès que `val_loss` cesse de progresser")
    st.caption(
        "Restaure les poids de la meilleure époque plutôt que ceux de la dernière. Configuration "
        "des runs finaux : `scheduler.patience = 3`, `train.early_stopping_patience = 10` — une "
        "marge d'environ trois réductions de palier avant l'arrêt définitif."
    )

with tab_metrics:
    st.markdown(
        "L'*accuracy* globale est écartée comme métrique de pilotage : à 34 % de neutrophiles, un "
        "modèle qui ne prédirait que la classe majoritaire afficherait une accuracy trompeusement "
        "élevée tout en manquant l'intégralité des blastes."
    )
    metrics_table = pd.DataFrame(
        [
            (
                "F1 macro",
                "Globale, par classe équipondérée",
                "Insensible au déséquilibre ; critère secondaire de sélection",
            ),
            ("MCC", "Globale, matrice entière", "Résumé fiable en contexte déséquilibré"),
            (
                "Rappel par classe",
                "Chaque classe",
                "Proportion de cellules critiques réellement détectées",
            ),
            ("Précision par classe", "Chaque classe", "Contrôle du volume de fausses alertes"),
            (
                "Rappel critique min.",
                "Min. sur les classes critiques",
                "Identifie la classe critique la plus fragile d'un run",
            ),
            (
                "Fβ, β = 2",
                "Classes critiques / intermédiaires",
                "Pondère le rappel ×4 : « rater ≫ fausse alerte »",
            ),
            (
                "AUPRC (AP)",
                "Chaque classe, un-contre-tous",
                "Qualité indépendante du seuil ; adaptée aux classes rares",
            ),
            (
                "Matrice de confusion",
                "10 classes + sous-matrice anormal→normal",
                "Isole la direction d'erreur dangereuse",
            ),
            (
                "Rappel critique groupé",
                "Groupe critique (blaste + promyélocyte), argmax",
                "La cellule critique est-elle envoyée en revue, quel que soit son type exact",
            ),
            (
                "AP de criticité (screening)",
                "Groupe critique, indépendant du seuil",
                "Qualité du tri critique/normal ; alimente la courbe de screening",
            ),
            (
                "Top-3 accuracy / F1",
                "Globale (tâches ≥ 5 classes)",
                "Tolère l'hésitation entre classes voisines, utile en aide à la décision",
            ),
            (
                "Rappel / préc. / F1 binaires",
                "Tâche normal vs anormal",
                "Mesure directe du tri pour relecture humaine",
            ),
        ],
        columns=["Métrique", "Portée", "Pourquoi"],
    )
    st.dataframe(metrics_table, hide_index=True)
    # st.info(
    #     "Sélection de checkpoint : filtre d'abord sur un **rappel ≥ 0,95** pour chaque classe "
    #     "critique, puis retient le **F1 macro** maximal parmi les checkpoints éligibles. Si aucun "
    #     "ne passe le filtre, la fonction renvoie `None` plutôt que de relâcher l'exigence clinique.",
    #     icon=":material/verified:",
    # )
