"""Modèles entraînés et résultats.

Classifieur plat (baseline, architecture avancée, transfert), cascade de modèles, comparaison des
deux paradigmes et journal des runs.
"""

from __future__ import annotations

import pandas as pd
from utils.data import figure, load_leaderboard

import streamlit as st

st.title("Modèles et résultats", icon=":material/insights:")
st.markdown(
    "Deux paradigmes de classification ont été développés en parallèle puis comparés sur la "
    "nomenclature commune à 10 classes :\n\n"
    "- **A — classifieur plat** : un réseau unique prédit directement l'une des 10 classes "
    "(*baseline*, architecture avancée, transfert d'apprentissage) ; l'exigence de rappel "
    "critique est tenue après coup, par un ajustement de seuil ;\n"
    "- **B — cascade de modèles** : une succession de classifieurs spécialisés isole d'abord les "
    "classes faciles et concentre la capacité — et un **rejet calibré** — sur les classes ambiguës."
)

(
    tab_baseline,
    tab_archi,
    tab_transfer,
    tab_cascade,
    tab_comparaison,
    tab_runs,
) = st.tabs(
    [
        ":material/rocket_launch: Baseline",
        ":material/hub: Architecture avancée",
        ":material/sync_alt: Transfert d'apprentissage",
        ":material/account_tree: Cascade de modèles",
        ":material/compare_arrows: Comparaison & benchmarks",
        ":material/table_chart: Journal des runs",
    ]
)

with tab_baseline:
    st.markdown(
        "Un petit CNN maison (`src/models/baseline.py`), sans poids pré-entraînés, sert de "
        "**témoin** : trois blocs de convolution (16, 32, 64 filtres) + max-pooling, puis deux "
        "couches denses. Environ 99 % des paramètres sont concentrés dans la première couche "
        "dense — la principale faiblesse structurelle de la baseline."
    )
    archi_cols = st.columns(3)
    archi_cols[0].metric("Paramètres", "≈ 3,0 M")
    archi_cols[1].metric("Mult-adds (1 image)", "≈ 138 M")
    archi_cols[2].metric("Poids", "≈ 12 Mo")

    st.subheader("Runs finaux", icon=":material/flag:")
    st.caption(
        "Entraînement complet (35 457 images, 50 époques, Adam, scheduler plateau, early stopping) puis évaluation test."
    )
    resultats_finaux = pd.DataFrame(
        [
            ("Criticité (3)", 0.933, 0.929, 0.903, 0.943),
            ("Type (10)", 0.928, 0.916, 0.932, 0.843),
        ],
        columns=["Tâche", "Accuracy", "F1 macro", "MCC", "Rappel critique"],
    )
    st.dataframe(
        resultats_finaux,
        hide_index=True,
        column_config={
            "Accuracy": st.column_config.ProgressColumn(
                min_value=0.8, max_value=1.0, format="%.3f"
            ),
            "F1 macro": st.column_config.ProgressColumn(
                min_value=0.8, max_value=1.0, format="%.3f"
            ),
            "MCC": st.column_config.ProgressColumn(min_value=0.8, max_value=1.0, format="%.3f"),
            "Rappel critique": st.column_config.ProgressColumn(
                min_value=0.8, max_value=1.0, format="%.3f"
            ),
        },
    )
    st.caption(
        "Pour `type`, le rappel critique **groupé** (une cellule critique est-elle envoyée en "
        "revue, quel que soit son type exact) atteint 0,950, contre 0,843 pour le minimum "
        "*par classe* (promyélocyte), la classe critique la plus fragile."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            figure("baseline_confusion_criticality.png"),
            caption="Matrice de confusion — criticité.",
        )
    with col2:
        st.image(
            figure("baseline_confusion_type.png"),
            caption="Matrice de confusion — type (10 classes).",
        )

    col1, col2 = st.columns(2)
    with col1:
        st.image(figure("baseline_training_loss_type.png"), caption="Perte train / validation.")
    with col2:
        st.image(
            figure("baseline_training_lr_type.png"), caption="Taux d'apprentissage (échelle log)."
        )

    st.image(
        figure("baseline_critical_screening_type.png"),
        caption="Rappel critique et charge de revue en fonction du seuil sur P(blaste) + P(promyélocyte).",
    )
    st.markdown(
        "Le seuil par défaut (*argmax*) n'atteint pas la cible de rappel critique de 0,95, mais un "
        "**ajustement de seuil** en post-traitement s'en approche à moindre coût : en abaissant le "
        "seuil à ≈ 0,48, le rappel critique atteint 0,95 (précision ≈ 0,87) en n'envoyant que "
        "≈ 12 % des images de test en revue supplémentaire. Pousser jusqu'à un rappel de 0,99 demande "
        "un seuil beaucoup plus bas (≈ 0,01), pour une charge de revue qui reste modérée "
        "(≈ 16 %, précision ≈ 0,71)"
    )

with tab_archi:
    st.markdown(
        "Cette architecture convolutionnelle est entraînée **from scratch**, sans poids pré-entraînés. "
        "Elle est comparée à la Baseline avec les mêmes paramètres de données et d'entraînement "
        "pour chaque tâche ; seule la structure du modèle change."
    )
    st.subheader("Comparaison des architectures testées", icon=":material/insights:")
    st.image(
        figure("comparaison_blocks_classifieur_epochs.png"),
        caption=(
            "F1 macro sur le jeu de test selon le nombre d'époques. Les couleurs distinguent "
            "les blocs convolutionnels et les formes les tailles du classifieur dense."
        ),
    )

    st.subheader("Architecture choisie", icon=":material/account_tree:")
    archi_composants = pd.DataFrame(
        [
            ("Bloc convolutif 1", "2 couches · 32 canaux · noyau 3×3"),
            ("Bloc convolutif 2", "2 couches · 64 canaux · noyau 3×3"),
            ("Normalisation", "Batch Normalization"),
            ("Pooling", "Global Average Pooling"),
            ("Classifieur", "Couche dense de 256 unités"),
        ],
        columns=["Composant", "Configuration"],
    )
    st.dataframe(archi_composants, hide_index=True)

    st.subheader("Runs finaux", icon=":material/flag:")
    st.caption(
        "Entraînements sur les 35 457 images, avec un budget maximal de 50 époques, "
        "scheduler plateau et early stopping, puis évaluation sur le jeu de test."
    )
    archi_final_table = pd.DataFrame(
        [
            ("Criticité (3)", 0.929564, 0.919540, 0.886856, 0.954698, 50),
            ("Type (10)", 0.923687, 0.893514, 0.914287, 0.892157, 46),
        ],
        columns=[
            "Tâche",
            "Accuracy",
            "F1 macro",
            "MCC",
            "Rappel critique min.",
            "Époques",
        ],
    )
    st.dataframe(
        archi_final_table,
        hide_index=True,
        column_config={
            "Accuracy": st.column_config.ProgressColumn(
                min_value=0.8, max_value=1.0, format="%.3f"
            ),
            "F1 macro": st.column_config.ProgressColumn(
                min_value=0.8, max_value=1.0, format="%.3f"
            ),
            "MCC": st.column_config.ProgressColumn(min_value=0.8, max_value=1.0, format="%.3f"),
            "Rappel critique min.": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.3f"
            ),
        },
    )
    st.caption(
        "Pour la tâche type, le rappel critique minimum est le rappel de la classe critique "
        "la plus difficile ; il diffère du rappel critique groupé utilisé pour le dépistage."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            figure("archi_final_confusion_criticality.png"),
            caption="Matrice de confusion — criticité, jeu de test.",
        )
    with col2:
        st.image(
            figure("archi_final_confusion_type.png"),
            caption="Matrice de confusion — type cellulaire, jeu de test.",
        )

    st.image(
        figure("archi_final_critical_screening_type.png"),
        caption="Rappel critique groupé et charge de revue selon le seuil — tâche type.",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            figure("archi_final_training_loss_criticality.png"),
            caption="Perte train / validation — criticité.",
        )
    with col2:
        st.image(
            figure("archi_final_training_loss_type.png"),
            caption="Perte train / validation — type cellulaire.",
        )

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            figure("archi_final_training_lr_criticality.png"),
            caption="Taux d'apprentissage — criticité.",
        )
    with col2:
        st.image(
            figure("archi_final_training_lr_type.png"),
            caption="Taux d'apprentissage — type cellulaire.",
        )

with tab_transfer:
    st.markdown(
        "Le transfert d'apprentissage remplace le tronc convolutif *from scratch* par un "
        "extracteur **pré-entraîné sur ImageNet**, éventuellement ré-affiné, avec la même tête "
        "*global average pooling* + classifieur compact retenue précédemment."
    )

    with st.expander("Repères dans la littérature", icon=":material/menu_book:"):
        st.markdown(
            "- **Matek et al.** — ResNeXt-50 sur Matek-19 (LMU seul) : ≈ 90 % d'accuracy sur la "
            "discrimination binaire blaste / non-blaste, le benchmark clinique historique.\n"
            "- **Acevedo et al.** — VGG-16 et InceptionV3 sur PBC seul : ≈ 96 %.\n"
            "- Travaux plus récents sur PBC seul (8 classes) — EfficientNet-B0/B3, "
            "EfficientNetV2-S, DenseNet-121 : 98–98,6 %.\n"
            "- Une revue dédiée aux *Vision Transformers* sur ce type d'images les juge "
            "compétitifs face aux CNN classiques, à condition de données et d'augmentation "
            "suffisantes."
        )
        st.caption(
            "La grille de ce projet reprend les principaux modèles de cette littérature "
            "(ResNet-50, EfficientNet-B0, EfficientNetV2-S, DenseNet-121, ViT-Base/16, "
            "Swin-Tiny), plus ConvNeXt-Tiny. DinoBloom (modèle de fondation spécialisé "
            "hématologie) et l'adaptation de domaine explicite (DANN) n'ont pas pu être testés "
            "faute de budget de calcul."
        )

    backbones = pd.DataFrame(
        [
            ("ResNet-50", "CNN à connexions résiduelles, 50 couches", "23,5 M"),
            (
                "EfficientNet-B0",
                "CNN à mise à l'échelle composée (profondeur/largeur/résolution)",
                "4,0 M",
            ),
            ("DenseNet-121", "CNN à connexions denses entre couches", "7,0 M"),
            ("ConvNeXt-Tiny", "CNN « modernisé » (grands noyaux, LayerNorm, GELU)", "27,8 M"),
            ("ViT-Base/16", "Vision Transformer pur, patches 16×16", "85,8 M"),
            ("EfficientNetV2-S", "CNN, blocs Fused-MBConv puis MBConv", "≈ 21 M"),
            ("Swin-Tiny", "Transformer hiérarchique à fenêtres glissantes", "≈ 28 M"),
        ],
        columns=["Backbone", "Architecture", "Paramètres"],
    )
    st.dataframe(backbones, hide_index=True)
    st.caption("Poids ImageNet (`timm`). Nombre de paramètres entraînables en fine-tuning complet.")

    st.subheader("Grilles de recherche", icon=":material/grid_on:")
    st.markdown(
        "Deux grilles sur les 7 backbones, sous-ensemble de 15 000 images (10 époques, sans "
        "early stopping) : linear probing, fine-tuning complet (sweep du multiplicateur de LR "
        "du backbone), gel partiel, contrôle poids aléatoires, normalisation (ImageNet vs "
        "dataset) et architecture de la tête. **93 entraînements** (≈ 19h12) puis **50** "
        "(≈ 8h12, taille de batch différente — scores systématiquement inférieurs de quelques "
        "points, à garder en tête pour toute comparaison)."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            figure("transfer_grid_freeze_strategy.png"),
            caption="Meilleur score de validation (F1 macro + MCC) par backbone, selon la stratégie de gel.",
        )
    with col2:
        st.image(
            figure("transfer_grid_pretrained.png"),
            caption="Apport des poids ImageNet vs initialisation aléatoire, fine-tuning identique.",
        )
    st.markdown(
        "Le **dégel complet** domine sur les 7 architectures (0,93 à 0,96) : contrairement à un "
        "transfert « classique », les cellules sanguines profitent d'un ré-apprentissage complet "
        "des caractéristiques bas niveau, cohérent avec un domaine visuel (cytologie) très "
        "éloigné d'ImageNet. Le **pré-entraînement ImageNet reste indispensable** : à stratégie "
        "de dégel identique, les poids aléatoires perdent 0,10 à 0,20 de score selon le backbone. "
        "Les stats de normalisation **ImageNet** devancent systématiquement celles du dataset "
        "harmonisé ; l'architecture de la tête n'apporte pas de tendance uniforme (effet marginal "
        "comparé au dégel complet)."
    )

    st.subheader("Runs finaux", icon=":material/flag:")
    st.caption(
        "Backbones retenus après la grille (ViT-Base/16, ConvNeXt-Tiny, EfficientNet-B0, "
        "DenseNet-121), ré-entraînés sur le jeu complet (fine-tuning complet, budget de 100 "
        "époques, scheduler plateau, early stopping) puis évalués sur le test."
    )
    transfer_resultats = pd.DataFrame(
        [
            ("EfficientNet-B0", 14, 0.949, 0.934, 0.949, 0.957, 0.970, 0.956, 0.931),
            ("DenseNet-121", 34, 0.959, 0.949, 0.961, 0.997, 0.979, 0.968, 0.902),
            ("ViT-Base/16", 22, 0.955, 0.944, 0.958, 0.965, 0.977, 0.956, 0.873),
            ("ConvNeXt-Tiny", 19, 0.962, 0.949, 0.961, 0.998, 0.971, 0.963, 0.922),
        ],
        columns=[
            "Backbone",
            "Époques",
            "Accuracy",
            "F1 macro",
            "MCC",
            "Top-3 acc.",
            "AP screening",
            "Rappel crit. groupé",
            "Rappel crit. min.",
        ],
    )
    st.dataframe(
        transfer_resultats,
        hide_index=True,
        column_config={
            "F1 macro": st.column_config.ProgressColumn(
                min_value=0.9, max_value=1.0, format="%.3f"
            ),
            "MCC": st.column_config.ProgressColumn(min_value=0.9, max_value=1.0, format="%.3f"),
            "Rappel crit. groupé": st.column_config.ProgressColumn(
                min_value=0.85, max_value=1.0, format="%.3f"
            ),
            "Rappel crit. min.": st.column_config.ProgressColumn(
                min_value=0.85, max_value=1.0, format="%.3f"
            ),
        },
    )
    st.markdown(
        "Les 4 modèles dépassent nettement la *baseline* (F1 macro 0,916, rappel critique min. "
        "0,843). **ConvNeXt-Tiny et DenseNet-121** obtiennent des performances quasi identiques "
        "(F1 macro 0,949, MCC 0,961) et devancent nettement ViT-Base/16 malgré un nombre de "
        "paramètres entraînables bien inférieur — confirmation que la taille du modèle ne "
        "garantit pas la meilleure performance. Le rappel critique **minimum** reste "
        "l'indicateur le plus irrégulier : EfficientNet-B0, pourtant le plus faible sur les "
        "autres métriques, obtient le meilleur rappel sur le promyélocyte (0,931)."
    )

    st.image(
        figure("transfer_final_per_class_recall.png"),
        caption="Rappel de test par classe des 4 runs finaux, triées par rappel croissant pour DenseNet-121.",
    )
    st.caption(
        "Les granulocytes immatures (IG) et le promyélocyte restent, pour les 4 modèles, les "
        "deux classes les plus difficiles — cohérent avec les confusions déjà observées sur la "
        "baseline. Sur toutes les autres classes, les 4 modèles dépassent 0,89 de rappel."
    )

    st.image(
        figure("transfer_final_training_curves.png"),
        caption="Courbes d'entraînement (perte et F1 macro, train / validation) des 4 runs finaux.",
    )
    st.warning(
        "Arrêt anticipé bien avant le budget de 100 époques (14 à 34 selon le backbone) : la "
        "perte d'entraînement continue de baisser tandis que la validation stagne — la signature "
        "classique d'un **surapprentissage**, une fois le backbone entièrement dégelé sur un jeu "
        "de 35 457 images.",
        icon=":material/warning:",
    )

with tab_cascade:
    st.subheader(
        "Point de départ : la séparabilité classe par classe", icon=":material/scatter_plot:"
    )
    st.markdown(
        "Une première étude oppose chaque classe à toutes les autres (en commençant par les "
        "neutrophiles segmentés puis les promyélocytes) pour mesurer la **séparabilité** des "
        "populations cellulaires :\n\n"
        "- certaines classes se distinguent presque parfaitement (plaquette, éosinophile), "
        "d'autres restent très ambiguës (promyélocyte, myélocyte, métamyélocyte), y compris à l'œil ;\n"
        "- les erreurs observées ont permis d'identifier des cas correspondant potentiellement à "
        "des **erreurs d'annotation** dans les données sources ;\n"
        "- à interpréter avec prudence : une bonne performance ne garantit pas que le modèle "
        "s'appuie sur la morphologie plutôt que sur le **site d'acquisition**."
    )
    st.markdown(
        "Cette hétérogénéité motive la **cascade de modèles**, alternative à un classifieur "
        "*plat* à 10 classes : **router** successivement chaque image à travers des classifieurs "
        "de plus en plus spécialisés, en isolant tôt les classes bien séparées et en ne réservant "
        "les niveaux profonds — et le rejet calibré — qu'aux classes réellement ambiguës."
    )

    st.subheader("Taxonomie fine", icon=":material/category:")
    st.markdown(
        "Entraîné sur **14 classes** (plus fines que les 10 communes au projet) : la classe "
        "*Blaste* est rouverte en myéloblaste et monoblaste, et *Granulocytes immatures* en "
        "myélocyte, métamyélocyte et IG — nécessaire pour que la cascade puisse isoler le "
        "myéloblaste (bien séparé) du monoblaste (critique mais trop rare pour peser sur un "
        "seuil commun). La cascade utilise un **jeu filtré** de 33 577 images (filtre qualité "
        "heuristique), où le monoblaste ne compte que 21 images, dont 3 en test."
    )

    st.subheader("Architecture en cascade (5 niveaux)", icon=":material/account_tree:")
    st.markdown(
        r"""
```mermaid
flowchart TD
    N1["Niveau 1 — 6 sorties"] -->|décision directe| T1["Plaquette, Érythroblaste, Lymphocyte,\nBasophile, Monocyte"]
    N1 --> N2["Niveau 2 — Éosinophile vs reste"]
    N2 -->|décision directe| T2["Éosinophile"]
    N2 --> N3["Niveau 3 — 3 sorties"]
    N3 -->|décision directe| T3["Neutrophile band / segmenté"]
    N3 --> N4["Niveau 4 — nœud dur, 6 sorties\nrejet calibré (T=0,809)"]
    N4 -->|décision ou rejet| T4["Myéloblaste, Promyélocyte, Myélocyte,\nMétamyélocyte, Monoblaste"]
    N4 -->|candidat IG| N5["Niveau 5 — second avis IG\nrejet calibré (T=0,651)"]
    N5 --> T5["IG confirmé (ou rejeté)"]
```
        """
    )
    st.caption(
        "Chaque niveau est un EfficientNet-B0 pré-entraîné distinct, fine-tuné en 2 phases (tête "
        "seule, puis dégel des dernières couches). Le routage suit l'ordre de difficulté "
        "croissante observé lors de l'étude de séparabilité ; le Niveau 4 concentre les 6 classes les "
        "plus confondues. Le Niveau 5 n'est pas un routeur mais une vérification a posteriori des "
        "cas classés IG par le Niveau 4."
    )
    st.markdown(
        "Plutôt qu'un seuillage post-hoc sur une sortie unique, la cascade intègre un mécanisme "
        "de **rejet calibré** aux niveaux 4 et 5 : une image n'est acceptée que si la confiance "
        "dépasse un seuil $T$, calibré pour garantir une précision cible sur les cas acceptés "
        "(0,95 au Niveau 4, 0,97 au Niveau 5) ; en deçà, elle est étiquetée « à confirmer "
        "manuellement » plutôt que forcée vers une classe."
    )

    niveaux = pd.DataFrame(
        [
            ("1 — routeur grossier", 0.977, "—", "—"),
            ("2 — Éosinophile vs reste", 0.997, "—", "—"),
            ("3 — band / segmenté / reste", 0.948, "—", "—"),
            ("4 — nœud dur (6 classes)", 0.852, "0,809", "0,95"),
            ("5 — second avis IG", 1.000, "0,651", "0,97"),
        ],
        columns=["Niveau", "Accuracy val. (phase 2)", "Seuil de rejet T", "Précision cible"],
    )
    st.dataframe(
        niveaux,
        hide_index=True,
        column_config={
            "Accuracy val. (phase 2)": st.column_config.ProgressColumn(
                min_value=0.8, max_value=1.0, format="%.3f"
            ),
        },
    )

    st.subheader("Résultats", icon=":material/insights:")
    res_cols = st.columns(3)
    res_cols[0].metric("Décisions fermes", "93,6 %")
    res_cols[1].metric("Accuracy (décisions fermes)", "94,2 %")
    res_cols[2].metric("Accuracy globale (rejets = erreurs)", "88,2 %")

    st.image(
        figure("hierarchique_confusion_14classes.png"),
        caption="Matrice de confusion de test, taxonomie fine à 14 classes (normalisée par ligne).",
    )
    st.markdown(
        "Les classes sans rejet (Niveaux 1–3) atteignent 91 à 99 % de rappel direct. Le "
        "**nœud dur** (Niveau 4) reste le point faible : myélocyte et métamyélocyte sont "
        "**rejetées dans 3 cas sur 4** plutôt que mal classées — cohérent avec l'objectif du "
        "seuil calibré (préférer un rejet à une erreur silencieuse), révélateur d'une difficulté "
        "que l'architecture atténue sans la résoudre. Le monoblaste (0 % de rappel, 3 images de "
        "test sur 21 au total) reste hors d'atteinte : sa rareté numérique est le facteur "
        "limitant, pas l'architecture."
    )

    st.subheader("Interprétabilité par Grad-CAM", icon=":material/visibility:")
    st.image(
        figure("hierarchique_gradcam_comparaison.png"),
        caption="Superposition Grad-CAM pour quatre classes représentatives : deux bien classées, une du nœud dur en difficulté, et IG.",
    )
    st.error(
        "**Signal d'alerte sur la classe IG.** Le Niveau 5 confirme IG avec 100 % de rappel, un "
        "score qui semblerait excellent isolément. Mais la carte Grad-CAM montre que la cellule "
        "se situe dans la zone la **moins** activée de l'image : le modèle fonde sa décision "
        "presque exclusivement sur des hématies environnantes plutôt que sur la cellule. Ce "
        "résultat corrobore, par une méthode indépendante, l'hypothèse du biais de site (les "
        "artefacts JPEG de PBC) — IG est un candidat exposé : cette étiquette est **100 % PBC**, "
        "alors que le « reste » auquel le Niveau 5 la compare mélange images PBC et LMU (et "
        "compte même des classes exclusivement LMU, comme le myéloblaste). À lire comme un "
        "signal d'alerte, pas une performance à "
        "retenir, tant qu'un test de contrôle « PBC contre LMU » n'a pas isolé la part imputable "
        "au site.",
        icon=":material/priority_high:",
    )

    with st.expander(
        "Pistes explorées, non intégrées à la configuration retenue", icon=":material/science:"
    ):
        st.markdown(
            "- **Niveau supplémentaire** isolant le myéloblaste du nœud dur : gain net sur le "
            "myéloblaste (jusqu'à 94 % de rappel), mais sans confirmation que l'architecture à 6 "
            "niveaux dépasse celle à 5 niveaux une fois les autres pistes contrôlées.\n"
            "- **Normalisation de coloration inter-sites** (Lab) : sans régression mesurée, mais "
            "sans preuve qu'elle corrige le raccourci identifié sur IG.\n"
            "- **Augmentation de données ciblée** sur les classes du nœud dur : améliore "
            "mesurablement l'accuracy de validation du Niveau 4, pas encore combinée aux deux "
            "pistes précédentes.\n\n"
            "Ces trois pistes sont indépendantes et cumulables ; leur combinaison reste à évaluer."
        )

with tab_comparaison:
    st.markdown(
        "Comparaison des trois approches ayant abouti à un run final, ramenées à la nomenclature "
        "commune à 10 classes (la cascade, entraînée sur 14 classes, est remontée par "
        "produit des probabilités le long de la cascade)."
    )
    comparaison = pd.DataFrame(
        [
            ("Baseline", 0.916, 0.932, 0.995, 0.843),
            ("Transfert d'apprentissage (ConvNeXt-Tiny)", 0.949, 0.961, 0.998, 0.922),
            ("Cascade de modèles (remappée 10 classes)", 0.944, 0.959, 0.993, 0.753),
        ],
        columns=["Approche", "F1 macro", "MCC", "Top-3 acc.", "Rappel promyélocyte"],
    )
    st.dataframe(
        comparaison,
        hide_index=True,
        column_config={
            "F1 macro": st.column_config.ProgressColumn(
                min_value=0.85, max_value=1.0, format="%.3f"
            ),
            "MCC": st.column_config.ProgressColumn(min_value=0.85, max_value=1.0, format="%.3f"),
            "Rappel promyélocyte": st.column_config.ProgressColumn(
                min_value=0.6, max_value=1.0, format="%.3f"
            ),
        },
    )
    st.caption(
        "ConvNeXt-Tiny retenu comme représentant du transfert (meilleur run, de très peu) ; "
        "DenseNet-121 obtient des résultats quasi identiques. Le rappel promyélocyte de la "
        "cascade est calculé sur tout le test, sans restriction aux décisions fermes."
    )
    st.markdown(
        "Sur les métriques globales, **la complexité architecturale ne prédit pas la "
        "performance** : la cascade (5 backbones EfficientNet-B0 distincts) se "
        "situe entre baseline et transfert, sans dépasser un unique backbone transféré. L'écart "
        "le plus net porte sur le **rappel du promyélocyte** : 0,922 (transfert), 0,843 "
        "(baseline), 0,753 (cascade) — ce dernier chiffre doit être relativisé, cette classe "
        "étant régulièrement *rejetée* plutôt que mal classée dans le nœud dur de la cascade, ce "
        "qu'un rappel « sans rejet » pénalise alors qu'en usage réel un rejet déclenche une "
        "relecture humaine, pas une erreur silencieuse."
    )

    st.warning(
        "**Limites de la comparaison** — taxonomie 14 classes remappée vers 10 ; jeu filtré de "
        "33 577 images (test : 5 037) contre 35 457 (test : 5 317) ; pré-traitements différents ; "
        "un seul run par approche, alors que l'écart de F1 macro entre ConvNeXt-Tiny et la cascade "
        "(0,005) est inférieur à la variance inter-seed.",
        icon=":material/rule:",
    )
    qualitatif = pd.DataFrame(
        [
            (
                "Décision",
                "Une seule décision argmax sur 10 classes",
                "5 décisions spécialisées successives",
            ),
            (
                "Incertitude",
                "Seuil post-hoc sur P(blaste) + P(promyélocyte)",
                "Rejet calibré intégré (niveaux 4 et 5) : 6,4 % des images en relecture",
            ),
            ("Nombre de modèles", "1", "5 EfficientNet-B0"),
            (
                "Interprétabilité",
                "Confusion agrégée entre 10 classes",
                "Erreur localisable au niveau qui l'a produite",
            ),
            (
                "Point fort",
                "Performance brute, pipeline simple",
                "Abstention explicite sur les cas ambigus",
            ),
            (
                "Limite",
                "Chaque erreur est silencieuse",
                "Pipeline plus lourd ; nœud dur encore largement rejeté",
            ),
        ],
        columns=["Critère", "A — classifieur plat", "B — cascade de modèles"],
    )
    st.dataframe(qualitatif, hide_index=True)

    st.subheader("Comparaison aux références de la littérature", icon=":material/menu_book:")
    lit_cols = st.columns(3)
    lit_cols[0].metric("Notre meilleur run (type, 10 classes)", "96,2 % acc.", "F1 macro 0,949")
    lit_cols[1].metric("Acevedo et al. (PBC seul, VGG-16/InceptionV3)", "≈ 96 %")
    lit_cols[2].metric("EfficientNet/DenseNet récents (PBC seul, 8 classes)", "98–98,6 %")
    st.markdown(
        "Notre meilleure configuration se situe dans la **fourchette haute des travaux "
        "fondateurs** sur PBC seul, et en retrait des travaux les plus récents sur PBC seul — "
        "un écart cohérent avec la difficulté supplémentaire de la fusion PBC/LMU (domain gap "
        "caractérisé en détail page *Pre-processing*) plutôt qu'avec le choix d'architecture. "
        "Sur la question la plus proche de Matek et al. (détecter qu'une cellule est critique "
        "quel que soit son type), notre **rappel critique groupé atteint 0,963** (déjà 0,950 "
        "pour la baseline), au-dessus des ≈ 90 % de Matek et al. — un résultat encourageant, "
        "mais une comparaison indicative plutôt qu'un remplacement direct (définitions et jeux "
        "de test différents)."
    )
    st.caption(
        "DinoBloom, modèle de fondation spécifique à l'hématologie annoncé comme dépassant les "
        "modèles génériques, n'a pas pu être testé faute de budget de calcul — prochaine "
        "référence naturelle à confronter à ces résultats."
    )

with tab_runs:
    df = load_leaderboard()
    if df.empty:
        st.warning("`results/leaderboard.csv` introuvable ou vide.", icon=":material/warning:")
    else:
        st.caption(
            f"{len(df)} runs enregistrés dans le journal cumulatif des expériences (`results/leaderboard.csv`)."
        )

        filter_cols = st.columns(3)
        tasks = sorted(df["task"].dropna().unique()) if "task" in df else []
        models = sorted(df["model.name"].dropna().unique()) if "model.name" in df else []
        statuses = sorted(df["status"].dropna().unique()) if "status" in df else []
        task_sel = filter_cols[0].multiselect("Tâche", tasks, default=tasks)
        model_sel = filter_cols[1].multiselect("Modèle", models, default=models)
        status_sel = (
            filter_cols[2].segmented_control("Statut", options=statuses) if statuses else None
        )

        filtered = df.copy()
        if task_sel:
            filtered = filtered[filtered["task"].isin(task_sel)]
        if model_sel:
            filtered = filtered[filtered["model.name"].isin(model_sel)]
        if status_sel:
            filtered = filtered[filtered["status"] == status_sel]

        display_cols = [
            c
            for c in [
                "name",
                "task",
                "model.name",
                "model.transfer.backbone",
                "model.architecture",
                "loss.name",
                "test_f1_macro",
                "test_mcc",
                "min_critical_recall",
                "duration_s",
                "status",
            ]
            if c in filtered.columns
        ]
        table = filtered[display_cols]
        if "test_f1_macro" in table.columns:
            table = table.sort_values("test_f1_macro", ascending=False, na_position="last")
        st.dataframe(
            table,
            hide_index=True,
            column_config={
                "test_f1_macro": st.column_config.ProgressColumn(
                    "F1 macro (test)", min_value=0, max_value=1, format="%.3f"
                ),
                "min_critical_recall": st.column_config.ProgressColumn(
                    "Rappel critique min.", min_value=0, max_value=1, format="%.3f"
                ),
                "duration_s": st.column_config.NumberColumn("Durée", format="%.0f s"),
            },
        )

        chartable = (
            filtered.dropna(subset=["test_f1_macro"])
            if {"test_f1_macro", "task"}.issubset(filtered.columns)
            else pd.DataFrame()
        )
        if not chartable.empty:
            best_per_task = (
                chartable.sort_values("test_f1_macro", ascending=False)
                .groupby("task", as_index=False)
                .first()
            )
            st.bar_chart(
                best_per_task,
                x="task",
                y="test_f1_macro",
                x_label="Tâche",
                y_label="Meilleur F1 macro (test)",
            )
