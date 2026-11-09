"""Introduction : contexte et objectifs du projet."""

from __future__ import annotations

import streamlit as st

st.title("Classification des cellules sanguines", icon=":material/bloodtype:")
st.caption(
    "Exploration, pré-traitement et modélisation d'images de frottis sanguins issues de deux sources cliniques."
)

cols = st.columns(4)
cols[0].metric("Images", "35 457", help="Cumul des deux sources après fusion, aucune image perdue.")
cols[1].metric("Sources", "2", help="PBC (Barcelone) et PKG-AML-Cytomorphology (Munich).")
cols[2].metric(
    "Classes finales", "10", help="Taxonomie fusionnée, plus une catégorie exclue (smudge)."
)
cols[3].metric(
    "Déséquilibre max/min",
    "×17,5",
    help="Neutrophile contre promyélocyte, sur la taxonomie fusionnée.",
)

st.header("Contexte", icon=":material/info:")
st.markdown(
    "Le projet interroge plusieurs difficultés propres aux données médicales :\n\n"
    "- Un **déséquilibre de classes marqué.**\n"
    "    - Jusqu'à 17 fois plus d'exemples pour la classe majoritaire que pour la plus rare.\n"
    "- Une **asymétrie clinique du coût des erreurs.**\n"
    "    - Rater une cellule critique est plus grave qu'une fausse alerte.\n"
    "- Un **biais de domaine** entre les deux sources qui persiste malgré l'harmonisation.\n"
    "- La nécessité d'une **fonction de perte et de métriques adaptées** à ce contexte plutôt "
    "que des critères génériques comme l'accuracy."
)

st.header("Objectifs", icon=":material/flag_circle:")
st.markdown(
    "- **Préparer et exploiter** un jeu de données d'images de cellules sanguines afin de "
    "permettre leur classification automatique, par **criticité clinique** "
    "(normal / intermédiaire / critique).\n"
    "- **Entraîner des modèles** et **comparer leurs performances** à celles des **modèles "
    "existants** dans la littérature.\n"
    "- **Proposer une fonction de perte et des métriques d'évaluation adaptées** au déséquilibre "
    "des classes et à l'asymétrie clinique du coût des erreurs."
)

st.info(
    "La modélisation confronte **deux paradigmes** : un **classifieur plat** qui prédit "
    "directement les 10 classes (baseline, architecture avancée, transfert d'apprentissage) et "
    "une **cascade de modèles** spécialisés, capable de s'abstenir sur les cas ambigus.",
    icon=":material/alt_route:",
)
