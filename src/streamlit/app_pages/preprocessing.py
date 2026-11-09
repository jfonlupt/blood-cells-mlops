"""Harmonisation des sources et augmentation de données."""

from __future__ import annotations

import pandas as pd
from utils.data import figure

import streamlit as st

st.title("Pre-processing et feature engineering", icon=":material/tune:")
st.markdown(
    "Le traitement des images poursuit deux objectifs distincts : **harmoniser** les deux sources "
    "par des transformations déterministes (appliquées identiquement au train, à la validation et "
    "au test), et **augmenter** la variabilité du seul jeu d'entraînement pour améliorer la "
    "généralisation et compenser le déséquilibre des classes rares."
)

st.header("Pipeline d'harmonisation", icon=":material/conversion_path:")
st.markdown("Assemblé dans `build_transform()` sous la forme d'un unique `Compose` (`torchvision.transforms.v2`) :")

synthese = pd.DataFrame(
    [
        ("Résolution et ratio différents", "Distorsion des cellules si redimensionnement naïf", "`ResizeWithPad` 224×224, padding blanc"),
        ("Profondeur de canaux (RGB vs RGBA)", "Lecture incohérente : fond noir", "`convert(\"RGB\")` systématique"),
        ("Teinte du fond (chaud/froid)", "Le modèle apprend le site, pas la cellule", "Reinhard dans l'espace Lab"),
        ("Contraste plus élevé à Munich", "Chromatine et granulations lues différemment", "CLAHE (canal L) + jitter contraste"),
        ("Format / compression (JPEG vs TIFF)", "Artefacts de blocs 8×8 → indice de site", "Ré-encodage JPEG commun (qualité 60)"),
        ("Échelles des valeurs des pixels", "Valeurs non centrées-réduites", "`Normalize` avec les statistiques calculées sur le train"),
        ("Classes rares, fort déséquilibre", "Rappel effondré sur les classes minoritaires", "Augmentation géométrique"),
    ],
    columns=["Différence observée", "Risque pour le modèle", "Traitement retenu"],
)
st.dataframe(synthese, hide_index=True)

st.image(figure("transforms_comparaison.png"), caption="Effet du pipeline sur une cellule de Barcelone (haut) et une de Munich (bas), étape par étape.")

with st.expander("Méthodes de normalisation colorimétrique écartées", icon=":material/block:"):
    st.markdown(
        "- **Recalage RGB canal par canal** : aligne les fonds mais reste trop grossier, la PCA "
        "continue de séparer les sources.\n"
        "- **Macenko** et **Vahadane** : conçues pour l'histologie H&E (hypothèse de deux teintes "
        "dominantes), fausse pour des frottis May-Grünwald-Giemsa — décomposition instable, "
        "dérives de couleur qui détruisent le signal des granulations."
    )

st.header("Data augmentation (train uniquement)", icon=":material/shuffle:")
aug_cols = st.columns(2)
with aug_cols[0]:
    st.markdown("**Géométrie**")
    st.markdown(
        "- Rotation aléatoire ±180° (remplissage blanc)\n"
        "- Miroirs horizontal et vertical\n\n"
        "L'orientation d'une cellule sur la lame est arbitraire ; sert aussi de rééquilibrage "
        "implicite des classes rares."
    )
with aug_cols[1]:
    st.markdown("**Photométrie**")
    st.markdown(
        "- `ColorJitter` luminosité (±0,15) et contraste (±0,15)\n"
        "- Placé **après** Reinhard, sinon annulé par le recalage vers des statistiques fixes\n\n"
        "Amplitude modérée pour ne pas sortir la teinte des granulations de sa plage diagnostique."
    )

st.header("Synthèse", icon=":material/checklist:")


st.header("Le biais de domaine résiste à l'harmonisation", icon=":material/science:")
st.markdown(
    "Pour mesurer le biais de domaine résiduel, un petit CNN est entraîné à une tâche annexe : "
    "**prédire la source** (Barcelone contre Munich) à partir de l'image transformée. Une "
    "harmonisation parfaite rendrait les deux sources indistinguables (précision de 0,5, le hasard)."
)

cols = st.columns(3)
cols[0].metric("Précision médiane (200 configs)", "0,99", help="Sur un pas d'apprentissage stabilisé (10⁻³).")
cols[1].metric("Plage observée", "0,87 – 1,00")
cols[2].metric("Test de Wilcoxon", "p = 2,2 × 10⁻³⁵", help="Rejette H₀ : précision = 0,5 (hasard).")

st.markdown(
    "Aucune combinaison de transformations ne fait tomber la précision près du hasard : le CNN "
    "retrouve le site **quasi parfaitement quel que soit le prétraitement**. Le biais de domaine "
    "est réel et robuste — mais l'harmonisation reste utile : elle réduit la séparabilité "
    "**linéaire** des sources (PCA), même si un réseau non linéaire exploite des indices plus fins."
)
st.image(
    figure("pca_comparaison.png"),
    caption="PCA d'un sous-échantillon d'images, avant (gauche) et après (droite) le pipeline d'harmonisation.",
)
