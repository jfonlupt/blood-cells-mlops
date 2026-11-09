"""Visualisation et statistiques descriptives sur les deux jeux de données."""

from __future__ import annotations

from utils.data import figure

import streamlit as st

st.title("Visualisation et statistiques", icon=":material/image_search:")
st.caption(
    "Analyses exclusivement descriptives — effectifs, proportions, distributions — pour repérer "
    "les caractéristiques susceptibles d'influencer l'apprentissage."
)

st.header("Répartition des classes", icon=":material/bar_chart:")
st.image(figure("repartition_classes.png"), caption="Répartition des classes dans les deux jeux de données (échelle logarithmique).")
st.markdown(
    "Le déséquilibre est le plus marqué dans LMU (**771,3** entre NGS et LYA) ; il vaut **20,6** "
    "pour PBC (**2,7** entre les huit groupes cellulaires principaux)."
)

st.header("Comparaison visuelle des classes et des sources", icon=":material/grid_view:")
col1, col2 = st.columns(2)
with col1:
    st.image(figure("mosaique_classes_pbc.png"), caption="Une image par classe — PBC (Barcelone).")
with col2:
    st.image(figure("mosaique_classes_lmu.png"), caption="Une image par classe — LMU (Munich).")
st.caption("Les images PBC et LMU se distinguent nettement à l'œil, par leur cadrage, leur coloration et leur fond.")

st.header("Distributions colorimétriques", icon=":material/palette:")
tab1, tab2 = st.tabs(["Par source", "Par type cellulaire"])
with tab1:
    st.image(figure("histogrammes_rgb.png"), caption="Histogrammes moyens des canaux R, G, B et des niveaux de gris, PBC contre LMU.")
with tab2:
    st.image(figure("histogrammes_rgb_par_type.png"), caption="Histogramme moyen des niveaux de gris par type cellulaire et par source.")
st.markdown(
    "Ces distributions constituent des variables explicatives potentielles pour un modèle, mais "
    "peuvent aussi encoder une information liée à la **source d'acquisition** plutôt qu'à la "
    "morphologie de la cellule."
)

st.header("Luminosité, contraste et netteté", icon=":material/brightness_6:")
col1, col2 = st.columns(2)
with col1:
    st.image(figure("luminosite_contraste.png"), caption="Luminosité et contraste, par source et par classe.")
    st.caption("La luminosité moyenne est proche entre sources ; le contraste est systématiquement plus élevé et plus dispersé à Munich.")
with col2:
    st.image(figure("nettete.png"), caption="Distribution de la netteté par source (variance du Laplacien).")
    st.caption("Seuil de contrôle au 2ᵉ percentile : ≈ 40 images sur 2 000 par source signalées comme potentiellement floues.")

st.header("Format, taille et poids des images", icon=":material/file_present:")
st.image(figure("format_images.png"), caption="Résolutions dominantes (gauche) et poids des fichiers en échelle logarithmique (droite), par source.")

st.warning(
    "La compression **JPEG** de Barcelone laisse des artefacts caractéristiques (blocs 8×8) que "
    "les TIFF de Munich n'ont pas — un indicateur de site quasi parfait, invisible à l'œil, qui "
    "survit à la plupart des normalisations colorimétriques.",
    icon=":material/warning:",
)

st.header("Séparabilité par source (PCA)", icon=":material/scatter_plot:")
st.image(
    figure("pca_source.png"),
    caption="Projection PCA d'un échantillon d'images. Colorée par source (gauche), les deux nuages se séparent nettement ; "
    "colorée par type cellulaire (droite), aucune structure nette n'apparaît.",
)
st.markdown(
    "L'ensemble de ces régularités de bas niveau — résolution, artefacts de compression, "
    "bordures, dynamique des pixels — offre au modèle autant de moyens de prédire la **source** "
    "plutôt que le **type cellulaire**, ce qui dégrade la généralisation."
)

st.header("Synthèse", icon=":material/checklist:")

st.dataframe(
    [
        {"Caractéristique": "Déséquilibre (max/min)", "PBC (Barcelone)": "20,6 (2,7 par groupe)", "LMU (Munich)": "771,3"},
        {"Caractéristique": "Résolution dominante", "PBC (Barcelone)": "360 × 363 px", "LMU (Munich)": "400 × 400 px"},
        {"Caractéristique": "Canaux", "PBC (Barcelone)": "3 (RGB)", "LMU (Munich)": "4 (RGBA)"},
        {"Caractéristique": "Variation du Laplacien", "PBC (Barcelone)": "Répartie avant et après 10²", "LMU (Munich)": "Répartie entre 10¹ et 10²"},
        {"Caractéristique": "Densitée", "PBC (Barcelone)": "Fortes variations", "LMU (Munich)": "Plus homogène"},
        {"Caractéristique": "Format / poids médian", "PBC (Barcelone)": "JPEG, ≈ 16 Ko", "LMU (Munich)": "TIFF, ≈ 625 Ko"},
    ],
    hide_index=True,
)
