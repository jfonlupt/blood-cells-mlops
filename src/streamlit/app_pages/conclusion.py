"""Conclusion : résultats, limites et suites du projet."""

from __future__ import annotations

import streamlit as st

st.title("Conclusion", icon=":material/flag:")
st.caption("Synthèse du projet et de son usage dans un contexte clinique.")
st.markdown(
    "L’objectif du projet est simple mais exigeant : **classer les cellules sanguines** et "
    "**mettant en évidence les cas critiques** afin de mieux orienter la relecture humaine."
)

st.header("Objectifs atteints", icon=":material/task_alt:")
project_metrics = st.columns(2)
with project_metrics[0], st.container(border=True):
    st.metric("Accuracy du projet ", "96,2 %")
with project_metrics[1], st.container(border=True):
    st.metric("Repère de comparaison : accuracy rapportée dans la littérature sur PBC seul.", "98,6 %")

obj_cols = st.columns(2)
with obj_cols[0], st.container(border=True):
    st.markdown(":material/check_circle: **Base commune**")
    st.write("Deux sources ont été réunies dans une taxonomie harmonisée.")
with obj_cols[1], st.container(border=True):
    st.markdown(":material/arrow_forward: **Aide au tri et usage clair**")
    st.write("Le modèle aide à orienter les cas incertains vers une relecture humaine, sans se substituer à la décision clinique.")

st.header("Difficultés rencontrées", icon=":material/warning:")
difficulties = st.columns(2)
with difficulties[0]:
    with st.container(border=True):
        st.subheader("Biais de site")
        st.write("Même après harmonisation, un CNN peut encore reconnaître le centre d’acquisition plutôt que la morphologie cellulaire.")
    with st.container(border=True):
        st.subheader("Entraînement sensible")
        st.write("L’initialisation aléatoire, le taux d’apprentissage et le choix des seuils peuvent modifier fortement le rappel sur les cas critiques.")
with difficulties[1]:
    with st.container(border=True):
        st.subheader("Seuils à ajuster")
        st.write("La matrice de coût et le seuil d’alerte doivent être recalibrés selon la tolérance aux erreurs cliniquement coûteuses.")
    with st.container(border=True):
        st.subheader("Calcul coûteux")
        st.write("Comparer plusieurs configurations demande du temps et des ressources de calcul, surtout pour les modèles plus lourds.")

st.header("Pour la suite", icon=":material/trending_up:")
with st.container(border=True):
    st.markdown(
        "1. **Tester d’autres modèles et données** en intégrant des architectures et des jeux de données supplémentaires en hématologie.\n"
        "2. **Valider les seuils** à partir d’un protocole explicite centré sur les cas critiques.\n"
        "3. **Réduire le biais de site** en étudiant des méthodes d’adaptation de domaine et des protocoles de validation plus stricts.\n"
        "4. **Clarifier le rôle du système** en le positionnant comme outil d’assistance au diagnostic, sans le substituer à l’expertise humaine."
    )

st.caption(
    "Les assistants d’IA ont contribué au développement et à la rédaction ; leur contenu a été relu et validé par l’équipe."
)
