"""Point d'entrée de la présentation Streamlit du projet cellules sanguines."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Classification des cellules sanguines",
    page_icon=":material/bloodtype:",
    layout="wide",
)

page = st.navigation(
    [
        st.Page(
            "app_pages/introduction.py",
            title="Introduction",
            icon=":material/home:",
        ),
        st.Page(
            "app_pages/donnees.py",
            title="Données",
            icon=":material/dataset:",
        ),
        st.Page(
            "app_pages/exploration.py",
            title="Exploration",
            icon=":material/image_search:",
        ),
        st.Page(
            "app_pages/preprocessing.py",
            title="Pre-processing",
            icon=":material/tune:",
        ),
        st.Page(
            "app_pages/modelisation.py",
            title="Modélisation",
            icon=":material/model_training:",
        ),
        st.Page(
            "app_pages/resultats.py",
            title="Résultats",
            icon=":material/insights:",
        ),
        st.Page(
            "app_pages/demo.py",
            title="Démo",
            icon=":material/biotech:",
        ),
        st.Page(
            "app_pages/conclusion.py",
            title="Conclusion",
            icon=":material/flag:",
        ),
    ],
    position="top",
)

with st.sidebar:
    st.markdown("**Liora — Machine Learning Engineer**")
    st.caption("Projet fil rouge · Classification de cellules sanguines")
    st.caption("Joseph Fonlupt · Romain Hild · Rémi Launay")

page.run()
