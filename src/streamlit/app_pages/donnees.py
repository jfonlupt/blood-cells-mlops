"""Cadre du projet : les deux sources de données et la taxonomie fusionnée."""

from __future__ import annotations

import pandas as pd

import streamlit as st

st.title("Compréhension et manipulation des données", icon=":material/dataset:")

st.header("Cadre", icon=":material/source:")
st.markdown(
    "Le projet s'appuie sur deux jeux de données d'images de cellules sanguines individuelles, "
    "accessibles librement et acquis dans des contextes cliniques distincts."
)

synthese = pd.DataFrame(
    {
        "Caractéristique": [
            "Nombre d'images",
            "Nombre de classes",
            "Population",
        ],
        "PBC (Barcelone)": [
            "17 092",
            "12 (13 étiquettes)",
            "individus sains",
        ],
        "LMU (Munich)": [
            "18 365",
            "15",
            "100 patients LMA + 100 témoins",
        ],
    }
)
st.dataframe(synthese, hide_index=True)

st.markdown(
    "Les deux sources ne partagent pas la même nomenclature. Deux facteurs doivent être pris en "
    "compte dès la constitution des jeux d'entraînement, de validation et de test : "
    "l'**hétérogénéité entre sources** (format, résolution, qualité, colorimétrie) et le "
    "**fort déséquilibre des classes**, aussi bien à l'intérieur de chaque source qu'entre elles."
)

st.header("Catégories des jeux de données", icon=":material/category:")
st.caption("Abréviation, type cellulaire et signification clinique, tels que fournis par chaque source.")

pbc = pd.DataFrame(
    [
        ("BA", "Basophile", "Réactions allergiques et inflammatoires ; syndrome myéloprolifératif"),
        ("BNE", "Neutrophile non segmenté (band)", "Virage à gauche, infections bactériennes aiguës"),
        ("EO", "Éosinophile", "Allergies, parasitoses, hémopathies"),
        ("ERB", "Érythroblaste", "Régénération médullaire, hémolyse, envahissement médullaire"),
        ("IG", "Granulocytes immatures (non sous-typés)", "Virage à gauche marqué, infection sévère, sepsis"),
        ("LY", "Lymphocyte", "Infections virales, syndromes lymphoprolifératifs"),
        ("MMY", "Métamyélocyte", "Virage à gauche, infection, régénération"),
        ("MO", "Monocyte", "Infections chroniques, inflammation, leucémies myélomonocytaires"),
        ("MY", "Myélocyte", "Virage à gauche, syndrome myéloprolifératif"),
        ("NEUTROPHIL", "Neutrophile (non spécifié)", "Défense antibactérienne"),
        ("PLATELET", "Plaquette (thrombocyte)", "Hémostase primaire ; thrombopénie ou thrombocytose"),
        ("PMY", "Promyélocyte", "Leucémie aiguë promyélocytaire (LMA-3), urgence hématologique"),
        ("SNE", "Neutrophile segmenté", "Infections bactériennes, neutropénie"),
    ],
    columns=["Abr.", "Type cellulaire", "Signification clinique"],
)

lmu = pd.DataFrame(
    [
        ("BAS", "Basophile", "Réactions allergiques et inflammatoires"),
        ("EBO", "Érythroblaste", "Régénération érythroïde, hémolyse, envahissement médullaire"),
        ("EOS", "Éosinophile", "Allergies, parasitoses, hémopathies"),
        ("KSC", "Cellule écrasée (smudge, artefact)", "Artefact de préparation, associé à la LLC"),
        ("LYA", "Lymphocyte atypique", "Mononucléose infectieuse, syndromes lymphoprolifératifs"),
        ("LYT", "Lymphocyte typique", "Infections virales, LLC"),
        ("MMZ", "Métamyélocyte", "Virage à gauche, régénération médullaire"),
        ("MOB", "Monoblaste", "Leucémies aiguës monoblastique / myélomonocytaire (LMA-M5/M4)"),
        ("MON", "Monocyte", "Infections chroniques, inflammation, leucémies myélomonocytaires"),
        ("MYB", "Myélocyte", "Virage à gauche, syndrome myéloprolifératif"),
        ("MYO", "Myéloblaste", "Leucémie aiguë myéloïde (≥ 20 % dans sang/moelle)"),
        ("NGB", "Neutrophile (band)", "Virage à gauche, infection bactérienne aiguë"),
        ("NGS", "Neutrophile (segmenté)", "Infections bactériennes, neutropénie"),
        ("PMB", "Promyélocyte (bilobé)", "LMA promyélocytaire, forme microgranulaire"),
        ("PMO", "Promyélocyte", "Leucémie aiguë promyélocytaire (LMA-3)"),
    ],
    columns=["Abr.", "Type cellulaire", "Signification clinique"],
)

tab_pbc, tab_lmu = st.tabs([":material/local_hospital: PBC (Barcelone)", ":material/local_hospital: LMU (Munich)"])
with tab_pbc:
    st.dataframe(pbc, hide_index=True)
with tab_lmu:
    st.dataframe(lmu, hide_index=True)

st.header("Taxonomie fusionnée", icon=":material/merge:")
st.markdown(
    "Les nomenclatures des deux sources sont ramenées à une **taxonomie commune** : regrouper le "
    "plus largement possible entre sites, ne conserver un niveau de détail fin que lorsqu'il est "
    "cliniquement justifié, et n'écarter qu'une seule catégorie qui n'est pas un véritable type "
    "cellulaire. On aboutit à **10 classes entraînables** pour un total inchangé de 35 457 images."
)

taxonomie = pd.DataFrame(
    [
        ("Neutrophile", "BNE, SNE, NEUTROPHIL", "NGB, NGS", 11922, "Normal"),
        ("Lymphocyte", "LY", "LYT, LYA", 5162, "Normal"),
        ("Éosinophile", "EO", "EOS", 3541, "Normal"),
        ("Monocyte", "MO", "MON", 3209, "Normal"),
        ("Plaquette", "PLATELET", "—", 2348, "Normal"),
        ("Basophile", "BA", "BAS", 1297, "Normal"),
        ("Érythroblaste", "ERB", "EBO", 1629, "Intermédiaire"),
        ("Granulocytes immatures", "MMY, MY, IG", "MYB, MMZ", 2360, "Intermédiaire"),
        ("Promyélocyte", "PMY", "PMB, PMO", 680, "Critique"),
        ("Blaste", "—", "MYO, MOB", 3294, "Critique"),
    ],
    columns=["Classe finale", "Étiquettes PBC", "Étiquettes LMU", "Total", "Criticité"],
)


def _highlight_criticite(row: pd.Series) -> list[str]:
    colors = {"Critique": "#E4002B33", "Intermédiaire": "#E4A50033", "Normal": ""}
    color = colors.get(row["Criticité"], "")
    return [f"background-color: {color}" if color else "" for _ in row]


st.dataframe(
    taxonomie.style.apply(_highlight_criticite, axis=1),
    hide_index=True,
    column_config={"Total": st.column_config.NumberColumn(format="localized")},
)
st.caption(
    "La catégorie *smudge* (KSC, 15 images) est exclue de la taxonomie : ce n'est pas un type "
    "cellulaire mais un artefact d'étalement."
)

st.subheader("Criticité clinique", icon=":material/priority_high:")
st.markdown(
    "En complément du type cellulaire, la fusion définit une seconde variable cible : un niveau "
    "de **criticité** en trois paliers, destiné à prioriser la relecture humaine."
)

crit_cols = st.columns(3)
with crit_cols[0]:
    st.markdown(":red-badge[Critique] — 3 974 images")
    st.caption("Blaste, promyélocyte. Marqueurs directs d'une leucémie aiguë ; un faux négatif est un raté diagnostique majeur.")
with crit_cols[1]:
    st.markdown(":orange-badge[Intermédiaire] — 3 989 images")
    st.caption("Granulocytes immatures, érythroblaste. Anormales mais non malignes par nature ; faux négatif = signal manqué, moins urgent.")
with crit_cols[2]:
    st.markdown(":green-badge[Normal] — 27 479 images")
    st.caption("Neutrophile, lymphocyte, éosinophile, monocyte, plaquette, basophile. Aucune sur-pondération nécessaire.")

st.info(
    "Deux classes restent **mono-site** : les plaquettes ne proviennent que de PBC, les blastes "
    "que de LMU — ce sont aussi les deux extrêmes de l'échelle de criticité, un point de vigilance "
    "pour la suite (biais de domaine, page *Pre-processing*).",
    icon=":material/warning:",
)
