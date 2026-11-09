"""Démonstration : classification d'une image de cellule uploadée, avec Grad-CAM."""

from __future__ import annotations

import pandas as pd
from utils.inference import (
    DEMO_RUNS,
    available_runs,
    domain_bias_examples,
    load_demo_model,
    load_test_df,
    overlay,
    predict,
)

import streamlit as st

st.title("Démonstration", icon=":material/biotech:")
st.caption(
    "Classification d'une cellule par le modèle entraîné, avec explication Grad-CAM des zones "
    "de l'image qui ont pesé dans la décision."
)

runs = available_runs()
if not runs:
    st.warning(
        "Aucun checkpoint configuré n'est disponible sous `results/`.",
        icon=":material/folder_off:",
    )
    st.stop()
if len(runs) < len(DEMO_RUNS):
    missing = [label for label in DEMO_RUNS if label not in runs]
    st.caption(f"Modèles indisponibles, non proposés : {', '.join(missing)}.")

model_label = st.selectbox("Modèle", list(runs))
run_id = runs[model_label]
demo = load_demo_model(run_id)
with st.expander("Détails du modèle", icon=":material/model_training:"):
    m = demo.test_metrics
    st.markdown(
        f"Run `{demo.name}` — backbone **{demo.backbone}**, tâche **{demo.task}** "
        f"({len(demo.class_names)} classes). Score test : F1 macro {m['test_f1_macro']:.3f}, "
        f"MCC {m['test_mcc']:.3f}."
    )

BIAS_MODE = ":material/visibility_off: Cas difficile (biais de fond)"

input_col, result_col = st.columns([1, 2], gap="large")

source = None
label = None
truth = None
bias_note = None
with input_col:
    bias_examples = domain_bias_examples(run_id)
    source_options = ["Uploader", "Test set"] + ([BIAS_MODE] if bias_examples else [])
    mode = st.segmented_control(
        "Source de l'image",
        source_options,
        default="Uploader",
    )
    if mode == BIAS_MODE:
        st.caption(
            "Images où le modèle prédit juste (parfois avec une grande confiance) mais où le "
            "Grad-CAM révèle qu'il s'appuie sur le fond plutôt que sur la cellule — un biais de "
            "site plutôt qu'une erreur de classification."
        )
        choice = (
            st.selectbox(
                "Exemple", bias_examples, format_func=lambda e: e["path"].rsplit("/", 1)[-1]
            )
            if len(bias_examples) > 1
            else bias_examples[0]
        )
        source = choice["path"]
        label = source.rsplit("/", 1)[-1]
        bias_note = choice["note"]
    elif mode == "Test set":
        test_df = load_test_df(run_id)
        if test_df.empty:
            st.warning(
                "Jeu de test indisponible : les dossiers `data/` sont absents.",
                icon=":material/folder_off:",
            )
            st.stop()
        st.caption(f"{len(test_df):,} images jamais vues à l'entraînement.".replace(",", " "))
        counts = test_df[demo.task].value_counts()
        wanted = st.selectbox(
            "Classe à tirer",
            ["Toutes"] + sorted(counts.index),
            format_func=lambda c: c if c == "Toutes" else f"{c} ({counts[c]})",
        )
        if st.button("Image aléatoire", icon=":material/casino:"):
            pool = test_df if wanted == "Toutes" else test_df[test_df[demo.task] == wanted]
            st.session_state.test_pick = int(pool.sample(1).index[0])
            st.session_state.test_pick_run = run_id
        if st.session_state.get("test_pick_run") == run_id and "test_pick" in st.session_state:
            row = test_df.loc[st.session_state.test_pick]
            source = row["path"]
            label = source.rsplit("/", 1)[-1]
            truth = row[demo.task]
    else:
        uploaded = st.file_uploader(
            "Image d'une cellule (JPEG, PNG, TIFF)",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
        )
        if uploaded is not None:
            source = uploaded
            label = uploaded.name
    alpha = st.slider("Opacité de la heatmap", 0.0, 1.0, 0.4, 0.05)

with result_col:
    if source is None:
        st.info(
            "Uploade une image ou tire-en une au hasard dans le test set.", icon=":material/upload:"
        )
    else:
        with st.spinner("Inférence…"):
            result = predict(demo, source)

        top_prob = result.probs[result.pred_class]
        metrics = st.columns(3)
        metrics[0].metric("Classe prédite", result.pred_class)
        metrics[1].metric("Confiance", f"{top_prob:.1%}")
        metrics[2].metric("Criticité", result.criticality)

        if bias_note is not None:
            st.warning(bias_note, icon=":material/visibility_off:")

        if truth is not None:
            if truth == result.pred_class:
                st.success(
                    f"Vérité terrain : **{truth}** — prédiction correcte.",
                    icon=":material/check_circle:",
                )
            else:
                st.error(
                    f"Vérité terrain : **{truth}** — prédiction incorrecte.",
                    icon=":material/cancel:",
                )

        if result.criticality == "critique":
            st.error(
                "Cellule de criticité critique : à envoyer en relecture humaine.",
                icon=":material/priority_high:",
            )
        elif result.criticality == "intermediaire":
            st.warning("Cellule anormale (criticité intermédiaire).", icon=":material/warning:")
        else:
            st.success("Cellule de criticité normale.", icon=":material/check_circle:")

        img_col, cam_col = st.columns(2)
        img_col.image(result.image, caption=f"{label} — après redimensionnement", clamp=True)
        cam_col.image(overlay(result.image, result.cam, alpha), caption="Grad-CAM", clamp=True)

        probs = pd.DataFrame(
            {"Classe": list(result.probs), "Probabilité": list(result.probs.values())}
        ).sort_values("Probabilité", ascending=False)
        st.bar_chart(probs, x="Classe", y="Probabilité", horizontal=True, sort=False)

st.caption(
    "Le modèle a été entraîné sur les images PBC et LMU : une image issue d'un autre microscope "
    "ou d'une autre coloration (photo web, smartphone…) est hors distribution, et le modèle peut "
    "se tromper avec une forte confiance — cf. la page *Pre-processing* sur le biais de domaine."
)
