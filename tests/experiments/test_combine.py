"""Combinaison de deux runs (type + criticité) : mapping, fusion de probas, validation."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from src.config import ExperimentConfig
from src.constants import (
    BINARY_CRITICALITY_TO_LABEL,
    CRITICALITY_TO_LABEL,
    FINAL_CLASSES,
    TYPE_TO_LABEL,
)
from src.experiments import combine
from src.experiments.combine import (
    combine_probs,
    evaluate_pair,
    load_pair,
    predict_pair,
    type_to_group_mapping,
)
from src.models import build_model
from src.tasks import get_task

_RNG = np.random.default_rng(0)


def _img(path, size=16):
    Image.fromarray((_RNG.random((size, size, 3)) * 255).astype(np.uint8)).save(path)
    return str(path)


def _make_run_dir(tmp_path, name, *, task="type", with_preprocessing=True):
    cfg = ExperimentConfig.model_validate(
        {"task": task, "transform": {"reinhard": False, "clahe": False, "jpeg": False}}
    )
    model = build_model(cfg.model, get_task(task), img_size=cfg.transform.img_size)
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    if with_preprocessing:
        (run_dir / "preprocessing.json").write_text(
            json.dumps(
                {"dataset_mean": [0.5, 0.5, 0.5], "dataset_std": [0.25, 0.25, 0.25], "reinhard": None}
            )
        )
    return run_dir


def test_type_to_group_mapping_criticality():
    mapping = type_to_group_mapping("criticality")
    assert mapping[TYPE_TO_LABEL["Blast"]] == CRITICALITY_TO_LABEL["critique"]
    assert mapping[TYPE_TO_LABEL["Neutrophil"]] == CRITICALITY_TO_LABEL["normal"]
    assert mapping[TYPE_TO_LABEL["IG"]] == CRITICALITY_TO_LABEL["intermediaire"]
    assert len(mapping) == len(FINAL_CLASSES)


def test_type_to_group_mapping_binary_criticality():
    mapping = type_to_group_mapping("binary_criticality")
    assert mapping[TYPE_TO_LABEL["Blast"]] == BINARY_CRITICALITY_TO_LABEL["abnormal"]
    assert mapping[TYPE_TO_LABEL["Neutrophil"]] == BINARY_CRITICALITY_TO_LABEL["normal"]


def test_combine_probs_renormalizes_to_one():
    mapping = type_to_group_mapping("binary_criticality")
    type_probs = torch.full((len(FINAL_CLASSES),), 1.0 / len(FINAL_CLASSES))
    crit_probs = torch.tensor([0.9, 0.1])  # [normal, abnormal]
    combined = combine_probs(type_probs, crit_probs, mapping)
    assert torch.isclose(combined.sum(), torch.tensor(1.0), atol=1e-6)


def test_combine_probs_downweights_types_of_unlikely_group():
    mapping = type_to_group_mapping("binary_criticality")
    type_probs = torch.full((len(FINAL_CLASSES),), 1.0 / len(FINAL_CLASSES))
    crit_probs = torch.tensor([0.9, 0.1])  # normal favorisé
    combined = combine_probs(type_probs, crit_probs, mapping)
    blast_idx = TYPE_TO_LABEL["Blast"]  # abnormal
    neutrophil_idx = TYPE_TO_LABEL["Neutrophil"]  # normal
    assert combined[neutrophil_idx] > combined[blast_idx]


def test_load_pair_accepts_type_and_criticality_in_any_order(tmp_path):
    type_dir = _make_run_dir(tmp_path, "type_run", task="type")
    crit_dir = _make_run_dir(tmp_path, "crit_run", task="criticality")

    type_run, crit_run = load_pair(type_dir, crit_dir)
    assert type_run.task.name == "type"
    assert crit_run.task.name == "criticality"

    type_run2, crit_run2 = load_pair(crit_dir, type_dir)
    assert type_run2.task.name == "type"
    assert crit_run2.task.name == "criticality"


def test_load_pair_accepts_type_and_binary_criticality(tmp_path):
    type_dir = _make_run_dir(tmp_path, "type_run", task="type")
    crit_dir = _make_run_dir(tmp_path, "crit_run", task="binary_criticality")

    type_run, crit_run = load_pair(type_dir, crit_dir)
    assert type_run.task.name == "type"
    assert crit_run.task.name == "binary_criticality"


def test_load_pair_rejects_two_type_runs(tmp_path):
    a = _make_run_dir(tmp_path, "a", task="type")
    b = _make_run_dir(tmp_path, "b", task="type")
    with pytest.raises(ValueError, match="type.*criticalité|criticalité.*type"):
        load_pair(a, b)


def test_load_pair_rejects_criticality_and_binary_criticality(tmp_path):
    a = _make_run_dir(tmp_path, "a", task="criticality")
    b = _make_run_dir(tmp_path, "b", task="binary_criticality")
    with pytest.raises(ValueError, match="type.*criticalité|criticalité.*type"):
        load_pair(a, b)


def test_load_pair_rejects_dataset_task(tmp_path):
    a = _make_run_dir(tmp_path, "a", task="type")
    b = _make_run_dir(tmp_path, "b", task="dataset")
    with pytest.raises(ValueError, match="type.*criticalité|criticalité.*type"):
        load_pair(a, b)


def test_predict_pair_combines_both_models(tmp_path):
    type_dir = _make_run_dir(tmp_path, "type_run", task="type")
    crit_dir = _make_run_dir(tmp_path, "crit_run", task="binary_criticality")
    img_path = _img(tmp_path / "cell.png")

    type_run, crit_run = load_pair(type_dir, crit_dir)
    results = predict_pair(type_run, crit_run, [img_path])

    assert len(results) == 1
    result = results[0]
    assert result["path"] == img_path
    assert result["pred_class"] in FINAL_CLASSES
    assert abs(sum(result["probs"].values()) - 1.0) < 1e-4
    assert set(result["probs"]) == set(FINAL_CLASSES)
    assert "type" in result and "criticality" in result
    assert result["type"]["pred_class"] in FINAL_CLASSES
    assert result["criticality"]["pred_class"] in {"normal", "abnormal"}


def test_evaluate_pair_rejects_mismatched_data_config(tmp_path):
    type_dir = _make_run_dir(tmp_path, "type_run", task="type")
    crit_dir = _make_run_dir(tmp_path, "crit_run", task="criticality")

    crit_cfg = ExperimentConfig.model_validate(
        yaml.safe_load((crit_dir / "config.yaml").read_text())
    )
    crit_cfg = crit_cfg.model_copy(
        update={"data": crit_cfg.data.model_copy(update={"test_size": 0.5})}
    )
    (crit_dir / "config.yaml").write_text(yaml.safe_dump(crit_cfg.model_dump(mode="json")))

    type_run, crit_run = load_pair(type_dir, crit_dir)
    with pytest.raises(ValueError, match="data|seed"):
        evaluate_pair(type_run, crit_run)


def test_evaluate_pair_rejects_mismatched_seed(tmp_path):
    type_dir = _make_run_dir(tmp_path, "type_run", task="type")
    crit_dir = _make_run_dir(tmp_path, "crit_run", task="criticality")

    crit_cfg = ExperimentConfig.model_validate(
        yaml.safe_load((crit_dir / "config.yaml").read_text())
    )
    crit_cfg = crit_cfg.model_copy(update={"seed": crit_cfg.seed + 1})
    (crit_dir / "config.yaml").write_text(yaml.safe_dump(crit_cfg.model_dump(mode="json")))

    type_run, crit_run = load_pair(type_dir, crit_dir)
    with pytest.raises(ValueError, match="data|seed"):
        evaluate_pair(type_run, crit_run)


def test_evaluate_pair_combines_and_computes_metrics(tmp_path, monkeypatch):
    type_dir = _make_run_dir(tmp_path, "type_run", task="type")
    crit_dir = _make_run_dir(tmp_path, "crit_run", task="binary_criticality")
    type_run, crit_run = load_pair(type_dir, crit_dir)

    n = 6
    paths = [_img(tmp_path / f"img_{i}.png") for i in range(n)]
    types = [FINAL_CLASSES[i % len(FINAL_CLASSES)] for i in range(n)]
    fake_df = pd.DataFrame(
        {
            "path": paths,
            "type": types,
            "label": [TYPE_TO_LABEL[t] for t in types],
            "dataset": ["barcelona"] * n,
        }
    )
    monkeypatch.setattr(combine, "create_dataframe", lambda *a, **k: fake_df)
    monkeypatch.setattr(combine, "split_off_test", lambda df, *a, **k: (df.iloc[:0], df))

    result = evaluate_pair(type_run, crit_run)
    assert "f1_macro" in result
    assert _to_shape(result["confusion_matrix"]) == (len(FINAL_CLASSES), len(FINAL_CLASSES))


def _to_shape(value):
    return tuple(torch.as_tensor(value).shape)
