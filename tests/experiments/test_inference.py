"""Runner d'inférence : chargement d'un run, reconstruction du transform, prédiction."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from src.config import ExperimentConfig
from src.experiments import inference
from src.experiments.inference import (
    load_run,
    predict_images,
    rebuild_eval_transform,
    run_inference,
)
from src.models import build_model
from src.tasks import get_task

_RNG = np.random.default_rng(0)


def _img(path, size=16):
    Image.fromarray((_RNG.random((size, size, 3)) * 255).astype(np.uint8)).save(path)
    return str(path)


def _make_run_dir(tmp_path, *, task="type", with_preprocessing=True):
    cfg = ExperimentConfig.model_validate(
        {"task": task, "transform": {"reinhard": False, "clahe": False, "jpeg": False}}
    )
    model = build_model(cfg.model, get_task(task), img_size=cfg.transform.img_size)
    run_dir = tmp_path / "runs" / "abcd1234"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    if with_preprocessing:
        (run_dir / "preprocessing.json").write_text(
            json.dumps(
                {"dataset_mean": [0.5, 0.5, 0.5], "dataset_std": [0.25, 0.25, 0.25], "reinhard": None}
            )
        )
    return run_dir, model, cfg


def _fake_df(n=12):
    return pd.DataFrame(
        {
            "path": [f"/nonexistent/{i}.png" for i in range(n)],
            "label": [i % 10 for i in range(n)],
            "dataset": ["barcelona"] * n,
        }
    )


def test_load_run_reconstructs_model_and_loads_weights(tmp_path):
    run_dir, orig, _ = _make_run_dir(tmp_path)
    _cfg, model, task = load_run(run_dir)
    assert type(model).__name__ == "BaselineClassifier"
    assert task.name == "type"
    assert not model.training  # .eval()
    for a, b in zip(orig.parameters(), model.parameters()):
        assert torch.equal(a, b)


def test_rebuild_eval_transform_fast_path_needs_no_data(tmp_path, monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("create_dataframe ne doit pas être appelé (chemin rapide)")

    monkeypatch.setattr(inference, "create_dataframe", _boom)
    run_dir, _, cfg = _make_run_dir(tmp_path, with_preprocessing=True)
    transform = rebuild_eval_transform(cfg, run_dir)
    out = transform(torch.randint(0, 255, (3, 40, 40), dtype=torch.uint8))
    assert out.shape == (3, 224, 224)


def test_rebuild_eval_transform_fast_path_rebuilds_reinhard(tmp_path):
    run_dir, _, cfg = _make_run_dir(tmp_path, with_preprocessing=False)
    cfg = cfg.model_copy(update={"transform": cfg.transform.model_copy(update={"reinhard": True})})
    (run_dir / "preprocessing.json").write_text(
        json.dumps(
            {
                "dataset_mean": [0.5, 0.5, 0.5],
                "dataset_std": [0.25, 0.25, 0.25],
                "reinhard": {"target_means": [70.0, 5.0, 10.0], "target_stds": [15.0, 4.0, 6.0]},
            }
        )
    )
    transform = rebuild_eval_transform(cfg, run_dir)
    out = transform(torch.randint(0, 255, (3, 30, 30), dtype=torch.uint8))
    assert out.shape == (3, 224, 224)


def test_rebuild_eval_transform_fallback_replays_pipeline(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        inference, "create_dataframe", lambda *a, **k: (calls.append("df"), _fake_df())[1]
    )
    monkeypatch.setattr(inference, "split_off_test", lambda df, *a, **k: (df, df))
    monkeypatch.setattr(inference, "build_reinhard", lambda *a, **k: None)

    class _FakeTB:
        def __init__(self, *a, **k):
            pass

        def build(self, df, *, training):
            return lambda x: x

    monkeypatch.setattr(inference, "TransformBuilder", _FakeTB)

    run_dir, _, cfg = _make_run_dir(tmp_path, with_preprocessing=False)
    rebuild_eval_transform(cfg, run_dir)
    assert calls == ["df"]


def test_predict_images_multiclass_probs_sum_to_one(tmp_path):
    task = get_task("type")
    logits = torch.zeros(1, 10)
    logits[0, 3] = 5.0
    out = predict_images(lambda x: logits, lambda t: t.float(), task, [_img(tmp_path / "a.png")])
    assert len(out) == 1
    assert set(out[0]["probs"]) == set(task.class_names)
    assert sum(out[0]["probs"].values()) == pytest.approx(1.0)
    assert out[0]["pred_class"] == task.class_names[3]
    assert out[0]["pred_index"] == 3


def test_predict_images_binary_uses_sigmoid(tmp_path):
    task = get_task("dataset")
    out = predict_images(
        lambda x: torch.tensor([[2.0]]), lambda t: t.float(), task, [_img(tmp_path / "b.png")]
    )
    assert out[0]["pred_class"] == task.class_names[1]
    assert sum(out[0]["probs"].values()) == pytest.approx(1.0)


def test_run_inference_end_to_end_writes_predictions(tmp_path):
    run_dir, _, _ = _make_run_dir(tmp_path)
    imgs = [_img(tmp_path / "one.png"), _img(tmp_path / "two.png")]
    out_dir = tmp_path / "out"
    preds = run_inference(run_dir, imgs, output=out_dir)
    assert len(preds) == 2
    assert {p["pred_class"] for p in preds} <= set(get_task("type").class_names)
    saved = json.loads((out_dir / "predictions.json").read_text())
    assert len(saved) == 2
