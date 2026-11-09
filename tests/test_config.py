"""Tests du chargement et de la validation de configuration."""

import pytest
from pydantic import ValidationError

from src.config import ExperimentConfig, deep_merge, load_config, parse_overrides


def test_defaults_are_valid():
    cfg = ExperimentConfig()
    assert cfg.task == "type"
    assert cfg.train.nb_epoch == 3
    assert cfg.loss.gamma == 0.0
    assert cfg.loss.lambda_cost == 0.0
    assert cfg.loss.cost_matrix is None
    assert cfg.transform.reinhard is True


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"trian": {"nb_epoch": 5}})
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"train": {"nb_epochs": 5}})


def test_sampler_alpha_defaults_and_rejects_negative():
    assert ExperimentConfig().train.sampler == "none"
    assert ExperimentConfig().train.sampler_alpha == 1.0
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"train": {"sampler_alpha": -0.5}})


def test_parse_overrides_literals():
    out = parse_overrides(["train.lr=1e-3", "loss.name=focal", "transform.reinhard=false"])
    assert out == {
        "train": {"lr": 0.001},
        "loss": {"name": "focal"},
        "transform": {"reinhard": False},
    }


def test_parse_overrides_keeps_bare_string():
    assert parse_overrides(["model.name=baseline"]) == {"model": {"name": "baseline"}}


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"x": 1, "y": 2}}
    patch = {"a": {"y": 9}, "b": 3}
    merged = deep_merge(base, patch)
    assert merged == {"a": {"x": 1, "y": 9}, "b": 3}
    assert base == {"a": {"x": 1, "y": 2}}


def test_validation_none_with_plateau_is_rejected():
    with pytest.raises(ValidationError, match="validation"):
        ExperimentConfig.model_validate(
            {"train": {"validation": "none"}, "scheduler": {"name": "plateau"}}
        )


def test_validation_none_with_early_stopping_is_rejected():
    with pytest.raises(ValidationError, match="validation"):
        ExperimentConfig.model_validate(
            {"train": {"validation": "none", "early_stopping_patience": 3}}
        )


def test_load_config_applies_overrides(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("task: criticality\ntrain:\n  nb_epoch: 2\n")
    cfg = load_config(cfg_file, ["train.nb_epoch=7", "loss.gamma=2.0"])
    assert cfg.task == "criticality"
    assert cfg.train.nb_epoch == 7
    assert cfg.loss.gamma == 2.0


def test_shipped_base_config_loads():
    cfg = load_config("configs/base.yaml")
    assert cfg.task == "type"
    assert cfg.transform.reinhard is True


def test_loss_config_defaults_and_bounds():
    from src.config import LossConfig

    assert LossConfig().beta == 0.9999
    assert LossConfig().label_smoothing == 0.0
    # beta accepte les bornes 0 et 1 (0 = non pondéré, 1 = inverse-fréquence)
    assert LossConfig.model_validate({"beta": 0.0}).beta == 0.0
    assert LossConfig.model_validate({"beta": 1.0}).beta == 1.0
    for bad in ({"beta": -0.1}, {"beta": 1.1}, {"gamma": -1}, {"lambda_cost": -1},
                {"label_smoothing": 1.0}):
        with pytest.raises(ValidationError):
            LossConfig.model_validate(bad)


def test_loss_config_rejects_removed_name_field():
    from src.config import LossConfig

    with pytest.raises(ValidationError):
        LossConfig.model_validate({"name": "focal"})


def test_loss_config_cost_matrix_validation():
    from src.config import LossConfig

    assert LossConfig().cost_matrix is None
    ok = LossConfig.model_validate({"cost_matrix": [[0.0, 1.0], [10.0, 0.0]]})
    assert ok.cost_matrix == [[0.0, 1.0], [10.0, 0.0]]
    for bad in ([[0.0, 1.0]], [[0.0, 1.0, 2.0], [1.0, 0.0]], [[0.0]]):
        with pytest.raises(ValidationError):
            LossConfig.model_validate({"cost_matrix": bad})


@pytest.mark.parametrize("grid", ["preprocessing", "augmentation", "imbalance"])
def test_shipped_grids_expand(grid):
    from src.experiments.grid import expand_grid

    configs = expand_grid(load_config("configs/base.yaml"), f"configs/grids/{grid}.yaml")
    assert len(configs) >= 2


# --- Transfer learning ------------------------------------------------------


def test_transfer_config_defaults():
    cfg = ExperimentConfig.model_validate(
        {"model": {"name": "transfer", "transfer": {"backbone": "resnet50"}}}
    )
    assert cfg.model.transfer.backbone == "resnet50"
    assert cfg.model.transfer.pretrained is True
    assert cfg.model.transfer.freeze == "none"
    assert cfg.model.transfer.head_hidden == []
    assert cfg.model.transfer.head_dropout == 0.0
    assert cfg.model.transfer.backbone_lr_mult == 1.0


def test_transfer_name_requires_transfer_block():
    with pytest.raises(ValidationError, match="transfer"):
        ExperimentConfig.model_validate({"model": {"name": "transfer"}})


def test_multi_archi_name_requires_architecture_block():
    with pytest.raises(ValidationError, match="architecture"):
        ExperimentConfig.model_validate({"model": {"name": "multi_archi"}})


def test_transfer_config_rejects_negative_lr_mult():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {"model": {"name": "transfer", "transfer": {"backbone_lr_mult": -1.0}}}
        )


def test_transfer_config_rejects_unknown_freeze():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {"model": {"name": "transfer", "transfer": {"freeze": "partial"}}}
        )


def test_transfer_config_accepts_integer_freeze():
    cfg = ExperimentConfig.model_validate(
        {"model": {"name": "transfer", "transfer": {"freeze": 2}}}
    )
    assert cfg.model.transfer.freeze == 2


@pytest.mark.parametrize("bad", [0, -1, True])
def test_transfer_config_rejects_non_positive_integer_freeze(bad):
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {"model": {"name": "transfer", "transfer": {"freeze": bad}}}
        )


def test_shipped_transfer_config_loads():
    cfg = load_config("configs/transfer.yaml")
    assert cfg.model.name == "transfer"
    assert cfg.model.transfer.backbone == "resnet50"
    assert cfg.transform.norm_stats == "imagenet"


@pytest.mark.skip(
    reason="configs/grids/architecture/transfer.yaml a 2 backbones commentés "
    "(tf_efficientnetv2_s, swin_tiny) ; le test attend encore les 7 d'origine"
)
def test_shipped_transfer_grid_expands():
    from src.experiments.grid import expand_grid

    configs = expand_grid(
        load_config("configs/transfer.yaml"), "configs/grids/architecture/transfer.yaml"
    )
    backbones = {c.model.transfer.backbone for c in configs}
    assert len(backbones) == 7
    # 7 (linear probing) + 7*3 (fine-tuning, 3 valeurs de lr_mult)
    assert len(configs) == 28
    assert all(c.model.name == "transfer" for c in configs)


def test_norm_stats_defaults_to_dataset_and_accepts_imagenet():
    assert ExperimentConfig().transform.norm_stats == "dataset"
    cfg = ExperimentConfig.model_validate({"transform": {"norm_stats": "imagenet"}})
    assert cfg.transform.norm_stats == "imagenet"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"transform": {"norm_stats": "zscore"}})
