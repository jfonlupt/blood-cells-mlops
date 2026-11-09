"""Modèle par transfer learning : construction, tête, gel, groupes de paramètres."""

import pytest
import torch
from torch import nn

from src.config import ModelConfig, TransferConfig
from src.models import build_model
from src.models.transfer import TransferModel
from src.tasks import get_task


def _cfg(**kw):
    return TransferConfig(backbone="resnet18", pretrained=False, **kw)


def test_build_model_transfer_output_dim_matches_task():
    model = build_model(ModelConfig(name="transfer", transfer=_cfg()), get_task("type"))
    assert isinstance(model, TransferModel)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 10)


def test_head_hidden_adds_linear_layers():
    plain = TransferModel(_cfg(head_hidden=[]), num_classes=10)
    mlp = TransferModel(_cfg(head_hidden=[64]), num_classes=10)
    n_linear = lambda m: sum(isinstance(x, nn.Linear) for x in m.head.modules())
    assert n_linear(plain) == 1
    assert n_linear(mlp) == 2


def test_head_dropout_is_wired():
    model = TransferModel(_cfg(head_hidden=[32], head_dropout=0.3), num_classes=10)
    drops = [m for m in model.head.modules() if isinstance(m, nn.Dropout)]
    assert drops and all(d.p == 0.3 for d in drops)


def test_freeze_all_freezes_backbone_but_not_head():
    model = TransferModel(_cfg(freeze="all"), num_classes=10)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_freeze_none_leaves_backbone_trainable():
    model = TransferModel(_cfg(freeze="none"), num_classes=10)
    assert any(p.requires_grad for p in model.backbone.parameters())


def _grad_names(module):
    return {n for n, p in module.named_parameters() if p.requires_grad}


def test_freeze_integer_keeps_only_last_stages_and_head_trainable():
    model = TransferModel(_cfg(freeze=1), num_classes=10)
    trainable = _grad_names(model.backbone)
    # resnet18 : groupes stem / layer1..layer4 ; freeze=1 -> seul layer4 reste
    assert trainable, "au moins un groupe doit rester entraînable"
    assert all(name.startswith("layer4") for name in trainable)
    assert not any(n.startswith(("conv1", "bn1", "layer1.", "layer2.", "layer3.")) for n in trainable)
    assert all(p.requires_grad for p in model.head.parameters())


def test_freeze_integer_keeps_backbone_in_train_mode():
    model = TransferModel(_cfg(freeze=1), num_classes=10)
    model.train()
    assert model.backbone.training  # seul freeze="all" fige les stats BN


def test_freeze_integer_larger_than_group_count_freezes_nothing():
    with pytest.warns(UserWarning, match="rien n'est gelé"):
        model = TransferModel(_cfg(freeze=99), num_classes=10)
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_freeze_integer_param_groups_split_backbone_and_head():
    model = TransferModel(_cfg(freeze=1, backbone_lr_mult=0.1), num_classes=10)
    groups = model.param_groups(base_lr=1e-3)
    assert len(groups) == 2
    backbone_group = next(g for g in groups if g["lr"] == pytest.approx(1e-4))
    assert {id(p) for p in backbone_group["params"]} == {
        id(p) for p in model.backbone.parameters() if p.requires_grad
    }


def test_param_groups_discriminative_lr():
    model = TransferModel(_cfg(freeze="none", backbone_lr_mult=0.01), num_classes=10)
    groups = model.param_groups(base_lr=1e-3)
    assert len(groups) == 2
    lrs = sorted(g["lr"] for g in groups)
    assert lrs == pytest.approx([1e-5, 1e-3])


def test_param_groups_frozen_backbone_yields_head_only():
    model = TransferModel(_cfg(freeze="all", backbone_lr_mult=0.1), num_classes=10)
    groups = model.param_groups(base_lr=1e-3)
    assert len(groups) == 1
    assert groups[0]["lr"] == pytest.approx(1e-3)
    head_ids = {id(p) for p in model.head.parameters()}
    assert {id(p) for p in groups[0]["params"]} == head_ids


def test_train_mode_keeps_frozen_backbone_in_eval():
    model = TransferModel(_cfg(freeze="all"), num_classes=10)
    model.train()
    assert not model.backbone.training  # BatchNorm figée pour le linear probing
    assert model.head.training


def test_train_mode_follows_when_backbone_not_frozen():
    model = TransferModel(_cfg(freeze="none"), num_classes=10)
    model.train()
    assert model.backbone.training
    model.eval()
    assert not model.backbone.training


def test_features_returns_backbone_output_before_head():
    model = TransferModel(_cfg(), num_classes=10)
    out = model.features(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, model.backbone.num_features)
