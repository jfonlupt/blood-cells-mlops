"""Tests de la boucle d'entraînement (Trainer)."""

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import TensorDataset, WeightedRandomSampler

from src.config import TrainConfig
from src.training.callbacks import EarlyStopping
from src.training.trainer import Trainer, _flatten_metric


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def _ds(n=16):
    return TensorDataset(torch.randn(n, 4), torch.randint(0, 2, (n,)))


def _trainer(model, *, scheduler=None, callbacks=(), nb_epoch=3, lr=0.1, batch_size=4):
    cfg = TrainConfig(nb_epoch=nb_epoch, batch_size=batch_size, validation="holdout")
    opt = SGD(model.parameters(), lr=lr)
    return Trainer(
        model,
        cfg,
        nn.CrossEntropyLoss(),
        optimizer=opt,
        scheduler=scheduler(opt) if scheduler else None,
        callbacks=callbacks,
        device="cpu",
    )


def test_step_scheduler_advances_once_per_epoch():
    model = TinyModel()
    opt = SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(opt, step_size=1, gamma=0.5)
    t = Trainer(model, TrainConfig(nb_epoch=3, batch_size=4), nn.CrossEntropyLoss(),
                optimizer=opt, scheduler=scheduler, device="cpu")
    t.fit(_ds(), _ds())
    assert scheduler.last_epoch == 3
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1 * 0.5**3)


def test_plateau_scheduler_receives_val_loss():
    model = TinyModel()
    opt = SGD(model.parameters(), lr=0.1)
    scheduler = ReduceLROnPlateau(opt, mode="min", patience=0, factor=0.5)
    t = Trainer(model, TrainConfig(nb_epoch=4, batch_size=4), nn.CrossEntropyLoss(),
                optimizer=opt, scheduler=scheduler, device="cpu")
    history = t.fit(_ds(), _ds())
    assert scheduler.best == min(history["val_loss"][: scheduler.last_epoch + 1])


def test_lr_logged_in_history():
    history = _trainer(TinyModel(), scheduler=lambda o: StepLR(o, step_size=1)).fit(_ds(), _ds())
    assert "lr" in history and len(history["lr"]) == 3


def test_early_stopping_breaks_and_restores():
    class StopAfterSecond(EarlyStopping):
        restore_calls = 0

        def step(self, m, model):
            super().step(m, model)
            return self._epoch >= 1

        def restore(self, model):
            type(self).restore_calls += 1
            super().restore(model)

    model = TinyModel()
    with torch.no_grad():
        model.linear.weight.fill_(0.123)
    frozen = model.linear.weight.clone()
    es = StopAfterSecond(monitor="val_loss", patience=99)
    history = _trainer(model, callbacks=[es], nb_epoch=10, lr=0.0).fit(_ds(), _ds())
    assert len(history["train_loss"]) == 2
    assert StopAfterSecond.restore_calls == 1
    assert torch.equal(model.linear.weight, frozen)


def test_keyboard_interrupt_restores_best_and_reraises():
    class InterruptAfterSecond(EarlyStopping):
        restore_calls = 0

        def step(self, m, model):
            improved = super().step(m, model)
            if self._epoch >= 1:
                raise KeyboardInterrupt
            return improved

        def restore(self, model):
            type(self).restore_calls += 1
            super().restore(model)

    model = TinyModel()
    with torch.no_grad():
        model.linear.weight.fill_(0.123)
    es = InterruptAfterSecond(monitor="val_loss", patience=99)
    t = _trainer(model, callbacks=[es], nb_epoch=10, lr=0.01)

    with pytest.raises(KeyboardInterrupt):
        t.fit(_ds(), _ds())

    assert len(t.history["train_loss"]) == 2  # les 2 epochs terminées avant l'interruption
    assert InterruptAfterSecond.restore_calls == 1  # restore() appelé malgré l'exception


def test_early_stopping_without_validation_raises():
    with pytest.raises(ValueError, match="validation"):
        _trainer(TinyModel(), callbacks=[EarlyStopping()]).fit(_ds())


def test_plateau_without_validation_raises():
    with pytest.raises(ValueError, match="validation"):
        _trainer(TinyModel(), scheduler=lambda o: ReduceLROnPlateau(o)).fit(_ds())


def test_train_sampler_drives_which_samples_are_drawn():
    seen = []

    def spy_loss(logits, target):
        seen.append(target.clone())
        return nn.functional.cross_entropy(logits, target)

    x = torch.randn(20, 4)
    y = torch.cat([torch.zeros(10), torch.ones(10)]).long()
    ds = TensorDataset(x, y)
    # tout le poids sur les 10 exemples de classe 0
    sampler = WeightedRandomSampler([1.0] * 10 + [0.0] * 10, num_samples=20, replacement=True)

    model = TinyModel()
    t = Trainer(
        model,
        TrainConfig(nb_epoch=1, batch_size=5, validation="none"),
        spy_loss,
        optimizer=SGD(model.parameters(), lr=0.0),
        train_sampler=sampler,
        device="cpu",
    )
    t.fit(ds)

    drawn = torch.cat(seen)
    assert len(drawn) == 20
    assert (drawn == 0).all()  # jamais un exemple de classe 1


def test_flatten_metric_scalar_and_vector():
    assert _flatten_metric("test_f1_macro", torch.tensor(0.5)) == pytest.approx(0.5)
    assert _flatten_metric("test_recall_per_class", torch.tensor([0.25, 0.75])) == pytest.approx(
        [0.25, 0.75]
    )


def test_flatten_metric_known_curve_becomes_dict():
    p = torch.tensor([[1.0, 0.5], [0.25, 0.75]])
    r = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    thr = torch.tensor([0.5])
    out = _flatten_metric("test_pr_curve", (p, r, thr))
    assert set(out) == {"precision", "recall", "thresholds"}
    assert out["precision"] == [[1.0, 0.5], [0.25, 0.75]]
    assert out["thresholds"] == [0.5]


def test_flatten_metric_unknown_or_malformed_tuple_returns_none():
    triplet = (torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0]))
    assert _flatten_metric("test_roc_curve", triplet) is None  # nom hors CURVE_METRICS
    assert _flatten_metric("test_pr_curve", triplet[:2]) is None  # pas un triplet


def test_evaluate_flattens_pr_curve_and_drops_unknown_tuples():
    from torchmetrics import MetricCollection
    from torchmetrics.classification import (
        MulticlassF1Score,
        MulticlassPrecisionRecallCurve,
        MulticlassROC,
    )

    model = nn.Linear(4, 3)
    ds = TensorDataset(torch.randn(24, 4), torch.randint(0, 3, (24,)))
    mc = MetricCollection(
        {
            "f1": MulticlassF1Score(num_classes=3),
            "pr_curve": MulticlassPrecisionRecallCurve(num_classes=3, thresholds=10),
            "roc": MulticlassROC(num_classes=3, thresholds=10),  # tuple inconnu -> ignoré
        }
    )
    t = Trainer(
        model,
        TrainConfig(nb_epoch=1, batch_size=8),
        nn.CrossEntropyLoss(),
        optimizer=SGD(model.parameters(), lr=0.0),
        device="cpu",
    )
    out = t.evaluate(ds, metrics=mc)
    assert set(out["test_pr_curve"]) == {"precision", "recall", "thresholds"}
    assert "test_roc" not in out
    assert isinstance(out["test_f1"], float)


def test_evaluate_returns_test_loss():
    from torchmetrics import MetricCollection
    from torchmetrics.classification import MulticlassAccuracy

    model = TinyModel()
    t = _trainer(model)
    t.fit(_ds(), _ds())
    out = t.evaluate(_ds(), metrics=MetricCollection({"acc": MulticlassAccuracy(num_classes=2)}))
    assert "test_loss" in out
