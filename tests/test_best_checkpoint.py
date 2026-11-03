"""Verify saved weights follow the best raw validation score from epoch one."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taimma.data.split import SequenceSample
from taimma.evaluation.evaluator import IndependentEvaluator
from taimma.models.sasrec import SASRecModel
from taimma.models.late_fusion import ResidualLateFusionModel, SourceSequentialEncoder
from taimma.training.trainer import SASRecTrainer
from taimma.training.multimodal_trainer import ResidualLateFusionTrainer


def result(score):
    return SimpleNamespace(ndcg_10=score, recall_10=score,
                           to_dict=lambda: {'NDCG@10': round(score, 6), 'Recall@10': round(score, 6)})


@pytest.mark.parametrize('late', [False, True])
@pytest.mark.parametrize('scores,expected', [
    ([.1, .3, .2], 2),
    ([.3, .3, .2], 1),
    ([.1, .10000001, .09], 2),
    ([.3, .2, .1], 1),
])
def test_checkpoint_tracks_best_weights_and_restores_them(tmp_path, monkeypatch, capsys, late, scores, expected):
    model = SASRecModel(3, max_seq_len=3, embedding_dim=8, num_heads=2)
    if late:
        model = ResidualLateFusionModel(model, {'brand': SourceSequentialEncoder('brand', 3,
            feature_dim=8, max_seq_len=3, num_heads=2)}, sources_enabled=('brand',))
        trainer = ResidualLateFusionTrainer(model, IndependentEvaluator(3),
            {'brand': torch.tensor([1, 2, 3])}, {'brand': torch.ones(3)}, patience=5)
    else:
        trainer = SASRecTrainer(model, IndependentEvaluator(3), patience=5)
    model.register_buffer('selection_test_epoch', torch.tensor(0))
    values = iter([.9] + scores)
    monkeypatch.setattr(trainer, 'evaluate', lambda *args, **kwargs: result(next(values)))
    updates = []
    def train_epoch(loader):
        epoch = len(updates) + 1
        model.selection_test_epoch.fill_(epoch)
        updates.append(epoch)
        return float(4 - epoch)
    monkeypatch.setattr(trainer, 'train_epoch', train_epoch)
    sample = SequenceSample(0, [0, 1, 2], 3, 2)
    path = tmp_path / 'checkpoint.pt'
    history, returned = trainer.fit([sample], [sample], {}, max_epochs=3,
                                    checkpoint_path=str(path), verbose=True)
    saved = torch.load(path, weights_only=True)
    assert updates == [1, 2, 3]
    assert saved['epoch'] == history.best_epoch == expected
    assert saved['val_ndcg_10'] == history.best_val_ndcg == scores[expected - 1]
    assert saved['model_state_dict']['selection_test_epoch'].item() == expected
    assert returned.selection_test_epoch.item() == expected
    assert history.initial_validation_metrics['NDCG@10'] == .9
    log = pd.read_csv(tmp_path / 'epochs.csv')
    assert log.epoch.tolist() == [0, 1, 2, 3]
    assert log.phase.tolist() == ['initial_validation', 'train', 'train', 'train']
    assert log.loc[log.selected_checkpoint, 'epoch'].tolist() == [expected]
    assert pd.isna(log.iloc[0].train_loss)
    assert not log.iloc[0].trained
    assert saved['selection_scope'] == 'trained_epochs_only'
    console = capsys.readouterr().out
    assert all(f'Epoch {epoch:03d}' in console for epoch in [1, 2, 3])
    assert 'Initial validation (before training; excluded from selection)' in console


def test_late_fusion_never_selects_initial_checkpoint_even_when_it_is_better(tmp_path, monkeypatch):
    model = ResidualLateFusionModel(SASRecModel(3, max_seq_len=3, embedding_dim=8, num_heads=2),
        {'brand': SourceSequentialEncoder('brand', 3, feature_dim=8, max_seq_len=3, num_heads=2)},
        sources_enabled=('brand',))
    model.register_buffer('selection_test_epoch', torch.tensor(0))
    trainer = ResidualLateFusionTrainer(model, IndependentEvaluator(3),
        {'brand': torch.tensor([1, 2, 3])}, {'brand': torch.ones(3)}, patience=3)
    values = iter([.9, .2, .3, .4])
    monkeypatch.setattr(trainer, 'evaluate', lambda *args, **kwargs: result(next(values)))
    def train_epoch(loader):
        model.selection_test_epoch.add_(1)
        return 1.
    monkeypatch.setattr(trainer, 'train_epoch', train_epoch)
    sample = SequenceSample(0, [0, 1, 2], 3, 2)
    path = tmp_path / 'checkpoint.pt'
    history, returned = trainer.fit([sample], [sample], {}, max_epochs=3,
                                    checkpoint_path=str(path), verbose=False)
    saved = torch.load(path, weights_only=True)
    assert saved['epoch'] == history.best_epoch == 3
    assert saved['val_ndcg_10'] == .4
    assert returned.selection_test_epoch.item() == 3
    assert history.initial_validation_metrics['NDCG@10'] == .9


@pytest.mark.parametrize('interrupted', [False, True])
def test_early_fusion_saves_later_improvement_and_restores_best(tmp_path, monkeypatch, interrupted):
    spec = importlib.util.spec_from_file_location('early_selection_test', ROOT / 'scripts/checkpoint1_early.py')
    early = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(early)
    class FakeEarly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
            self.register_buffer('selection_test_epoch', torch.tensor(0))
        def encode_context(self, ids, **kwargs):
            if self.training and torch.is_grad_enabled():
                self.selection_test_epoch.add_(1)
            return self.weight.expand(len(ids), 1)
        def score_full_catalog(self, context, **kwargs):
            return torch.cat([context, -context], dim=1)
    model = FakeEarly()
    monkeypatch.setattr(early, 'EarlyFusionSASRec', lambda *args, **kwargs: model)
    values = iter([.9, .1, .3, .2, .3, .25])
    evaluations = []
    def evaluate(records):
        evaluations.append(1)
        if interrupted and len(evaluations) == 3:
            raise RuntimeError('deliberate interrupted validation')
        return result(next(values))
    monkeypatch.setattr(early, 'IndependentEvaluator', lambda *args, **kwargs: SimpleNamespace(
        evaluate_ranks=lambda *args, **kwargs: SimpleNamespace(per_user_records=[]),
        result_from_records=evaluate))
    folder = tmp_path / 'early'
    folder.mkdir()
    logger = SimpleNamespace(create_run_dir=lambda key: ('early-test', str(folder)),
        capture_output=lambda path: nullcontext(), save_run_artifacts=lambda *args, **kwargs: None,
        register_result=lambda *args, **kwargs: None)
    sample = SequenceSample(0, [0, 1], 2, 1)
    runner = SimpleNamespace(progress={},device=torch.device('cpu'),backbone_dim=8,logger=logger,
        args=SimpleNamespace(batch_size=1,epochs=3,patience=5),model=lambda *args: None,
        attrs={'category':torch.tensor([1,2]),'brand':torch.tensor([1,2]),'title':torch.zeros(2,3)},
        masks={source:torch.ones(2) for source in ['category','brand','title']},save_progress=lambda: None,
        split=SimpleNamespace(num_items=2,train_base_samples=[sample],val_samples=[sample],test_samples=[sample]))
    if interrupted:
        with pytest.raises(RuntimeError, match='deliberate interrupted validation'):
            early.run_early(runner)
        log = pd.read_csv(folder / 'epochs.csv')
        assert log.epoch.tolist() == [0, 1]
        assert len(json.loads((folder / 'history.json').read_text())['train_losses']) == 1
        assert torch.load(folder / 'checkpoint.pt', weights_only=True)['epoch'] == 1
        return
    early.run_early(runner)
    saved = torch.load(folder / 'checkpoint.pt', weights_only=True)
    history = json.loads((folder / 'history.json').read_text())
    assert saved['epoch'] == history['best_epoch'] == 2
    assert saved['model_state_dict']['selection_test_epoch'].item() == 2
    assert model.selection_test_epoch.item() == 2
    assert history['best_val_ndcg'] == .3
    assert len(history['val_metrics_by_epoch']) == 3
    assert history['initial_validation_metrics']['NDCG@10'] == .9
    log = pd.read_csv(folder / 'epochs.csv')
    assert log.epoch.tolist() == [0, 1, 2, 3]
    assert log.loc[log.selected_checkpoint, 'epoch'].tolist() == [2]


@pytest.mark.parametrize('late', [False, True])
def test_completed_epoch_logs_survive_later_validation_failure(tmp_path, monkeypatch, late):
    model = SASRecModel(3, max_seq_len=3, embedding_dim=8, num_heads=2)
    if late:
        model = ResidualLateFusionModel(model, {'brand': SourceSequentialEncoder('brand', 3,
            feature_dim=8, max_seq_len=3, num_heads=2)}, sources_enabled=('brand',))
        trainer = ResidualLateFusionTrainer(model, IndependentEvaluator(3),
            {'brand': torch.tensor([1, 2, 3])}, {'brand': torch.ones(3)})
    else:
        trainer = SASRecTrainer(model, IndependentEvaluator(3))
    calls = []
    def evaluate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError('deliberate interrupted validation')
        return result(.9 if len(calls) == 1 else .2)
    monkeypatch.setattr(trainer, 'evaluate', evaluate)
    monkeypatch.setattr(trainer, 'train_epoch', lambda loader: 1.)
    sample = SequenceSample(0, [0, 1, 2], 3, 2)
    path = tmp_path / 'checkpoint.pt'
    with pytest.raises(RuntimeError, match='deliberate interrupted validation'):
        trainer.fit([sample], [sample], {}, max_epochs=3, checkpoint_path=str(path), verbose=False)
    log = pd.read_csv(tmp_path / 'epochs.csv')
    assert log.epoch.tolist() == [0, 1]
    assert log.loc[log.selected_checkpoint, 'epoch'].tolist() == [1]
    assert len(json.loads((tmp_path / 'history.json').read_text())['val_ndcg_10']) == 1
    assert torch.load(path, weights_only=True)['epoch'] == 1


def test_failed_checkpoint_write_preserves_previous_checkpoint(tmp_path, monkeypatch):
    from taimma.experiments import reporting
    path = tmp_path / 'checkpoint.pt'
    reporting.save_checkpoint_atomic(path, {'epoch': 1})
    original = path.read_bytes()
    def fail(state, output):
        Path(output).write_bytes(b'incomplete checkpoint')
        raise OSError('deliberate storage failure')
    monkeypatch.setattr(reporting.torch, 'save', fail)
    with pytest.raises(OSError, match='deliberate storage failure'):
        reporting.save_checkpoint_atomic(path, {'epoch': 2})
    assert path.read_bytes() == original
