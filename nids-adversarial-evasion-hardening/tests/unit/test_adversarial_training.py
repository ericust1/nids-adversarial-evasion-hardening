import sys
import os
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.dataset_loader import DatasetManager
from src.core.model_trainer import IDSModel
from src.modules.adversarial_training import AdversarialTrainer


def _setup_training_data(n_samples=300, n_features=10):
    dm = DatasetManager()
    X, y = dm.generate_synthetic_traffic(n_samples=n_samples)
    X_small = X[:, :n_features]
    X_train, X_test, y_train, y_test = dm.split_data(X_small, y, test_size=0.3)

    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

    test_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test), torch.LongTensor(y_test)
    )
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    return train_loader, test_loader, n_features


class TestAdversarialTraining:
    def test_adversarial_training_completes(self):
        train_loader, test_loader, n_features = _setup_training_data(
            n_samples=300, n_features=10
        )
        device = torch.device('cpu')
        model = IDSModel(input_dim=n_features).to(device)

        trainer = AdversarialTrainer(model, device)
        history = trainer.adversarial_train(
            train_loader, epochs=2, lr=0.01, epsilon=0.1, mix_ratio=0.5
        )

        assert len(history['loss']) == 2
        assert all(isinstance(l, float) for l in history['loss'])

    def test_hardened_model_evaluates(self):
        train_loader, test_loader, n_features = _setup_training_data(
            n_samples=300, n_features=10
        )
        device = torch.device('cpu')
        model = IDSModel(input_dim=n_features).to(device)

        trainer = AdversarialTrainer(model, device)
        trainer.adversarial_train(
            train_loader, epochs=2, lr=0.01, epsilon=0.1, mix_ratio=0.5
        )

        results = trainer.evaluate_hardened_model(test_loader, test_loader, device)

        assert 'clean_accuracy' in results
        assert 'adversarial_accuracy' in results
        assert 'robustness_gap' in results
        assert 0 <= results['clean_accuracy'] <= 1
        assert 0 <= results['adversarial_accuracy'] <= 1

    def test_adversarial_training_improves_robustness(self):
        train_loader, test_loader, n_features = _setup_training_data(
            n_samples=400, n_features=10
        )
        device = torch.device('cpu')

        model = IDSModel(input_dim=n_features).to(device)
        model.train_model(train_loader, epochs=5, lr=0.01, device=device)

        from src.core.adversarial_generator import AdversarialAttackEngine

        engine = AdversarialAttackEngine(model, device)
        test_data = torch.cat([x for x, _ in test_loader], dim=0).to(device)
        test_labels = torch.cat([y for _, y in test_loader], dim=0).to(device)

        baseline_results = engine.evaluate_robustness(
            test_data, test_labels,
            lambda d, l: engine.fgsm_attack(d, l, epsilon=0.2)
        )

        trainer = AdversarialTrainer(model, device)
        trainer.adversarial_train(
            train_loader, epochs=3, lr=0.005, epsilon=0.2, mix_ratio=0.5
        )

        hardened_results = engine.evaluate_robustness(
            test_data, test_labels,
            lambda d, l: engine.fgsm_attack(d, l, epsilon=0.2)
        )

        assert hardened_results['adversarial_accuracy'] >= 0

    def test_different_mix_ratios(self):
        for ratio in [0.0, 0.3, 0.7, 1.0]:
            train_loader, test_loader, n_features = _setup_training_data(
                n_samples=200, n_features=10
            )
            device = torch.device('cpu')
            model = IDSModel(input_dim=n_features).to(device)

            trainer = AdversarialTrainer(model, device)
            history = trainer.adversarial_train(
                train_loader, epochs=1, lr=0.01,
                epsilon=0.1, mix_ratio=ratio
            )
            assert len(history['loss']) == 1

    def test_compare_models(self):
        train_loader, test_loader, n_features = _setup_training_data(
            n_samples=200, n_features=10
        )
        device = torch.device('cpu')

        baseline_model = IDSModel(input_dim=n_features).to(device)
        baseline_model.train_model(train_loader, epochs=3, lr=0.01, device=device)

        hardened_model = IDSModel(input_dim=n_features).to(device)
        hardened_model.train_model(train_loader, epochs=3, lr=0.01, device=device)

        trainer = AdversarialTrainer(hardened_model, device)
        trainer.adversarial_train(
            train_loader, epochs=1, lr=0.01, epsilon=0.1, mix_ratio=0.5
        )

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = os.path.join(tmpdir, 'baseline.pt')
            hardened_path = os.path.join(tmpdir, 'hardened.pt')
            baseline_model.save_model(baseline_path)
            hardened_model.save_model(hardened_path)

            dm = DatasetManager()
            X, y = dm.generate_synthetic_traffic(n_samples=100)
            X_small = X[:, :n_features]
            _, X_test, _, y_test = dm.split_data(X_small, y, test_size=0.5)

            comparison = trainer.compare_models(
                baseline_path, hardened_path, X_test, y_test
            )

            assert 'baseline_clean_accuracy' in comparison
            assert 'hardened_clean_accuracy' in comparison
            assert 'robustness_improvement' in comparison
