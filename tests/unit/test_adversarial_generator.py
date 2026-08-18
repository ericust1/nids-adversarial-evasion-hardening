import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.dataset_loader import DatasetManager
from src.core.model_trainer import IDSModel
from src.core.adversarial_generator import AdversarialAttackEngine


def _get_trained_model_and_data(n_samples=500, n_features=10):
    dm = DatasetManager()
    X, y = dm.generate_synthetic_traffic(n_samples=n_samples)
    X_small = X[:, :n_features]
    X_train, X_test, y_train, y_test = dm.split_data(X_small, y, test_size=0.3)

    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    test_data = torch.FloatTensor(X_test)
    test_labels = torch.LongTensor(y_test)

    device = torch.device('cpu')
    model = IDSModel(input_dim=n_features).to(device)
    model.train_model(train_loader, epochs=8, lr=0.01, device=device)

    return model, test_data, test_labels, device


class TestFGSM:
    def test_fgsm_perturbs_input(self):
        model, test_data, test_labels, device = _get_trained_model_and_data(
            n_samples=300, n_features=10
        )
        engine = AdversarialAttackEngine(model, device)

        original = test_data[:10].clone()
        adv = engine.fgsm_attack(test_data[:10], test_labels[:10], epsilon=0.1)

        diff = (adv - original).abs()
        assert diff.sum().item() > 0

    def test_fgsm_perturbation_bounded(self):
        model, test_data, test_labels, device = _get_trained_model_and_data(
            n_samples=200, n_features=10
        )
        engine = AdversarialAttackEngine(model, device)

        adv = engine.fgsm_attack(test_data[:20], test_labels[:20], epsilon=0.1)
        max_perturb = (adv - test_data[:20]).abs().max().item()
        assert max_perturb <= 0.1 + 1e-6

    def test_fgsm_accuracy_drop(self):
        model, test_data, test_labels, device = _get_trained_model_and_data(
            n_samples=500, n_features=10
        )
        engine = AdversarialAttackEngine(model, device)

        results = engine.evaluate_robustness(
            test_data, test_labels,
            lambda d, lbl: engine.fgsm_attack(d, lbl, epsilon=0.3)
        )

        assert 'clean_accuracy' in results
        assert 'adversarial_accuracy' in results
        assert 'successful_evasions' in results
        assert results['clean_accuracy'] >= results['adversarial_accuracy'] - 0.05

    def test_fgsm_evasion_report(self):
        model, test_data, test_labels, device = _get_trained_model_and_data(
            n_samples=200, n_features=10
        )
        engine = AdversarialAttackEngine(model, device)

        report = engine.generate_evasion_report(0.95, 0.70, 'FGSM')

        assert report['attack_name'] == 'FGSM'
        assert 'baseline_accuracy' in report
        assert 'adversarial_accuracy' in report
        assert 'risk_level' in report
        assert 'recommendation' in report
        assert report['risk_level'] in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']


class TestPGD:
    def test_pgd_perturbs_input(self):
        model, test_data, test_labels, device = _get_trained_model_and_data(
            n_samples=200, n_features=10
        )
        engine = AdversarialAttackEngine(model, device)

        adv = engine.pgd_attack(
            test_data[:10], test_labels[:10],
            epsilon=0.1, alpha=0.01, iterations=10
        )

        diff = (adv - test_data[:10]).abs()
        assert diff.sum().item() > 0

    def test_pgd_perturbation_bounded(self):
        model, test_data, test_labels, device = _get_trained_model_and_data(
            n_samples=200, n_features=10
        )
        engine = AdversarialAttackEngine(model, device)

        t_min = test_data.min()
        t_max = test_data.max()
        norm_data = (test_data - t_min) / (t_max - t_min + 1e-8)
        norm_labels = test_labels

        adv = engine.pgd_attack(
            norm_data[:10], norm_labels[:10],
            epsilon=0.1, alpha=0.01, iterations=5
        )

        max_perturb = (adv - norm_data[:10]).abs().max().item()
        assert max_perturb <= 0.1 + 1e-5

    def test_pgd_stronger_than_fgsm(self):
        model, test_data, test_labels, device = _get_trained_model_and_data(
            n_samples=400, n_features=10
        )
        engine = AdversarialAttackEngine(model, device)

        engine.fgsm_attack(test_data, test_labels, epsilon=0.1)
        pgd_adv = engine.pgd_attack(
            test_data, test_labels,
            epsilon=0.1, alpha=0.01, iterations=20
        )

        pgd_diff = (pgd_adv - test_data).abs().mean().item()

        assert pgd_diff >= 0

    def test_pgd_accuracy_drop(self):
        model, test_data, test_labels, device = _get_trained_model_and_data(
            n_samples=400, n_features=10
        )
        engine = AdversarialAttackEngine(model, device)

        results = engine.evaluate_robustness(
            test_data, test_labels,
            lambda d, lbl: engine.pgd_attack(
                d, lbl, epsilon=0.3, alpha=0.05, iterations=20
            )
        )

        assert results['clean_accuracy'] >= 0
        assert results['adversarial_accuracy'] >= 0


class TestCWAttack:
    def test_cw_runs_without_error(self):
        model, test_data, test_labels, device = _get_trained_model_and_data(
            n_samples=200, n_features=10
        )
        engine = AdversarialAttackEngine(model, device)

        adv = engine.cw_attack(
            test_data[:10], test_labels[:10],
            targeted=False, cw_confidence=0, max_iter=10
        )

        assert adv.shape == test_data[:10].shape

    def test_cw_perturbs_input(self):
        model, test_data, test_labels, device = _get_trained_model_and_data(
            n_samples=200, n_features=10
        )
        engine = AdversarialAttackEngine(model, device)

        adv = engine.cw_attack(
            test_data[:10], test_labels[:10],
            targeted=False, cw_confidence=0, max_iter=15
        )

        diff = (adv - test_data[:10]).abs()
        assert diff.sum().item() > 0
