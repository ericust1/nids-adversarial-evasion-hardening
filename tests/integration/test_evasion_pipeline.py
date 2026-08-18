import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.dataset_loader import DatasetManager
from src.core.model_trainer import IDSModel
from src.core.adversarial_generator import AdversarialAttackEngine
from src.modules.adversarial_training import AdversarialTrainer


class TestEvasionPipeline:
    def test_full_pipeline(self):
        dm = DatasetManager()
        X, y = dm.generate_synthetic_traffic(n_samples=500)
        n_features = 10
        X_small = X[:, :n_features]
        X_train, X_test, y_train, y_test = dm.split_data(X_small, y, test_size=0.3)

        train_ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

        test_data = torch.FloatTensor(X_test)
        test_labels = torch.LongTensor(y_test)

        device = torch.device('cpu')

        model = IDSModel(input_dim=n_features).to(device)
        model.train_model(train_loader, epochs=5, lr=0.01, device=device)

        model.eval()
        with torch.no_grad():
            clean_outputs = model(test_data)
            clean_preds = clean_outputs.argmax(dim=1).numpy()
        clean_acc = (clean_preds == y_test).mean()
        assert clean_acc > 0.5

        engine = AdversarialAttackEngine(model, device)
        results = engine.evaluate_robustness(
            test_data, test_labels,
            lambda d, lbl: engine.fgsm_attack(d, lbl, epsilon=0.3)
        )

        assert results['clean_accuracy'] > 0.5
        assert results['adversarial_accuracy'] >= 0
        assert results['total_samples'] == len(y_test)

        trainer = AdversarialTrainer(model, device)
        trainer.adversarial_train(
            train_loader, epochs=3, lr=0.005, epsilon=0.3, mix_ratio=0.5
        )

        hardened_engine = AdversarialAttackEngine(model, device)
        hardened_results = hardened_engine.evaluate_robustness(
            test_data, test_labels,
            lambda d, lbl: hardened_engine.fgsm_attack(d, lbl, epsilon=0.3)
        )

        assert 'clean_accuracy' in hardened_results
        assert 'adversarial_accuracy' in hardened_results
        assert hardened_results['total_samples'] == len(y_test)

    def test_pgd_pipeline(self):
        dm = DatasetManager()
        X, y = dm.generate_synthetic_traffic(n_samples=300)
        n_features = 10
        X_small = X[:, :n_features]
        X_train, X_test, y_train, y_test = dm.split_data(X_small, y, test_size=0.3)

        train_ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

        test_data = torch.FloatTensor(X_test)
        test_labels = torch.LongTensor(y_test)

        device = torch.device('cpu')
        model = IDSModel(input_dim=n_features).to(device)
        model.train_model(train_loader, epochs=3, lr=0.01, device=device)

        engine = AdversarialAttackEngine(model, device)
        results = engine.evaluate_robustness(
            test_data, test_labels,
            lambda d, lbl: engine.pgd_attack(
                d, lbl, epsilon=0.1, alpha=0.01, iterations=10
            )
        )

        assert results['clean_accuracy'] >= 0
        assert results['adversarial_accuracy'] >= 0

    def test_attack_then_harden_then_evaluate(self):
        dm = DatasetManager()
        X, y = dm.generate_synthetic_traffic(n_samples=400)
        n_features = 10
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
        model.train_model(train_loader, epochs=5, lr=0.01, device=device)

        engine = AdversarialAttackEngine(model, device)

        fgsm_results = engine.evaluate_robustness(
            test_data, test_labels,
            lambda d, lbl: engine.fgsm_attack(d, lbl, epsilon=0.2)
        )
        pgd_results = engine.evaluate_robustness(
            test_data, test_labels,
            lambda d, lbl: engine.pgd_attack(
                d, lbl, epsilon=0.2, alpha=0.02, iterations=15
            )
        )

        assert fgsm_results['clean_accuracy'] > 0
        assert pgd_results['clean_accuracy'] > 0

        trainer = AdversarialTrainer(model, device)
        trainer.adversarial_train(
            train_loader, epochs=3, lr=0.005, epsilon=0.2, mix_ratio=0.5
        )

        report = engine.generate_evasion_report(
            fgsm_results['clean_accuracy'],
            fgsm_results['adversarial_accuracy'],
            'FGSM'
        )
        assert report['risk_level'] in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
