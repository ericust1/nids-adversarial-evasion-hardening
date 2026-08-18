import sys
import os
import numpy as np
import torch
import pytest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.dataset_loader import DatasetManager
from src.core.model_trainer import IDSModel


class TestIDSModel:
    def _create_small_model_and_data(self, n_samples=200, n_features=10):
        dm = DatasetManager()
        X, y = dm.generate_synthetic_traffic(n_samples=n_samples)

        X_small = X[:, :n_features]
        X_train, X_test, y_train, y_test = dm.split_data(X_small, y, test_size=0.2)

        train_ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

        test_ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.LongTensor(y_test)
        )
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

        return train_loader, test_loader, n_features

    def test_model_creation(self):
        model = IDSModel(input_dim=10)
        assert isinstance(model, IDSModel)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_forward_pass_shape(self):
        model = IDSModel(input_dim=10)
        x = torch.randn(8, 10)
        output = model(x)
        assert output.shape == (8, 2)

    def test_train_for_two_epochs(self):
        train_loader, test_loader, n_features = self._create_small_model_and_data(
            n_samples=300, n_features=10
        )
        device = torch.device('cpu')
        model = IDSModel(input_dim=n_features).to(device)

        history = model.train_model(train_loader, epochs=2, lr=0.01, device=device)

        assert len(history['loss']) == 2
        assert len(history['accuracy']) == 2
        assert all(isinstance(l, float) for l in history['loss'])
        assert all(isinstance(a, float) for a in history['accuracy'])

    def test_accuracy_above_threshold(self):
        train_loader, test_loader, n_features = self._create_small_model_and_data(
            n_samples=500, n_features=10
        )
        device = torch.device('cpu')
        model = IDSModel(input_dim=n_features).to(device)

        model.train_model(train_loader, epochs=5, lr=0.01, device=device)
        metrics = model.evaluate(test_loader, device=device)

        assert metrics['accuracy'] > 0.5
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'confusion_matrix' in metrics

    def test_save_load_roundtrip(self):
        train_loader, test_loader, n_features = self._create_small_model_and_data(
            n_samples=200, n_features=10
        )
        device = torch.device('cpu')
        model = IDSModel(input_dim=n_features).to(device)
        model.train_model(train_loader, epochs=1, lr=0.01, device=device)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pt')
            model.save_model(model_path)
            assert os.path.exists(model_path)

            new_model = IDSModel(input_dim=n_features).to(device)
            new_model.load_model(model_path)

            x = torch.randn(5, n_features).to(device)
            model.eval()
            new_model.eval()

            with torch.no_grad():
                out1 = model(x)
                out2 = new_model(x)

            torch.testing.assert_close(out1, out2)

    def test_model_output_softmax_like(self):
        model = IDSModel(input_dim=10)
        model.eval()
        x = torch.randn(16, 10)
        with torch.no_grad():
            output = model(x)
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(16), atol=1e-5)

    def test_different_input_dims(self):
        for dim in [5, 10, 41, 100]:
            model = IDSModel(input_dim=dim)
            x = torch.randn(4, dim)
            output = model(x)
            assert output.shape == (4, 2)
